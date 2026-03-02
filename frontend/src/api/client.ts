import axios from 'axios'

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export const api = axios.create({ baseURL: BASE_URL })

// Attach JWT on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// On 401 → clear auth and redirect
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  },
)

// ---------- Types ----------

export interface User {
  id: string
  email: string
  name: string
  role: 'professor' | 'student'
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export interface Course {
  prof_id: string
  prof_name?: string
  course_id: string
  course_title: string
}

export interface Material {
  prof_id: string
  course_id: string
  filename: string
  stored_path: string
  uploaded_at: string
}

export interface IndexStatus {
  indexed: boolean
  chunk_count: number
}

export interface Source {
  doc_name: string
  source_type: string
  page_or_slide: number
  score: number
}

export interface AskResponse {
  answer: string
  sources: Source[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

// ---------- Auth ----------

export const authApi = {
  register: (body: { email: string; password: string; name: string; role: string }) =>
    api.post<AuthResponse>('/auth/register', body),
  login: (body: { email: string; password: string }) =>
    api.post<AuthResponse>('/auth/login', body),
  me: () => api.get<User>('/auth/me'),
}

// ---------- Professor: Courses ----------

export const courseApi = {
  list: () => api.get<Course[]>('/api/professor/courses'),
  create: (body: { course_id: string; course_title: string }) =>
    api.post<Course>('/api/professor/courses', body),
  remove: (courseId: string) => api.delete(`/api/professor/courses/${courseId}`),
  listForStudent: () => api.get<Course[]>('/api/student/courses'),
}

// ---------- Professor: Materials ----------

export const materialApi = {
  list: (courseId: string) => api.get<Material[]>(`/api/professor/courses/${courseId}/materials`),
  upload: (courseId: string, files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return api.post<{ uploaded: number; skipped: number }>(
      `/api/professor/courses/${courseId}/materials`,
      form,
    )
  },
  remove: (courseId: string, filename: string) =>
    api.delete(`/api/professor/courses/${courseId}/materials/${encodeURIComponent(filename)}`),
}

// ---------- Professor: Index ----------

export const indexApi = {
  rebuild: (courseId: string) =>
    api.post<{ status: string }>(`/api/professor/courses/${courseId}/rebuild-index`),
  status: (courseId: string) =>
    api.get<IndexStatus>(`/api/professor/courses/${courseId}/index-status`),
}

// ---------- Student: Chat ----------

export const chatApi = {
  history: (courseId: string, profId: string) =>
    api.get<ChatMessage[]>(`/api/student/courses/${courseId}/history`, {
      params: { prof_id: profId },
    }),
  ask: (courseId: string, body: { question: string; prof_id: string; top_k?: number }) =>
    api.post<AskResponse>(`/api/student/courses/${courseId}/ask`, body),
}
