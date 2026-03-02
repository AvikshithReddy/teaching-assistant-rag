import React, { useCallback, useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { AuthGuard } from '../context/AuthContext'
import { Course, Material, courseApi, materialApi } from '../api/client'
import { CourseManager } from '../components/professor/CourseManager'
import { MaterialUpload } from '../components/professor/MaterialUpload'
import { IndexStatus } from '../components/professor/IndexStatus'
import { Button } from '../components/ui/Button'
import { Spinner } from '../components/ui/Spinner'

type Tab = 'materials' | 'index'

export function ProfessorPage() {
  return (
    <AuthGuard role="professor">
      <ProfessorContent />
    </AuthGuard>
  )
}

function ProfessorContent() {
  const { user, logout } = useAuth()
  const [courses, setCourses] = useState<Course[]>([])
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null)
  const [materials, setMaterials] = useState<Material[]>([])
  const [tab, setTab] = useState<Tab>('materials')
  const [loadingCourses, setLoadingCourses] = useState(true)
  const [showCourseManager, setShowCourseManager] = useState(false)

  const fetchCourses = useCallback(async () => {
    try {
      const { data } = await courseApi.list()
      setCourses(data)
      if (data.length > 0 && !selectedCourse) setSelectedCourse(data[0])
    } finally {
      setLoadingCourses(false)
    }
  }, [])

  const fetchMaterials = useCallback(async () => {
    if (!selectedCourse) return
    try {
      const { data } = await materialApi.list(selectedCourse.course_id)
      setMaterials(data)
    } catch {
      setMaterials([])
    }
  }, [selectedCourse])

  useEffect(() => { fetchCourses() }, [fetchCourses])
  useEffect(() => { fetchMaterials() }, [fetchMaterials])

  function selectCourse(c: Course) {
    setSelectedCourse(c)
    setTab('materials')
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-100">
          <h1 className="text-lg font-bold text-primary">Teaching Assistant</h1>
          <p className="text-xs text-gray-500 mt-0.5">{user?.name}</p>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="flex items-center justify-between px-2 mb-2">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Courses</span>
            <button
              onClick={() => setShowCourseManager((v) => !v)}
              className="text-primary text-lg font-bold leading-none hover:text-primary-700"
              title="Manage courses"
            >
              +
            </button>
          </div>

          {loadingCourses ? (
            <div className="flex justify-center pt-4"><Spinner size="sm" /></div>
          ) : courses.length === 0 ? (
            <p className="text-xs text-gray-400 px-2">No courses yet. Click + to create one.</p>
          ) : (
            courses.map((c) => (
              <button
                key={c.course_id}
                onClick={() => selectCourse(c)}
                className={`w-full text-left rounded-lg px-3 py-2 text-sm transition-colors ${
                  selectedCourse?.course_id === c.course_id
                    ? 'bg-primary text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <div className="font-medium">{c.course_id}</div>
                <div className={`text-xs ${selectedCourse?.course_id === c.course_id ? 'text-primary-100' : 'text-gray-400'}`}>
                  {c.course_title}
                </div>
              </button>
            ))
          )}
        </div>

        <div className="p-3 border-t border-gray-100">
          <Button variant="ghost" onClick={logout} className="w-full justify-start text-gray-500">
            Logout
          </Button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {showCourseManager ? (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Manage Courses</h2>
              <Button variant="ghost" onClick={() => setShowCourseManager(false)}>Close</Button>
            </div>
            <CourseManager courses={courses} onRefresh={() => { fetchCourses(); setShowCourseManager(false) }} />
          </div>
        ) : selectedCourse ? (
          <>
            <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
              <div>
                <h2 className="font-semibold text-gray-800">{selectedCourse.course_id}</h2>
                <p className="text-sm text-gray-500">{selectedCourse.course_title}</p>
              </div>
              <div className="ml-auto flex gap-2">
                {(['materials', 'index'] as Tab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      tab === t ? 'bg-primary text-white' : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              {tab === 'materials' && (
                <MaterialUpload
                  courseId={selectedCourse.course_id}
                  materials={materials}
                  onRefresh={fetchMaterials}
                />
              )}
              {tab === 'index' && (
                <IndexStatus courseId={selectedCourse.course_id} />
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            Select or create a course to get started.
          </div>
        )}
      </main>
    </div>
  )
}
