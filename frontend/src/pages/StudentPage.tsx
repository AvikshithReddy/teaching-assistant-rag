import React, { useEffect, useState } from 'react'
import { useAuth, AuthGuard } from '../context/AuthContext'
import { Course, courseApi } from '../api/client'
import { CourseSelector } from '../components/student/CourseSelector'
import { ChatWindow } from '../components/student/ChatWindow'
import { Button } from '../components/ui/Button'
import { Spinner } from '../components/ui/Spinner'

export function StudentPage() {
  return (
    <AuthGuard role="student">
      <StudentContent />
    </AuthGuard>
  )
}

function StudentContent() {
  const { user, logout } = useAuth()
  const [courses, setCourses] = useState<Course[]>([])
  const [selected, setSelected] = useState<Course | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    courseApi.listForStudent().then(({ data }) => {
      setCourses(data)
      if (data.length > 0) setSelected(data[0])
    }).finally(() => setLoading(false))
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans">
      {/* Top bar */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-4 shrink-0">
        <h1 className="text-lg font-bold text-primary">AI Teaching Assistant</h1>
        <span className="text-sm text-gray-400">|</span>
        {loading ? (
          <Spinner size="sm" />
        ) : (
          <CourseSelector courses={courses} selected={selected} onChange={setSelected} />
        )}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-sm text-gray-600">{user?.name}</span>
          <Button variant="ghost" onClick={logout} className="text-sm">Logout</Button>
        </div>
      </header>

      {/* Chat area */}
      <div className="flex-1 overflow-hidden">
        {!selected ? (
          <div className="flex h-full items-center justify-center text-gray-400 text-sm">
            {courses.length === 0
              ? 'No courses available yet. Ask your professor to create a course.'
              : 'Select a course to start chatting.'}
          </div>
        ) : (
          <ChatWindow courseId={selected.course_id} profId={selected.prof_id} />
        )}
      </div>
    </div>
  )
}
