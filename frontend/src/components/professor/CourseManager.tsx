import React, { useState } from 'react'
import { Course, courseApi } from '../../api/client'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Alert } from '../ui/Alert'

interface Props {
  courses: Course[]
  onRefresh: () => void
}

export function CourseManager({ courses, onRefresh }: Props) {
  const [courseId, setCourseId] = useState('')
  const [courseTitle, setCourseTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setSuccess('')
    if (!courseId.trim() || !courseTitle.trim()) {
      setError('Both fields are required.')
      return
    }
    setLoading(true)
    try {
      await courseApi.create({ course_id: courseId.trim(), course_title: courseTitle.trim() })
      setSuccess(`Course "${courseTitle}" created.`)
      setCourseId('')
      setCourseTitle('')
      onRefresh()
    } catch (err: any) {
      setError(err.response?.data?.detail ?? 'Failed to create course.')
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete(cid: string) {
    if (!confirm(`Delete course "${cid}"? This also removes all materials and the index.`)) return
    try {
      await courseApi.remove(cid)
      onRefresh()
    } catch {
      alert('Failed to delete course.')
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleCreate} className="space-y-3">
        <h3 className="font-semibold text-gray-700">Create New Course</h3>
        {error && <Alert type="error">{error}</Alert>}
        {success && <Alert type="success">{success}</Alert>}
        <Input
          placeholder="Course ID (e.g. CS101)"
          value={courseId}
          onChange={(e) => setCourseId(e.target.value)}
        />
        <Input
          placeholder="Course Title"
          value={courseTitle}
          onChange={(e) => setCourseTitle(e.target.value)}
        />
        <Button type="submit" loading={loading}>Create Course</Button>
      </form>

      <div>
        <h3 className="font-semibold text-gray-700 mb-2">Your Courses</h3>
        {courses.length === 0 ? (
          <p className="text-sm text-gray-500">No courses yet.</p>
        ) : (
          <ul className="space-y-2">
            {courses.map((c) => (
              <li key={c.course_id} className="flex items-center justify-between rounded-lg border border-gray-100 bg-gray-50 px-4 py-3">
                <div>
                  <span className="font-medium text-gray-800">{c.course_id}</span>
                  <span className="ml-2 text-sm text-gray-500">— {c.course_title}</span>
                </div>
                <Button variant="danger" onClick={() => handleDelete(c.course_id)} className="text-xs px-2 py-1">
                  Delete
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
