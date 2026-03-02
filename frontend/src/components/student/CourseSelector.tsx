import React from 'react'
import { Course } from '../../api/client'

interface Props {
  courses: Course[]
  selected: Course | null
  onChange: (course: Course) => void
}

export function CourseSelector({ courses, selected, onChange }: Props) {
  return (
    <select
      value={selected ? `${selected.prof_id}::${selected.course_id}` : ''}
      onChange={(e) => {
        const [profId, courseId] = e.target.value.split('::')
        const c = courses.find((x) => x.prof_id === profId && x.course_id === courseId)
        if (c) onChange(c)
      }}
      className="rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 bg-white"
    >
      <option value="">Select a course...</option>
      {courses.map((c) => (
        <option key={`${c.prof_id}::${c.course_id}`} value={`${c.prof_id}::${c.course_id}`}>
          {c.course_id} — {c.course_title} (Prof. {c.prof_name ?? c.prof_id})
        </option>
      ))}
    </select>
  )
}
