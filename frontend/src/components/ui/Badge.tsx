import React from 'react'

type Color = 'teal' | 'blue' | 'gray' | 'green' | 'red'

const colors: Record<Color, string> = {
  teal:  'bg-primary-50 text-primary-700',
  blue:  'bg-blue-50 text-blue-700',
  gray:  'bg-gray-100 text-gray-600',
  green: 'bg-green-50 text-green-700',
  red:   'bg-red-50 text-red-600',
}

export function Badge({ label, color = 'teal' }: { label: string; color?: Color }) {
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${colors[color]}`}>
      {label}
    </span>
  )
}
