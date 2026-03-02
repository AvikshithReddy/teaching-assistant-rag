import React, { ReactNode } from 'react'

type Type = 'error' | 'success' | 'info'

const styles: Record<Type, string> = {
  error:   'bg-red-50 text-red-700 border-red-200',
  success: 'bg-green-50 text-green-700 border-green-200',
  info:    'bg-primary-50 text-primary-800 border-primary-200',
}

export function Alert({ type = 'info', children, className = '' }: { type?: Type; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${styles[type]} ${className}`}>{children}</div>
  )
}
