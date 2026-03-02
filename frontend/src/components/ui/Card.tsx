import React, { ReactNode } from 'react'

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl bg-white shadow-sm border border-gray-100 ${className}`}>
      {children}
    </div>
  )
}
