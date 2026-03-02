import React, { useState } from 'react'
import { Source } from '../../api/client'
import { Badge } from '../ui/Badge'

export function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)

  if (!sources.length) return null

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-xs font-medium text-primary underline-offset-2 hover:underline"
      >
        {open ? 'Hide' : 'Show'} {sources.length} source{sources.length > 1 ? 's' : ''}
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <li key={i} className="rounded-lg border border-gray-100 bg-gray-50 p-3">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <span className="text-sm font-medium text-gray-800">{s.doc_name}</span>
                <Badge
                  label={s.source_type?.toUpperCase() ?? '?'}
                  color={s.source_type?.toLowerCase() === 'pdf' ? 'blue' : 'teal'}
                />
                <span className="text-xs text-gray-500">p.{s.page_or_slide}</span>
              </div>
              {/* Score bar */}
              <div className="h-1.5 w-full rounded-full bg-gray-200 overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${Math.min(100, Math.round(s.score * 100))}%` }}
                />
              </div>
              <span className="text-xs text-gray-400">score {s.score?.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
