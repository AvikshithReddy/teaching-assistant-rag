import React, { useEffect, useState } from 'react'
import { IndexStatus as IStatus, indexApi } from '../../api/client'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'
import { Spinner } from '../ui/Spinner'
import { Alert } from '../ui/Alert'

export function IndexStatus({ courseId }: { courseId: string }) {
  const [status, setStatus] = useState<IStatus | null>(null)
  const [rebuilding, setRebuilding] = useState(false)
  const [message, setMessage] = useState('')

  async function fetchStatus() {
    try {
      const { data } = await indexApi.status(courseId)
      setStatus(data)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [courseId])

  async function rebuild() {
    setRebuilding(true)
    setMessage('')
    try {
      await indexApi.rebuild(courseId)
      setMessage('Index rebuild started. Refresh status in a moment.')
      setTimeout(fetchStatus, 5000)
    } catch {
      setMessage('Failed to start rebuild.')
    } finally {
      setRebuilding(false)
    }
  }

  return (
    <div className="space-y-4">
      {message && <Alert type="info">{message}</Alert>}
      <div className="flex items-center gap-4">
        {status === null ? (
          <Spinner />
        ) : (
          <>
            <Badge
              label={status.indexed ? `Indexed — ${status.chunk_count} chunks` : 'Not indexed'}
              color={status.indexed ? 'green' : 'red'}
            />
            <Button onClick={fetchStatus} variant="ghost" className="text-xs">Refresh</Button>
          </>
        )}
      </div>
      <Button onClick={rebuild} loading={rebuilding}>
        Rebuild Index
      </Button>
    </div>
  )
}
