import React, { useRef, useState } from 'react'
import { Material, materialApi } from '../../api/client'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'
import { Alert } from '../ui/Alert'

interface Props {
  courseId: string
  materials: Material[]
  onRefresh: () => void
}

export function MaterialUpload({ courseId, materials, onRefresh }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [dragging, setDragging] = useState(false)

  function addFiles(newFiles: File[]) {
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name))
      return [...prev, ...newFiles.filter((f) => !names.has(f.name))]
    })
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  async function upload() {
    if (!files.length) return
    setUploading(true)
    setMessage(null)
    try {
      const { data } = await materialApi.upload(courseId, files)
      setMessage({ type: 'success', text: `Uploaded ${data.uploaded}, skipped ${data.skipped}.` })
      setFiles([])
      onRefresh()
    } catch (err: any) {
      setMessage({ type: 'error', text: err.response?.data?.detail ?? 'Upload failed.' })
    } finally {
      setUploading(false)
    }
  }

  async function deleteMat(filename: string) {
    try {
      await materialApi.remove(courseId, filename)
      onRefresh()
    } catch {
      alert('Failed to delete.')
    }
  }

  const typeColor = (fname: string) =>
    fname.toLowerCase().endsWith('.pdf') ? 'blue' : 'teal'

  return (
    <div className="space-y-4">
      {message && <Alert type={message.type}>{message.text}</Alert>}

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging ? 'border-primary bg-primary-50' : 'border-gray-300 hover:border-primary'
        }`}
      >
        <p className="text-sm text-gray-500">Drag & drop PDF / PPTX files here, or click to browse</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.pptx"
          multiple
          className="hidden"
          onChange={(e) => addFiles(Array.from(e.target.files ?? []))}
        />
      </div>

      {files.length > 0 && (
        <div className="space-y-1">
          {files.map((f) => (
            <div key={f.name} className="flex items-center justify-between text-sm">
              <span className="text-gray-700 truncate">{f.name}</span>
              <button
                onClick={() => setFiles((prev) => prev.filter((x) => x.name !== f.name))}
                className="ml-2 text-red-400 hover:text-red-600"
              >
                &times;
              </button>
            </div>
          ))}
          <Button onClick={upload} loading={uploading} className="mt-2">
            Upload {files.length} file{files.length > 1 ? 's' : ''}
          </Button>
        </div>
      )}

      {/* Existing materials */}
      <div>
        <h4 className="text-sm font-semibold text-gray-600 mb-2">Uploaded Files</h4>
        {materials.length === 0 ? (
          <p className="text-sm text-gray-400">No files yet.</p>
        ) : (
          <ul className="space-y-2">
            {materials.map((m) => (
              <li key={m.filename} className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Badge label={m.filename.split('.').pop()?.toUpperCase() ?? '?'} color={typeColor(m.filename)} />
                  <span className="text-sm text-gray-700 truncate">{m.filename}</span>
                </div>
                <button
                  onClick={() => deleteMat(m.filename)}
                  className="ml-2 text-sm text-red-400 hover:text-red-600 shrink-0"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
