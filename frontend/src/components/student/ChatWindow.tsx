import React, { useEffect, useRef, useState } from 'react'
import { ChatMessage, Source, chatApi } from '../../api/client'
import { Spinner } from '../ui/Spinner'
import { SourceList } from './SourceCard'

interface ExtendedMessage extends ChatMessage {
  sources?: Source[]
}

interface Props {
  courseId: string
  profId: string
}

export function ChatWindow({ courseId, profId }: Props) {
  const [messages, setMessages] = useState<ExtendedMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [historyLoaded, setHistoryLoaded] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Load history whenever course/prof changes
  useEffect(() => {
    setMessages([])
    setHistoryLoaded(false)
    if (!courseId || !profId) return
    chatApi.history(courseId, profId).then(({ data }) => {
      setMessages(data ?? [])
      setHistoryLoaded(true)
    }).catch(() => setHistoryLoaded(true))
  }, [courseId, profId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const { data } = await chatApi.ask(courseId, { question: q, prof_id: profId })
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {!historyLoaded && (
          <div className="flex justify-center pt-8"><Spinner /></div>
        )}
        {historyLoaded && messages.length === 0 && (
          <p className="text-center text-sm text-gray-400 pt-8">
            Ask a question about your course materials.
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {m.role === 'user' ? (
              <div className="max-w-[70%] rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-sm text-white shadow-sm">
                {m.content}
              </div>
            ) : (
              <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-white px-4 py-3 text-sm text-gray-800 shadow-sm border border-gray-100">
                <p className="whitespace-pre-wrap">{m.content}</p>
                {m.sources && <SourceList sources={m.sources} />}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-white px-4 py-3 shadow-sm border border-gray-100">
              <Spinner size="sm" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-gray-200 bg-white p-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask a question from your course materials..."
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-2 focus:ring-primary/20"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50 transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  )
}
