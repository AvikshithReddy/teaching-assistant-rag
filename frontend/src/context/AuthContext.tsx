import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { authApi, User } from '../api/client'

interface AuthContextValue {
  user: User | null
  token: string | null
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, name: string, role: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const [user, setUser] = useState<User | null>(() => {
    const stored = localStorage.getItem('user')
    return stored ? JSON.parse(stored) : null
  })
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('token'))

  function _persist(tok: string, u: User) {
    localStorage.setItem('token', tok)
    localStorage.setItem('user', JSON.stringify(u))
    setToken(tok)
    setUser(u)
  }

  async function login(email: string, password: string) {
    const { data } = await authApi.login({ email, password })
    _persist(data.access_token, data.user)
    navigate(data.user.role === 'professor' ? '/professor' : '/student')
  }

  async function register(email: string, password: string, name: string, role: string) {
    const { data } = await authApi.register({ email, password, name, role })
    _persist(data.access_token, data.user)
    navigate(data.user.role === 'professor' ? '/professor' : '/student')
  }

  function logout() {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setToken(null)
    setUser(null)
    navigate('/login')
  }

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}

export function AuthGuard({ role, children }: { role?: string; children: ReactNode }) {
  const { user } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    if (!user) {
      navigate('/login')
    } else if (role && user.role !== role) {
      navigate(user.role === 'professor' ? '/professor' : '/student')
    }
  }, [user, role, navigate])

  if (!user || (role && user.role !== role)) return null
  return <>{children}</>
}
