import React, { useState, useEffect } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { Code2, Eye, EyeOff, ArrowLeft } from 'lucide-react'

/* ── Shared input style (mirrors .form-input) ── */
const inputCls = [
  'w-full px-3.5 py-[11px]',
  'bg-[var(--bg-secondary)] border border-[var(--border-medium)]',
  'rounded-[var(--radius-md)] text-[14px] text-[var(--text-primary)]',
  'placeholder:text-[var(--text-muted)]',
  'transition-all duration-200 outline-none',
  'focus:border-[var(--accent-green)] focus:shadow-[0_0_0_3px_var(--accent-green-dim)]',
].join(' ')

export default function AuthPage() {
  const [params]                = useSearchParams()
  const [tab, setTab]           = useState(params.get('tab') || 'login')
  const [showPass, setShowPass] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [form, setForm]         = useState({ email: '', username: '', password: '' })
  const { login, register }     = useAuth()
  const navigate                = useNavigate()

  useEffect(() => {
    setTab(params.get('tab') || 'login')
  }, [params])

  const handleChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const handleSubmit = async e => {
    e.preventDefault()
    setLoading(true)
    try {
      if (tab === 'login') {
        await login(form.username, form.password)
        toast.success('Welcome back!')
        navigate('/dashboard')
      } else {
        await register(form.email, form.username, form.password)
        toast.success('Account created! Please sign in.')
        setTab('login')
      }
    } catch (err) {
      toast.error(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center relative p-5">

      {/* .auth__bg — fixed fullscreen orbs */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* .auth__orb--1  400×400 green, top-right */}
        <div className="absolute w-[400px] h-[400px] rounded-full blur-[80px] opacity-10
                        bg-[var(--accent-green)] -top-[100px] -right-[50px]" />
        {/* .auth__orb--2  300×300 blue, bottom-left */}
        <div className="absolute w-[300px] h-[300px] rounded-full blur-[80px] opacity-10
                        bg-[var(--accent-blue)] bottom-[50px] -left-[50px]" />
      </div>

      {/* .auth__back */}
      <Link
        to="/"
        className="absolute top-6 left-6 z-[1] flex items-center gap-1.5
                   text-[13px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]
                   transition-colors duration-200"
      >
        <ArrowLeft size={14} />
        Back to home
      </Link>

      {/* .auth__container */}
      <div
        className="relative z-[1] w-full max-w-[420px]"
        style={{ animation: 'fadeIn 0.4s ease' }}
      >
        {/* .auth__card */}
        <div className="bg-[var(--bg-card)] border border-[var(--border-medium)]
                        rounded-[var(--radius-xl)] p-9 shadow-[var(--shadow-card)]">

          {/* .auth__logo */}
          <div className="flex items-center justify-center gap-2.5 mb-7
                          font-[var(--font-display)] font-bold text-[18px] text-[var(--text-primary)]">
            {/* .auth__logo-icon */}
            <div className="w-9 h-9 flex items-center justify-center shrink-0
                            rounded-[var(--radius-sm)] bg-[var(--accent-green-dim)]
                            border border-[var(--border-accent)] text-[var(--accent-green)]">
              <Code2 size={20} />
            </div>
            <span>CP Roadmap</span>
          </div>

          {/* .auth__tabs */}
          <div className="flex gap-1 bg-[var(--bg-secondary)] p-1 rounded-[var(--radius-md)] mb-7">
            {[
              { id: 'login',    label: 'Sign In'        },
              { id: 'register', label: 'Create Account' },
            ].map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={[
                  'flex-1 px-4 py-2 text-[14px] font-medium transition-all duration-200',
                  'rounded-[var(--radius-sm)]',
                  tab === id
                    ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-[0_1px_4px_rgba(0,0,0,0.3)]'
                    : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]',
                ].join(' ')}
              >
                {label}
              </button>
            ))}
          </div>

          {/* .auth__form */}
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">

            {/* Email — register only */}
            {tab === 'register' && (
              <div className="flex flex-col gap-2">
                <label className="text-[13px] font-medium text-[var(--text-secondary)]">Email</label>
                <input
                  type="email" name="email" required autoComplete="email"
                  placeholder="you@example.com"
                  value={form.email} onChange={handleChange}
                  className={inputCls}
                />
              </div>
            )}

            {/* Username */}
            <div className="flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[var(--text-secondary)]">Username</label>
              <input
                type="text" name="username" required autoComplete="username"
                placeholder="your_username"
                value={form.username} onChange={handleChange}
                className={inputCls}
              />
            </div>

            {/* Password */}
            <div className="flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[var(--text-secondary)]">Password</label>
              {/* .form-input-wrap */}
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'} name="password"
                  required minLength={6} placeholder="••••••••"
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  value={form.password} onChange={handleChange}
                  className={`${inputCls} pr-11`}
                />
                {/* .form-input-icon */}
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2
                             bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]
                             p-1 transition-colors duration-200"
                >
                  {showPass ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {/* .auth__submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full mt-1 py-3 flex items-center justify-center gap-2
                         rounded-[var(--radius-md)] text-[15px] font-semibold font-[var(--font-body)]
                         bg-[var(--accent-green)] text-[#080c14]
                         transition-all duration-200
                         hover:enabled:bg-[#00d48c]
                         hover:enabled:shadow-[0_0_20px_rgba(0,245,160,0.25)]
                         disabled:opacity-70 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  {/* .btn-spinner */}
                  <div className="w-4 h-4 rounded-full
                                  border-2 border-[rgba(8,12,20,0.3)] border-t-[#080c14]
                                  animate-[spin_0.7s_linear_infinite]" />
                  {tab === 'login' ? 'Signing in...' : 'Creating account...'}
                </>
              ) : (
                tab === 'login' ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          {/* .auth__switch */}
          <p className="text-center text-[13px] text-[var(--text-muted)] mt-5">
            {tab === 'login' ? "Don't have an account? " : 'Already have an account? '}
            {/* .auth__switch-btn */}
            <button
              onClick={() => setTab(tab === 'login' ? 'register' : 'login')}
              className="bg-transparent p-0 text-[13px] font-medium
                         text-[var(--accent-green)] hover:text-[#00d48c]
                         underline underline-offset-2 transition-colors duration-200"
            >
              {tab === 'login' ? 'Sign up' : 'Sign in'}
            </button>
          </p>

        </div>
      </div>
    </div>
  )
}