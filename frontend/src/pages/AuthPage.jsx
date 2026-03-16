import React, { useState, useEffect } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'
import { Code2, Eye, EyeOff, ArrowLeft, Zap } from 'lucide-react'

const inputCls = [
  'w-full px-3.5 py-[11px]',
  'bg-[var(--bg-secondary)] border border-[var(--border-medium)]',
  'rounded-[var(--radius-md)] text-[14px] text-[var(--text-primary)]',
  'placeholder:text-[var(--text-muted)]',
  'transition-all duration-300 outline-none',
  'focus:border-[var(--accent-green)] focus:shadow-[0_0_0_3px_var(--accent-green-dim)]',
  'hover:border-[rgba(255,255,255,0.2)]',
].join(' ')

export default function AuthPage() {
  const [params]                = useSearchParams()
  const [tab, setTab]           = useState(params.get('tab') || 'login')
  const [showPass, setShowPass] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [form, setForm]         = useState({ email: '', username: '', password: '' })
  const [mounted, setMounted]   = useState(false)
  const { login, register }     = useAuth()
  const navigate                = useNavigate()

  useEffect(() => { setTimeout(() => setMounted(true), 50) }, [])
  useEffect(() => { setTab(params.get('tab') || 'login') }, [params])

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
    <div className="min-h-screen flex flex-col items-center justify-center relative p-5 overflow-hidden">

      {/* Animated background orbs */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute w-[500px] h-[500px] rounded-full opacity-[0.08]
                        bg-[var(--accent-green)] -top-[150px] -right-[100px]
                        blur-[120px] animate-[drift1_8s_ease-in-out_infinite]" />
        <div className="absolute w-[400px] h-[400px] rounded-full opacity-[0.07]
                        bg-[var(--accent-blue)] bottom-[0px] -left-[80px]
                        blur-[100px] animate-[drift2_10s_ease-in-out_infinite]" />
        <div className="absolute w-[250px] h-[250px] rounded-full opacity-[0.05]
                        bg-[var(--accent-purple)] top-[40%] left-[60%]
                        blur-[80px] animate-[drift1_12s_ease-in-out_infinite_reverse]" />
        {/* Scanline effect */}
        <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(0,0,0,0.03)_2px,rgba(0,0,0,0.03)_4px)]
                        pointer-events-none" />
      </div>

      <style>{`
        @keyframes drift1 {
          0%,100% { transform: translate(0,0) scale(1); }
          33% { transform: translate(30px,-20px) scale(1.05); }
          66% { transform: translate(-20px,30px) scale(0.95); }
        }
        @keyframes drift2 {
          0%,100% { transform: translate(0,0) scale(1); }
          33% { transform: translate(-25px,20px) scale(1.08); }
          66% { transform: translate(35px,-15px) scale(0.92); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(24px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
          0%,100% { box-shadow: 0 0 20px rgba(0,245,160,0.15); }
          50% { box-shadow: 0 0 40px rgba(0,245,160,0.35), 0 0 80px rgba(0,245,160,0.1); }
        }
        @keyframes borderGlow {
          0%,100% { border-color: rgba(0,245,160,0.3); }
          50% { border-color: rgba(0,245,160,0.7); }
        }
        .auth-card-enter { animation: slideUp 0.5s cubic-bezier(0.16,1,0.3,1) both; }
        .back-link-enter { animation: slideDown 0.4s cubic-bezier(0.16,1,0.3,1) 0.1s both; }
        .form-field { animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1) both; }
        .field-1 { animation-delay: 0.15s; }
        .field-2 { animation-delay: 0.22s; }
        .field-3 { animation-delay: 0.29s; }
        .field-4 { animation-delay: 0.36s; }
        .logo-icon-glow { animation: glowPulse 3s ease-in-out infinite; }
        .tab-active-glow { animation: borderGlow 2s ease-in-out infinite; }
        input:focus + .input-glow-line { width: 100%; }
        .submit-btn:hover { transform: translateY(-2px); }
        .submit-btn:active { transform: translateY(0); }
      `}</style>

      {/* Back link */}
      <Link
        to="/"
        className="back-link-enter absolute top-6 left-6 z-[1] flex items-center gap-1.5
                   text-[13px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]
                   transition-all duration-200 group"
      >
        <ArrowLeft size={14} className="group-hover:-translate-x-1 transition-transform duration-200" />
        Back to home
      </Link>

      {/* Card */}
      <div className={`auth-card-enter relative z-[1] w-full max-w-[420px]`}>
        <div className="bg-[var(--bg-card)] border border-[var(--border-medium)]
                        rounded-[var(--radius-xl)] p-9 shadow-[var(--shadow-card)]
                        relative overflow-hidden">

          {/* Card top glow line */}
          <div className="absolute top-0 left-[10%] right-[10%] h-[1px]
                          bg-gradient-to-r from-transparent via-[var(--accent-green)] to-transparent
                          opacity-40" />

          {/* Logo */}
          <div className="flex items-center justify-center gap-2.5 mb-7
                          font-[var(--font-display)] font-bold text-[18px] text-[var(--text-primary)]">
            <div className="logo-icon-glow w-9 h-9 flex items-center justify-center shrink-0
                            rounded-[var(--radius-sm)] bg-[var(--accent-green-dim)]
                            border border-[var(--border-accent)] text-[var(--accent-green)]
                            transition-transform duration-300 hover:scale-110">
              <Code2 size={20} />
            </div>
            <span>CP Roadmap</span>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 bg-[var(--bg-secondary)] p-1 rounded-[var(--radius-md)] mb-7">
            {[{ id: 'login', label: 'Sign In' }, { id: 'register', label: 'Create Account' }].map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={[
                  'flex-1 px-4 py-2 text-[14px] font-medium transition-all duration-300',
                  'rounded-[var(--radius-sm)] relative',
                  tab === id
                    ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-[0_1px_4px_rgba(0,0,0,0.3)]'
                    : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]',
                ].join(' ')}
              >
                {tab === id && (
                  <span className="absolute inset-x-[20%] bottom-0 h-[2px] rounded-full
                                   bg-[var(--accent-green)] opacity-60" />
                )}
                {label}
              </button>
            ))}
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            {tab === 'register' && (
              <div className="form-field field-1 flex flex-col gap-2">
                <label className="text-[13px] font-medium text-[var(--text-secondary)]">Email</label>
                <input type="email" name="email" required autoComplete="email"
                  placeholder="you@example.com" value={form.email} onChange={handleChange}
                  className={inputCls} />
              </div>
            )}

            <div className="form-field field-2 flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[var(--text-secondary)]">Username</label>
              <input type="text" name="username" required autoComplete="username"
                placeholder="your_username" value={form.username} onChange={handleChange}
                className={inputCls} />
            </div>

            <div className="form-field field-3 flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[var(--text-secondary)]">Password</label>
              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'} name="password"
                  required minLength={6} placeholder="••••••••"
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  value={form.password} onChange={handleChange}
                  className={`${inputCls} pr-11`}
                />
                <button type="button" onClick={() => setShowPass(!showPass)}
                  className="absolute right-3 top-1/2 -translate-y-1/2
                             bg-transparent text-[var(--text-muted)] hover:text-[var(--accent-green)]
                             p-1 transition-all duration-200 hover:scale-110">
                  {showPass ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            <div className="form-field field-4">
              <button
                type="submit" disabled={loading}
                className="submit-btn w-full mt-1 py-3 flex items-center justify-center gap-2
                           rounded-[var(--radius-md)] text-[15px] font-semibold font-[var(--font-body)]
                           bg-[var(--accent-green)] text-[#080c14]
                           transition-all duration-300 relative overflow-hidden group
                           hover:bg-[#00d48c] hover:shadow-[0_0_30px_rgba(0,245,160,0.4)]
                           disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none">
                {/* Shimmer on hover */}
                <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent
                                 -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                {loading ? (
                  <>
                    <div className="w-4 h-4 rounded-full border-2 border-[rgba(8,12,20,0.3)] border-t-[#080c14]
                                    animate-[spin_0.7s_linear_infinite]" />
                    {tab === 'login' ? 'Signing in...' : 'Creating account...'}
                  </>
                ) : (
                  <>
                    <Zap size={15} className="relative z-[1]" />
                    <span className="relative z-[1]">{tab === 'login' ? 'Sign In' : 'Create Account'}</span>
                  </>
                )}
              </button>
            </div>
          </form>

          <p className="text-center text-[13px] text-[var(--text-muted)] mt-5">
            {tab === 'login' ? "Don't have an account? " : 'Already have an account? '}
            <button onClick={() => setTab(tab === 'login' ? 'register' : 'login')}
              className="bg-transparent p-0 text-[13px] font-medium
                         text-[var(--accent-green)] hover:text-[#00d48c]
                         underline underline-offset-2 transition-colors duration-200">
              {tab === 'login' ? 'Sign up' : 'Sign in'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}