import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../utils/api'
import toast from 'react-hot-toast'
import { Zap, Code2, Trophy, Clock, ChevronRight, CheckCircle, Loader, Sparkles } from 'lucide-react'

const STEPS = [
  'Fetching your submission history...',
  'Analyzing contest performance...',
  'Running forgetting curve model...',
  'Detecting hidden knowledge gaps (GNN)...',
  'Ranking problems by ML priority...',
  'Building your 7-day calendar...',
  'Finalizing your roadmap...',
]

export default function GeneratePage() {
  const [form, setForm]       = useState({ leetcode_username: '', codeforces_handle: '', session_hours: 3 })
  const [loading, setLoading] = useState(false)
  const [step, setStep]       = useState(0)
  const navigate              = useNavigate()

  const handleChange = e => {
    const { name, value } = e.target
    setForm(f => ({ ...f, [name]: name === 'session_hours' ? Number(value) : value }))
  }

  const handleSubmit = async e => {
    e.preventDefault()
    if (!form.leetcode_username && !form.codeforces_handle) {
      toast.error('Please enter at least one profile handle.')
      return
    }
    setLoading(true)
    setStep(0)
    const interval = setInterval(() => setStep(s => s < STEPS.length - 1 ? s + 1 : s), 4000)
    try {
      const payload = { session_hours: form.session_hours }
      if (form.leetcode_username) payload.leetcode_username = form.leetcode_username
      if (form.codeforces_handle) payload.codeforces_handle = form.codeforces_handle
      const res = await api.post('/roadmap/generate', payload)
      clearInterval(interval)
      toast.success('Roadmap generated!')
      navigate(`/roadmap/${res.data.id}`)
    } catch (err) {
      clearInterval(interval)
      toast.error(err.message)
      setLoading(false)
    }
  }

  return (
    <>
      <style>{`
        @keyframes slideUpFade {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes progressFill {
          from { width: 0; }
        }
        .g-enter { animation: slideUpFade 0.45s cubic-bezier(0.16,1,0.3,1) both; }
        .g1 { animation-delay: 0.05s; }
        .g2 { animation-delay: 0.12s; }
        .g3 { animation-delay: 0.19s; }
        .input-wrap-focus { transition: all 0.25s ease; }
        .input-wrap-focus:focus-within {
          box-shadow: 0 0 0 3px var(--accent-green-dim), 0 0 20px rgba(0,245,160,0.1);
          border-color: var(--accent-green) !important;
        }
        .gen-btn-main {
          position: relative; overflow: hidden;
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .gen-btn-main::after {
          content: '';
          position: absolute; inset: 0;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
          transform: translateX(-100%);
        }
        .gen-btn-main:hover::after { transform: translateX(100%); transition: transform 0.6s ease; }
        .gen-btn-main:hover { transform: translateY(-2px); box-shadow: 0 0 40px rgba(0,245,160,0.4); }
        .gen-btn-main:active { transform: translateY(0); }
        .info-card-hover {
          transition: all 0.25s ease;
        }
        .info-card-hover:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .step-done svg { animation: slideUpFade 0.3s ease both; }
        .progress-step {
          transition: all 0.4s cubic-bezier(0.16,1,0.3,1);
        }
        .slider-track {
          background: linear-gradient(to right, var(--accent-green) var(--pct), var(--bg-elevated) var(--pct));
        }
      `}</style>

      <div className="flex flex-col gap-7">
        <div className="g-enter g1 max-w-[600px]">
          <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)] mb-2">
            Generate Roadmap
          </h1>
          <p className="text-[15px] text-[var(--text-secondary)] leading-[1.6]">
            Connect your competitive programming profiles and let our ML pipeline build your personalized plan.
          </p>
        </div>

        {loading ? (
          <LoadingState step={step} />
        ) : (
          <div className="g-enter g2 grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start">

            {/* Form card */}
            <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-xl)] p-8
                            relative overflow-hidden">
              <div className="absolute top-0 left-[5%] right-[5%] h-[1px]
                              bg-gradient-to-r from-transparent via-[var(--accent-green)] to-transparent opacity-25" />

              <form onSubmit={handleSubmit} className="flex flex-col gap-5">
                <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.08em]
                                text-[var(--text-muted)] font-[var(--font-mono)]">
                  <Code2 size={14} /><span>Profile Handles</span>
                </div>

                {/* LeetCode input */}
                <div className="flex flex-col gap-2">
                  <label className="text-[13px] font-medium text-[var(--text-secondary)]">LeetCode Username</label>
                  <div className="input-wrap-focus flex items-center bg-[var(--bg-secondary)]
                                  border border-[var(--border-medium)] rounded-[var(--radius-md)] overflow-hidden">
                    <span className="px-3 py-[11px] text-[12px] font-[var(--font-mono)] text-[var(--text-muted)]
                                     bg-[var(--bg-elevated)] border-r border-[var(--border-subtle)] whitespace-nowrap
                                     select-none">
                      leetcode.com/
                    </span>
                    <input type="text" name="leetcode_username" placeholder="your_username"
                      value={form.leetcode_username} onChange={handleChange}
                      className="flex-1 px-3 py-[11px] text-[14px] text-[var(--text-primary)]
                                 bg-transparent border-none outline-none placeholder:text-[var(--text-muted)]" />
                  </div>
                </div>

                {/* Codeforces input */}
                <div className="flex flex-col gap-2">
                  <label className="text-[13px] font-medium text-[var(--text-secondary)]">Codeforces Handle</label>
                  <div className="input-wrap-focus flex items-center bg-[var(--bg-secondary)]
                                  border border-[var(--border-medium)] rounded-[var(--radius-md)] overflow-hidden">
                    <span className="px-3 py-[11px] text-[12px] font-[var(--font-mono)] text-[var(--text-muted)]
                                     bg-[var(--bg-elevated)] border-r border-[var(--border-subtle)] whitespace-nowrap
                                     select-none">
                      codeforces.com/
                    </span>
                    <input type="text" name="codeforces_handle" placeholder="your_handle"
                      value={form.codeforces_handle} onChange={handleChange}
                      className="flex-1 px-3 py-[11px] text-[14px] text-[var(--text-primary)]
                                 bg-transparent border-none outline-none placeholder:text-[var(--text-muted)]" />
                  </div>
                </div>

                <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.08em]
                                text-[var(--text-muted)] font-[var(--font-mono)] mt-2">
                  <Clock size={14} /><span>Session Settings</span>
                </div>

                {/* Slider */}
                <div className="flex flex-col gap-2">
                  <label className="text-[13px] font-medium text-[var(--text-secondary)]">Daily Session Hours</label>
                  <div className="flex items-center gap-4">
                    <div className="flex-1 relative">
                      <input
                        type="range" name="session_hours" min="1" max="8" step="0.5"
                        value={form.session_hours} onChange={handleChange}
                        style={{ '--pct': `${((form.session_hours - 1) / 7) * 100}%` }}
                        className="slider-track w-full h-1 rounded-full outline-none appearance-none cursor-pointer
                                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-[18px]
                                   [&::-webkit-slider-thumb]:h-[18px] [&::-webkit-slider-thumb]:rounded-full
                                   [&::-webkit-slider-thumb]:bg-[var(--accent-green)]
                                   [&::-webkit-slider-thumb]:shadow-[0_0_12px_rgba(0,245,160,0.6)]
                                   [&::-webkit-slider-thumb]:cursor-pointer
                                   [&::-webkit-slider-thumb]:transition-transform
                                   [&::-webkit-slider-thumb]:duration-150
                                   [&::-webkit-slider-thumb:hover]:scale-125" />
                    </div>
                    <span className="font-[var(--font-mono)] text-[14px] font-medium text-[var(--accent-green)] w-8 text-right shrink-0">
                      {form.session_hours}h
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px] text-[var(--text-muted)] font-[var(--font-mono)] mt-1">
                    <span>1h</span><span>4h</span><span>8h</span>
                  </div>
                </div>

                {/* Submit */}
                <button type="submit"
                  className="gen-btn-main flex items-center justify-center gap-2 mt-2 py-3.5
                             bg-[var(--accent-green)] text-[#080c14]
                             rounded-[var(--radius-md)] text-[15px] font-bold font-[var(--font-body)]">
                  <Zap size={16} className="relative z-[1]" />
                  <span className="relative z-[1]">Generate My Roadmap</span>
                  <ChevronRight size={16} className="relative z-[1]" />
                </button>
              </form>
            </div>

            {/* Info panel */}
            <div className="flex flex-col gap-4">
              <div className="info-card-hover bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6">
                <h3 className="font-[var(--font-display)] text-[15px] font-bold text-[var(--text-primary)] mb-4">
                  What happens next?
                </h3>
                <div className="flex flex-col gap-2.5">
                  {STEPS.map((s, i) => (
                    <div key={i} className="flex items-center gap-3 text-[13px] text-[var(--text-secondary)]
                                            group cursor-default"
                         style={{ animationDelay: `${0.2 + i * 0.05}s` }}>
                      <div className="w-5 h-5 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-medium)]
                                      flex items-center justify-center text-[10px] font-[var(--font-mono)]
                                      text-[var(--text-muted)] shrink-0
                                      transition-all duration-200
                                      group-hover:border-[var(--accent-green)] group-hover:text-[var(--accent-green)]
                                      group-hover:bg-[var(--accent-green-dim)]">
                        {i + 1}
                      </div>
                      <span className="group-hover:text-[var(--text-primary)] transition-colors duration-200">{s}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="info-card-hover border border-[var(--accent-orange-dim)] bg-[rgba(249,115,22,0.05)]
                              rounded-[var(--radius-lg)] p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 right-0 h-[1px]
                                bg-gradient-to-r from-transparent via-[var(--accent-orange)] to-transparent opacity-30" />
                <Trophy size={20} className="text-[var(--accent-orange)]" />
                <h3 className="font-[var(--font-display)] text-[15px] font-bold text-[var(--text-primary)] mt-3 mb-4">
                  Estimated time: 30-60s
                </h3>
                <p className="text-[13px] text-[var(--text-secondary)] leading-[1.6]">
                  Our ML pipeline fetches live data from both platforms and runs multiple analysis algorithms.
                  Grab a coffee while we crunch the numbers.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function LoadingState({ step }) {
  const progress = ((step + 1) / STEPS.length) * 100

  return (
    <div className="flex flex-col items-center text-center px-8 py-[60px]">
      <style>{`
        @keyframes rotate { to { transform: rotate(360deg); } }
        @keyframes ping { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.15); opacity: 0.7; } }
        .loading-ring { animation: rotate 1s linear infinite; }
        .loading-ping { animation: ping 2s ease-in-out infinite; }
        @keyframes progressBar {
          from { width: 0; }
        }
      `}</style>

      {/* Animated icon */}
      <div className="relative mb-6">
        <div className="loading-ping w-20 h-20 rounded-full bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                        flex items-center justify-center">
          <Loader size={32} className="text-[var(--accent-green)] loading-ring" />
        </div>
        {/* Orbiting dot */}
        <div className="absolute inset-0 animate-[spin_3s_linear_infinite]">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1
                          w-2 h-2 rounded-full bg-[var(--accent-green)]
                          shadow-[0_0_8px_var(--accent-green)]" />
        </div>
      </div>

      <h2 className="font-[var(--font-display)] text-[24px] font-bold text-[var(--text-primary)] mb-2">
        Building your roadmap...
      </h2>
      <p className="text-[14px] text-[var(--text-muted)] mb-6">
        This takes 30-60 seconds. Please don't close this tab.
      </p>

      {/* Progress bar */}
      <div className="w-full max-w-[420px] mb-8">
        <div className="h-1 bg-[var(--bg-elevated)] rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-[var(--accent-green)] to-[#00d48c] rounded-full
                          transition-all duration-1000 ease-out
                          shadow-[0_0_12px_rgba(0,245,160,0.5)]"
               style={{ width: `${progress}%` }} />
        </div>
        <div className="flex justify-between text-[11px] text-[var(--text-muted)] font-[var(--font-mono)] mt-1.5">
          <span>{step + 1}/{STEPS.length} steps</span>
          <span>{Math.round(progress)}%</span>
        </div>
      </div>

      {/* Steps */}
      <div className="flex flex-col gap-3 w-full max-w-[420px] text-left">
        {STEPS.map((s, i) => {
          const isDone   = i < step
          const isActive = i === step
          return (
            <div key={i}
              className={[
                'progress-step flex items-center gap-3 text-[14px]',
                isDone   ? 'text-[var(--text-secondary)]'           : '',
                isActive ? 'text-[var(--text-primary)] font-medium' : '',
                !isDone && !isActive ? 'text-[var(--text-muted)] opacity-50' : '',
              ].join(' ')}>
              <div className={[
                'w-[22px] h-[22px] rounded-full border-2 flex items-center justify-center shrink-0 transition-all duration-400',
                isDone
                  ? 'border-[var(--accent-green)] text-[var(--accent-green)] bg-[var(--accent-green-dim)]'
                  : isActive
                    ? 'border-[var(--accent-green)]'
                    : 'border-[var(--border-medium)]',
              ].join(' ')}>
                {isDone && <CheckCircle size={14} />}
                {isActive && (
                  <div className="w-2.5 h-2.5 rounded-full border-2 border-[var(--border-medium)]
                                  border-t-[var(--accent-green)] animate-[spin_0.7s_linear_infinite]" />
                )}
              </div>
              <span>{s}</span>
              {isDone && <span className="ml-auto text-[11px] text-[var(--accent-green)] font-[var(--font-mono)]">✓</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}