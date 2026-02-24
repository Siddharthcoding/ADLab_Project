import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../utils/api'
import toast from 'react-hot-toast'
import { Zap, Code2, Trophy, Clock, ChevronRight, CheckCircle, Loader } from 'lucide-react'

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
    const interval = setInterval(() => {
      setStep(s => (s < STEPS.length - 1 ? s + 1 : s))
    }, 4000)
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
    /* .generate */
    <div className="flex flex-col gap-7">

      {/* .generate__header */}
      <div className="max-w-[600px]">
        {/* .generate__title */}
        <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)] mb-2">
          Generate Roadmap
        </h1>
        {/* .generate__sub */}
        <p className="text-[15px] text-[var(--text-secondary)] leading-[1.6]">
          Connect your competitive programming profiles and let our ML pipeline build your personalized plan.
        </p>
      </div>

      {loading ? (
        <LoadingState step={step} />
      ) : (
        /* .generate__layout — 2 cols → 1 col @900px */
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start">

          {/* .generate__form-card */}
          <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-xl)] p-8">

            {/* .generate__form */}
            <form onSubmit={handleSubmit} className="flex flex-col gap-5">

              {/* .generate__section-label — Profile Handles */}
              <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.08em]
                              text-[var(--text-muted)] font-[var(--font-mono)]">
                <Code2 size={14} />
                Profile Handles
              </div>

              {/* LeetCode */}
              <div className="flex flex-col gap-2">
                <label className="text-[13px] font-medium text-[var(--text-secondary)]">
                  LeetCode Username
                </label>
                {/* .generate__input-wrap */}
                <div className="flex items-center bg-[var(--bg-secondary)] border border-[var(--border-medium)]
                                rounded-[var(--radius-md)] overflow-hidden transition-all duration-200
                                focus-within:border-[var(--accent-green)] focus-within:shadow-[0_0_0_3px_var(--accent-green-dim)]">
                  {/* .generate__input-prefix */}
                  <span className="px-3 py-[11px] text-[12px] font-[var(--font-mono)] text-[var(--text-muted)]
                                   bg-[var(--bg-elevated)] border-r border-[var(--border-subtle)] whitespace-nowrap">
                    leetcode.com/
                  </span>
                  {/* .generate__prefixed-input */}
                  <input
                    type="text" name="leetcode_username" placeholder="your_username"
                    value={form.leetcode_username} onChange={handleChange}
                    className="flex-1 px-3 py-[11px] text-[14px] text-[var(--text-primary)]
                               bg-transparent border-none outline-none
                               placeholder:text-[var(--text-muted)]"
                  />
                </div>
              </div>

              {/* Codeforces */}
              <div className="flex flex-col gap-2">
                <label className="text-[13px] font-medium text-[var(--text-secondary)]">
                  Codeforces Handle
                </label>
                {/* .generate__input-wrap */}
                <div className="flex items-center bg-[var(--bg-secondary)] border border-[var(--border-medium)]
                                rounded-[var(--radius-md)] overflow-hidden transition-all duration-200
                                focus-within:border-[var(--accent-green)] focus-within:shadow-[0_0_0_3px_var(--accent-green-dim)]">
                  {/* .generate__input-prefix */}
                  <span className="px-3 py-[11px] text-[12px] font-[var(--font-mono)] text-[var(--text-muted)]
                                   bg-[var(--bg-elevated)] border-r border-[var(--border-subtle)] whitespace-nowrap">
                    codeforces.com/
                  </span>
                  {/* .generate__prefixed-input */}
                  <input
                    type="text" name="codeforces_handle" placeholder="your_handle"
                    value={form.codeforces_handle} onChange={handleChange}
                    className="flex-1 px-3 py-[11px] text-[14px] text-[var(--text-primary)]
                               bg-transparent border-none outline-none
                               placeholder:text-[var(--text-muted)]"
                  />
                </div>
              </div>

              {/* .generate__section-label — Session Settings */}
              <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.08em]
                              text-[var(--text-muted)] font-[var(--font-mono)] mt-2">
                <Clock size={14} />
                Session Settings
              </div>

              {/* Slider */}
              <div className="flex flex-col gap-2">
                <label className="text-[13px] font-medium text-[var(--text-secondary)]">
                  Daily Session Hours
                </label>
                {/* .generate__slider-wrap */}
                <div className="flex items-center gap-4">
                  {/* .generate__slider — custom thumb via inline style */}
                  <input
                    type="range" name="session_hours" min="1" max="8" step="0.5"
                    value={form.session_hours} onChange={handleChange}
                    className="flex-1 h-1 rounded-full bg-[var(--bg-elevated)] outline-none appearance-none cursor-pointer
                               [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-[18px]
                               [&::-webkit-slider-thumb]:h-[18px] [&::-webkit-slider-thumb]:rounded-full
                               [&::-webkit-slider-thumb]:bg-[var(--accent-green)]
                               [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(0,245,160,0.4)]
                               [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  {/* .generate__slider-val */}
                  <span className="font-[var(--font-mono)] text-[14px] font-medium text-[var(--accent-green)] w-8 text-right shrink-0">
                    {form.session_hours}h
                  </span>
                </div>
                {/* .generate__slider-labels */}
                <div className="flex justify-between text-[11px] text-[var(--text-muted)] font-[var(--font-mono)] mt-1">
                  <span>1h</span><span>4h</span><span>8h</span>
                </div>
              </div>

              {/* .generate__submit */}
              <button
                type="submit"
                className="flex items-center justify-center gap-2 mt-2 py-3.5
                           bg-[var(--accent-green)] text-[#080c14]
                           rounded-[var(--radius-md)] text-[15px] font-bold font-[var(--font-body)]
                           transition-all duration-200
                           hover:bg-[#00d48c] hover:shadow-[0_0_24px_rgba(0,245,160,0.3)] hover:-translate-y-px"
              >
                <Zap size={16} />
                Generate My Roadmap
                <ChevronRight size={16} />
              </button>
            </form>
          </div>

          {/* .generate__info */}
          <div className="flex flex-col gap-4">

            {/* .generate__info-card */}
            <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6">
              {/* .generate__info-title */}
              <h3 className="font-[var(--font-display)] text-[15px] font-bold text-[var(--text-primary)] mb-4">
                What happens next?
              </h3>
              {/* .generate__steps */}
              <div className="flex flex-col gap-2.5">
                {STEPS.map((s, i) => (
                  /* .generate__step */
                  <div key={i} className="flex items-center gap-3 text-[13px] text-[var(--text-secondary)]">
                    {/* .generate__step-num */}
                    <div className="w-5 h-5 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-medium)]
                                    flex items-center justify-center text-[10px] font-[var(--font-mono)]
                                    text-[var(--text-muted)] shrink-0">
                      {i + 1}
                    </div>
                    <span>{s}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* .generate__info-card .generate__info-card--accent */}
            <div className="border border-[var(--accent-orange-dim)] bg-[rgba(249,115,22,0.05)]
                            rounded-[var(--radius-lg)] p-6">
              <Trophy size={20} className="text-[var(--accent-orange)]" />
              {/* .generate__info-title */}
              <h3 className="font-[var(--font-display)] text-[15px] font-bold text-[var(--text-primary)] mt-3 mb-4">
                Estimated time: 30-60s
              </h3>
              {/* .generate__info-desc */}
              <p className="text-[13px] text-[var(--text-secondary)] leading-[1.6]">
                Our ML pipeline fetches live data from both platforms and runs multiple analysis algorithms.
                Grab a coffee while we crunch the numbers.
              </p>
            </div>

          </div>
        </div>
      )}
    </div>
  )
}

/* ── LoadingState ── */
function LoadingState({ step }) {
  return (
    /* .generate__loading */
    <div className="flex flex-col items-center text-center px-8 py-[60px]">

      {/* .generate__loading-icon */}
      <div className="w-20 h-20 rounded-full bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                      flex items-center justify-center mb-6
                      animate-[pulseGlow_2s_ease-in-out_infinite]">
        <Loader
          size={32}
          className="text-[var(--accent-green)]"
          style={{ animation: 'spin 1s linear infinite' }}
        />
      </div>

      {/* .generate__loading-title */}
      <h2 className="font-[var(--font-display)] text-[24px] font-bold text-[var(--text-primary)] mb-2">
        Building your roadmap...
      </h2>

      {/* .generate__loading-sub */}
      <p className="text-[14px] text-[var(--text-muted)] mb-12">
        This takes 30-60 seconds. Please don't close this tab.
      </p>

      {/* .generate__progress-steps */}
      <div className="flex flex-col gap-3 w-full max-w-[420px] text-left">
        {STEPS.map((s, i) => {
          const isDone   = i < step
          const isActive = i === step
          return (
            /* .generate__progress-step [.done | .active] */
            <div
              key={i}
              className={[
                'flex items-center gap-3 text-[14px] transition-colors duration-200',
                isDone   ? 'text-[var(--text-secondary)]'              : '',
                isActive ? 'text-[var(--text-primary)] font-medium'    : '',
                !isDone && !isActive ? 'text-[var(--text-muted)]'      : '',
              ].join(' ')}
            >
              {/* .generate__progress-dot */}
              <div
                className={[
                  'w-[22px] h-[22px] rounded-full border-2 flex items-center justify-center shrink-0 transition-all duration-200',
                  isDone
                    ? 'border-[var(--accent-green)] text-[var(--accent-green)] bg-[var(--accent-green-dim)]'
                    : isActive
                      ? 'border-[var(--accent-green)] text-[var(--text-muted)]'
                      : 'border-[var(--border-medium)] text-[var(--text-muted)]',
                ].join(' ')}
              >
                {isDone   && <CheckCircle size={14} />}
                {isActive && (
                  /* .generate__progress-spinner */
                  <div className="w-2.5 h-2.5 rounded-full border-2 border-[var(--border-medium)]
                                  border-t-[var(--accent-green)]
                                  animate-[spin_0.7s_linear_infinite]" />
                )}
              </div>
              <span>{s}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}