import React from 'react'
import { Link } from 'react-router-dom'
import { Code2, Zap, Brain, Target, TrendingUp, ChevronRight } from 'lucide-react'

const features = [
  { icon: Brain,      label: 'ML-Powered Analysis',  desc: 'Advanced algorithms analyze your performance patterns and identify true knowledge gaps.' },
  { icon: Target,     label: 'Personalized Roadmap', desc: 'Problems ranked by your weak topics, forgetting curves, and contest relevance.' },
  { icon: TrendingUp, label: 'GNN Gap Detection',    desc: 'Graph neural networks uncover hidden prerequisite gaps you never knew existed.' },
  { icon: Zap,        label: 'Session Optimizer',    desc: 'SM-2 spaced repetition schedules your practice for maximum retention ROI.' },
]

const TERMINAL_LINES = [
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Fetched',  ' 847 submissions',    ' from LeetCode'],       delay: '0.6s' },
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Analyzed', ' contest history',    ' (CF: 1243 rating)'],   delay: '0.9s' },
  { sym: '⟳', symCls: 'text-[var(--accent-blue)]',   parts: [' Running',  ' forgetting curve',   ' analysis...'],         delay: '1.2s' },
  { sym: '!', symCls: 'text-[var(--accent-orange)]', parts: [' Detected', ' 6 at-risk topics',   ' (≥80% forgotten)'],    delay: '1.5s' },
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Generated',' 42-problem roadmap', ' · 7 days'],            delay: '1.8s' },
]

export default function LandingPage() {
  return (
    /* .landing */
    <div className="min-h-screen relative overflow-x-hidden">

      {/* .landing__bg — fixed fullscreen background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* .landing__orb--1  600×600 green, top-right */}
        <div className="absolute w-[600px] h-[600px] rounded-full blur-[100px] opacity-[0.12]
                        bg-[var(--accent-green)] -top-[200px] -right-[100px]" />
        {/* .landing__orb--2  500×500 blue, bottom-left */}
        <div className="absolute w-[500px] h-[500px] rounded-full blur-[100px] opacity-[0.12]
                        bg-[var(--accent-blue)] bottom-[100px] -left-[100px]" />
        {/* .landing__grid — dot grid with radial mask */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)',
            backgroundSize: '60px 60px',
            maskImage: 'radial-gradient(ellipse 80% 80% at 50% 0%, black 0%, transparent 100%)',
          }}
        />
      </div>

      {/* ── Nav ── .landing__nav */}
      <nav className="relative z-10 sticky top-0 border-b border-[var(--border-subtle)]
                      backdrop-blur-[12px] bg-[rgba(8,12,20,0.8)]">
        {/* .landing__nav-inner */}
        <div className="max-w-[1100px] mx-auto px-8 h-16 flex items-center justify-between">

          {/* .landing__nav-logo */}
          <div className="flex items-center gap-2.5 font-[var(--font-display)] font-bold text-[16px] text-[var(--text-primary)]">
            {/* .landing__nav-icon */}
            <div className="w-8 h-8 flex items-center justify-center rounded-[var(--radius-sm)]
                            bg-[var(--accent-green-dim)] border border-[var(--border-accent)] text-[var(--accent-green)]">
              <Code2 size={16} />
            </div>
            <span>CP Roadmap</span>
          </div>

          {/* .landing__nav-links */}
          <div className="flex items-center gap-3">
            {/* .btn .btn--ghost */}
            <Link
              to="/auth?tab=login"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[var(--radius-md)]
                         text-[14px] font-semibold border border-transparent
                         bg-transparent text-[var(--text-secondary)] font-[var(--font-body)]
                         transition-all duration-200
                         hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]"
            >
              Sign In
            </Link>
            {/* .btn .btn--primary */}
            <Link
              to="/auth?tab=register"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[var(--radius-md)]
                         text-[14px] font-semibold border border-[var(--accent-green)]
                         bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]
                         transition-all duration-200
                         hover:bg-[#00d48c] hover:shadow-[0_0_24px_rgba(0,245,160,0.3)] hover:-translate-y-px"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── .landing__hero */}
      <section className="relative z-[1] max-w-[1100px] mx-auto px-8 pt-20 pb-24
                          md:px-8 px-5 md:pt-20 pt-12 md:pb-24 pb-16">

        {/* .landing__badge */}
        <div className="inline-flex items-center gap-2 mb-8
                        bg-[var(--bg-card)] border border-[var(--border-medium)]
                        rounded-full px-3.5 py-1.5
                        text-[12px] font-[var(--font-mono)] text-[var(--text-secondary)]">
          {/* .landing__badge-dot */}
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-green)]
                           shadow-[0_0_8px_var(--accent-green)]
                           animate-[pulseGlow_2s_infinite]" />
          <span>ML-Powered · Spaced Repetition · GNN Analysis</span>
        </div>

        {/* .landing__title */}
        <h1 className="font-[var(--font-display)] font-extrabold leading-[1.0] tracking-[-0.03em]
                       text-[var(--text-primary)] mb-7 max-w-[700px]
                       text-[clamp(48px,7vw,88px)] md:text-[clamp(48px,7vw,88px)] text-[42px]">
          Your Path to<br />
          {/* .landing__title-accent */}
          <span className="text-[var(--accent-green)] drop-shadow-[0_0_40px_rgba(0,245,160,0.3)]">
            Competitive
          </span><br />
          Programming<br />
          {/* .landing__title-outline */}
          <span className="[-webkit-text-stroke:2px_var(--text-primary)] text-transparent">
            Mastery
          </span>
        </h1>

        {/* .landing__subtitle */}
        <p className="text-[18px] text-[var(--text-secondary)] max-w-[520px] leading-[1.7] mb-10">
          Connect your LeetCode and Codeforces profiles. Let our ML pipeline
          identify your weaknesses, forgotten topics, and hidden gaps — then
          generate a personalized 7-day practice plan.
        </p>

        {/* .landing__cta */}
        <div className="flex gap-3 flex-wrap mb-16 md:flex-row flex-col">
          {/* .btn .btn--primary .btn--lg */}
          <Link
            to="/auth?tab=register"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                       text-[15px] font-semibold border border-[var(--accent-green)]
                       bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]
                       transition-all duration-200
                       hover:bg-[#00d48c] hover:shadow-[0_0_24px_rgba(0,245,160,0.3)] hover:-translate-y-px"
          >
            Start Your Journey <ChevronRight size={18} />
          </Link>
          {/* .btn .btn--outline .btn--lg */}
          <Link
            to="/auth?tab=login"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                       text-[15px] font-semibold
                       bg-transparent text-[var(--text-primary)]
                       border border-[var(--border-medium)] font-[var(--font-body)]
                       transition-all duration-200
                       hover:bg-[var(--bg-elevated)]"
          >
            I have an account
          </Link>
        </div>

        {/* .landing__terminal */}
        <div
          className="bg-[var(--bg-card)] border border-[var(--border-medium)]
                     rounded-[var(--radius-lg)] max-w-[520px] overflow-hidden
                     shadow-[var(--shadow-card),var(--shadow-glow)]
                     [animation:fadeIn_0.8s_ease_0.3s_both]"
        >
          {/* .landing__terminal-header */}
          <div className="flex items-center gap-1.5 px-4 py-3
                          bg-[var(--bg-elevated)] border-b border-[var(--border-subtle)]">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#febc2e]" />
            <span className="w-2.5 h-2.5 rounded-full bg-[#28c840]" />
            {/* .landing__terminal-title */}
            <span className="font-[var(--font-mono)] text-[12px] text-[var(--text-muted)] ml-2">
              ml_pipeline.py
            </span>
          </div>

          {/* .landing__terminal-body */}
          <div className="p-5 font-[var(--font-mono)] text-[13px] flex flex-col gap-2.5">
            {TERMINAL_LINES.map((line, i) => (
              /* .landing__terminal-line + staggered animation-delay */
              <div
                key={i}
                className="flex gap-1 [animation:fadeInLeft_0.4s_ease_both]"
                style={{ animationDelay: line.delay }}
              >
                <span className={line.symCls}>{line.sym}</span>
                <span className="text-[var(--text-secondary)]">{line.parts[0]}</span>
                <span className="text-[var(--text-primary)]">{line.parts[1]}</span>
                <span className="text-[var(--text-secondary)]">{line.parts[2]}</span>
              </div>
            ))}
            {/* .landing__terminal-cursor */}
            <span className="text-[var(--accent-green)] animate-[pulseGlow_1s_step-end_infinite]">_</span>
          </div>
        </div>
      </section>

      {/* ── Features ── .landing__features */}
      <section className="relative z-[1] border-t border-[var(--border-subtle)]
                          py-20 px-8 md:py-20 md:px-8 py-16 px-5">
        {/* .landing__features-inner */}
        <div className="max-w-[1100px] mx-auto">
          {/* .landing__section-label */}
          <p className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.1em]
                        text-[var(--accent-green)] mb-3">
            What's under the hood
          </p>
          {/* .landing__section-title */}
          <h2 className="font-[var(--font-display)] font-bold text-[var(--text-primary)] mb-12
                         text-[clamp(28px,4vw,40px)]">
            Intelligence at every layer
          </h2>
          {/* .landing__features-grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {features.map(({ icon: Icon, label, desc }, i) => (
              /* .landing__feature-card */
              <div
                key={i}
                style={{ animationDelay: `${i * 0.1}s` }}
                className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                           rounded-[var(--radius-lg)] p-7
                           [animation:fadeIn_0.5s_ease_both]
                           transition-all duration-200
                           hover:border-[var(--border-medium)] hover:bg-[var(--bg-card-hover)]
                           hover:-translate-y-0.5 hover:shadow-[var(--shadow-card)]"
              >
                {/* .landing__feature-icon */}
                <div className="w-10 h-10 flex items-center justify-center mb-4
                                rounded-[var(--radius-md)]
                                bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                                text-[var(--accent-green)]">
                  <Icon size={20} />
                </div>
                {/* .landing__feature-label */}
                <h3 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)] mb-2">
                  {label}
                </h3>
                {/* .landing__feature-desc */}
                <p className="text-[14px] text-[var(--text-secondary)] leading-[1.6]">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Stats ── .landing__stats */}
      <section className="relative z-[1] border-t border-[var(--border-subtle)]
                          py-[60px] px-8 bg-[var(--bg-secondary)]">
        {/* .landing__stats-inner */}
        <div className="max-w-[1100px] mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { value: '2 Platforms', label: 'LeetCode + Codeforces' },
            { value: 'SM-2',        label: 'Spaced Repetition Algorithm' },
            { value: 'GNN',         label: 'Graph Neural Network Analysis' },
            { value: '7-Day',       label: 'Personalized Practice Calendar' },
          ].map(({ value, label }, i) => (
            /* .landing__stat */
            <div key={i} className="text-center">
              {/* .landing__stat-value */}
              <div className="font-[var(--font-display)] text-[28px] font-extrabold
                              text-[var(--accent-green)] mb-1.5">
                {value}
              </div>
              {/* .landing__stat-label */}
              <div className="text-[13px] text-[var(--text-secondary)]">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Final CTA ── .landing__final-cta */}
      <section className="relative z-[1] text-center py-[100px] px-8">
        {/* .landing__final-title */}
        <h2 className="font-[var(--font-display)] font-extrabold text-[var(--text-primary)] mb-4
                       text-[clamp(32px,5vw,56px)]">
          Ready to level up?
        </h2>
        {/* .landing__final-sub */}
        <p className="text-[18px] text-[var(--text-secondary)] mb-9">
          Generate your first personalized roadmap in under 60 seconds.
        </p>
        {/* .btn .btn--primary .btn--lg */}
        <Link
          to="/auth?tab=register"
          className="inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                     text-[15px] font-semibold border border-[var(--accent-green)]
                     bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]
                     transition-all duration-200
                     hover:bg-[#00d48c] hover:shadow-[0_0_24px_rgba(0,245,160,0.3)] hover:-translate-y-px"
        >
          Create Free Account <ChevronRight size={18} />
        </Link>
      </section>

      {/* ── Footer ── .landing__footer */}
      <footer className="relative z-[1] border-t border-[var(--border-subtle)] py-6 px-8">
        {/* .landing__footer-inner */}
        <div className="max-w-[1100px] mx-auto flex items-center justify-between gap-4 flex-wrap">
          {/* .landing__nav-logo (reused) */}
          <div className="flex items-center gap-2.5 font-[var(--font-display)] font-bold text-[16px] text-[var(--text-primary)]">
            {/* .landing__nav-icon */}
            <div className="w-8 h-8 flex items-center justify-center rounded-[var(--radius-sm)]
                            bg-[var(--accent-green-dim)] border border-[var(--border-accent)] text-[var(--accent-green)]">
              <Code2 size={14} />
            </div>
            <span>CP Roadmap</span>
          </div>
          {/* .landing__footer-copy */}
          <span className="text-[13px] text-[var(--text-muted)]">
            Built for competitive programmers, by competitive programmers.
          </span>
        </div>
      </footer>

    </div>
  )
}