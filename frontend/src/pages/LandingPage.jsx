import React, { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { Code2, Zap, Brain, Target, TrendingUp, ChevronRight, ArrowRight } from 'lucide-react'

const features = [
  { icon: Brain,      label: 'ML-Powered Analysis',  desc: 'Advanced algorithms analyze your performance patterns and identify true knowledge gaps.', color: 'green' },
  { icon: Target,     label: 'Personalized Roadmap', desc: 'Problems ranked by your weak topics, forgetting curves, and contest relevance.',          color: 'blue'  },
  { icon: TrendingUp, label: 'GNN Gap Detection',    desc: 'Graph neural networks uncover hidden prerequisite gaps you never knew existed.',          color: 'purple'},
  { icon: Zap,        label: 'Session Optimizer',    desc: 'SM-2 spaced repetition schedules your practice for maximum retention ROI.',               color: 'orange'},
]

const TERMINAL_LINES = [
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Fetched',  ' 847 submissions',    ' from LeetCode'],     delay: '0.6s' },
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Analyzed', ' contest history',    ' (CF: 1243 rating)'], delay: '0.9s' },
  { sym: '⟳', symCls: 'text-[var(--accent-blue)]',   parts: [' Running',  ' forgetting curve',   ' analysis...'],       delay: '1.2s' },
  { sym: '!', symCls: 'text-[var(--accent-orange)]', parts: [' Detected', ' 6 at-risk topics',   ' (≥80% forgotten)'],  delay: '1.5s' },
  { sym: '✓', symCls: 'text-[var(--accent-green)]',  parts: [' Generated',' 42-problem roadmap', ' · 7 days'],          delay: '1.8s' },
]

const iconDimCls = {
  green:  'bg-[var(--accent-green-dim)]  border-[rgba(0,245,160,0.25)]  text-[var(--accent-green)]',
  blue:   'bg-[var(--accent-blue-dim)]   border-[rgba(59,130,246,0.25)] text-[var(--accent-blue)]',
  purple: 'bg-[var(--accent-purple-dim)] border-[rgba(168,85,247,0.25)] text-[var(--accent-purple)]',
  orange: 'bg-[var(--accent-orange-dim)] border-[rgba(249,115,22,0.25)] text-[var(--accent-orange)]',
}

export default function LandingPage() {
  const [scrollY, setScrollY] = useState(0)
  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <>
      <style>{`
        @keyframes drift1 { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(40px,-30px) scale(1.06)} 66%{transform:translate(-30px,40px) scale(0.94)} }
        @keyframes drift2 { 0%,100%{transform:translate(0,0) scale(1)} 33%{transform:translate(-35px,25px) scale(1.1)} 66%{transform:translate(40px,-20px) scale(0.91)} }
        @keyframes drift3 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,-30px)} }
        @keyframes heroTitle { from{opacity:0;transform:translateY(32px) skewY(1deg)} to{opacity:1;transform:translateY(0) skewY(0)} }
        @keyframes heroBadge { from{opacity:0;transform:translateY(-12px)} to{opacity:1;transform:translateY(0)} }
        @keyframes heroSub { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        @keyframes heroCta { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes terminalSlide { from{opacity:0;transform:translateX(-24px)} to{opacity:1;transform:translateX(0)} }
        @keyframes featureCardIn { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
        @keyframes statIn { from{opacity:0;transform:scale(0.85)} to{opacity:1;transform:scale(1)} }
        @keyframes gradientShift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }
        @keyframes pulseGlow { 0%,100%{box-shadow:0 0 20px rgba(0,245,160,0.2)} 50%{box-shadow:0 0 40px rgba(0,245,160,0.5),0 0 80px rgba(0,245,160,0.1)} }
        @keyframes cursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes fadeInLeft { from{opacity:0;transform:translateX(-16px)} to{opacity:1;transform:translateX(0)} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

        .orb1 { animation: drift1 9s ease-in-out infinite; }
        .orb2 { animation: drift2 11s ease-in-out infinite; }
        .orb3 { animation: drift3 13s ease-in-out infinite; }
        .badge-enter { animation: heroBadge 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s both; }
        .title-enter { animation: heroTitle 0.7s cubic-bezier(0.16,1,0.3,1) 0.2s both; }
        .sub-enter   { animation: heroSub   0.6s cubic-bezier(0.16,1,0.3,1) 0.45s both; }
        .cta-enter   { animation: heroCta   0.6s cubic-bezier(0.16,1,0.3,1) 0.6s both; }
        .term-enter  { animation: terminalSlide 0.7s cubic-bezier(0.16,1,0.3,1) 0.35s both; }

        .primary-btn {
          position: relative; overflow: hidden;
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .primary-btn::after {
          content:''; position:absolute; inset:0;
          background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
          transform:translateX(-100%);
        }
        .primary-btn:hover::after { transform:translateX(100%); transition:transform 0.6s ease; }
        .primary-btn:hover { transform:translateY(-3px); box-shadow:0 0 40px rgba(0,245,160,0.45), 0 12px 24px rgba(0,0,0,0.3); }
        .primary-btn:active { transform:translateY(-1px); }

        .outline-btn { transition: all 0.25s ease; }
        .outline-btn:hover { background: var(--bg-elevated); transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.25); }

        .feature-card {
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
          position: relative; overflow: hidden;
        }
        .feature-card::before {
          content:''; position:absolute; inset:0;
          background:radial-gradient(circle at 50% 0%, rgba(0,245,160,0.04), transparent 60%);
          opacity:0; transition:opacity 0.3s;
        }
        .feature-card:hover::before { opacity:1; }
        .feature-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 12px 36px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08);
          border-color: rgba(255,255,255,0.12) !important;
        }

        .stat-card { transition: all 0.25s ease; }
        .stat-card:hover { transform: translateY(-3px); }

        .badge-dot-pulse { animation: pulseGlow 2s ease-in-out infinite; }
        .cursor-blink { animation: cursorBlink 1s step-end infinite; }

        .nav-logo-hover { transition: transform 0.2s ease; }
        .nav-logo-hover:hover { transform: scale(1.05); }

        .gradient-text {
          background: linear-gradient(135deg, var(--accent-green), #00d48c, var(--accent-blue));
          background-size: 200% 200%;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          animation: gradientShift 4s ease infinite;
        }
      `}</style>

      <div className="min-h-screen relative overflow-x-hidden">

        {/* Background */}
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="orb1 absolute w-[700px] h-[700px] rounded-full opacity-[0.09]
                          bg-[var(--accent-green)] -top-[200px] -right-[150px] blur-[130px]" />
          <div className="orb2 absolute w-[550px] h-[550px] rounded-full opacity-[0.07]
                          bg-[var(--accent-blue)] bottom-[50px] -left-[120px] blur-[110px]" />
          <div className="orb3 absolute w-[350px] h-[350px] rounded-full opacity-[0.05]
                          bg-[var(--accent-purple)] top-[35%] left-[55%] blur-[90px]" />
          {/* Grid */}
          <div className="absolute inset-0" style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
            backgroundSize: '60px 60px',
            maskImage: 'radial-gradient(ellipse 85% 85% at 50% 0%, black 0%, transparent 100%)',
          }} />
          {/* Vignette */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(8,12,20,0.4)_100%)]" />
        </div>

        {/* Nav */}
        <nav className="relative z-10 sticky top-0 border-b border-[var(--border-subtle)]
                        backdrop-blur-[16px] bg-[rgba(8,12,20,0.75)]"
             style={{ transform: `translateY(${Math.min(scrollY * 0.1, 0)}px)` }}>
          <div className="max-w-[1100px] mx-auto px-8 h-16 flex items-center justify-between">
            <Link to="/" className="nav-logo-hover flex items-center gap-2.5 font-[var(--font-display)] font-bold text-[16px] text-[var(--text-primary)]">
              <div className="badge-dot-pulse w-8 h-8 flex items-center justify-center rounded-[var(--radius-sm)]
                              bg-[var(--accent-green-dim)] border border-[var(--border-accent)] text-[var(--accent-green)]">
                <Code2 size={16} />
              </div>
              <span>CP Roadmap</span>
            </Link>
            <div className="flex items-center gap-3">
              <Link to="/auth?tab=login"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[var(--radius-md)]
                           text-[14px] font-semibold bg-transparent text-[var(--text-secondary)]
                           font-[var(--font-body)] transition-all duration-200
                           hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]">
                Sign In
              </Link>
              <Link to="/auth?tab=register"
                className="primary-btn inline-flex items-center gap-2 px-5 py-2.5 rounded-[var(--radius-md)]
                           text-[14px] font-semibold border border-[var(--accent-green)]
                           bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]">
                <span className="relative z-[1]">Get Started</span>
                <ChevronRight size={15} className="relative z-[1]" />
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero */}
        <section className="relative z-[1] max-w-[1100px] mx-auto px-8 pt-20 pb-24 md:pt-20 pt-12 md:pb-24 pb-16">

          {/* Badge */}
          <div className="badge-enter inline-flex items-center gap-2 mb-8
                          bg-[var(--bg-card)] border border-[var(--border-medium)]
                          rounded-full px-3.5 py-1.5 text-[12px] font-[var(--font-mono)] text-[var(--text-secondary)]">
            <span className="badge-dot-pulse w-1.5 h-1.5 rounded-full bg-[var(--accent-green)] shrink-0" />
            <span>ML-Powered · Spaced Repetition · GNN Analysis</span>
            <span className="ml-1 text-[var(--accent-green)] font-semibold">Beta</span>
          </div>

          {/* Title */}
          <h1 className="title-enter font-[var(--font-display)] font-extrabold leading-[1.0] tracking-[-0.03em]
                         text-[var(--text-primary)] mb-7 max-w-[700px]
                         text-[clamp(44px,7vw,88px)]">
            Your Path to<br />
            <span className="gradient-text">Competitive</span><br />
            Programming<br />
            <span className="[-webkit-text-stroke:2px_var(--text-primary)] text-transparent">Mastery</span>
          </h1>

          {/* Subtitle */}
          <p className="sub-enter text-[18px] text-[var(--text-secondary)] max-w-[520px] leading-[1.7] mb-10">
            Connect your LeetCode and Codeforces profiles. Let our ML pipeline
            identify your weaknesses, forgotten topics, and hidden gaps — then
            generate a personalized 7-day practice plan.
          </p>

          {/* CTAs */}
          <div className="cta-enter flex gap-3 flex-wrap mb-16 md:flex-row flex-col">
            <Link to="/auth?tab=register"
              className="primary-btn inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                         text-[15px] font-semibold border border-[var(--accent-green)]
                         bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]">
              <span className="relative z-[1]">Start Your Journey</span>
              <ArrowRight size={18} className="relative z-[1]" />
            </Link>
            <Link to="/auth?tab=login"
              className="outline-btn inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                         text-[15px] font-semibold bg-transparent text-[var(--text-primary)]
                         border border-[var(--border-medium)] font-[var(--font-body)]">
              I have an account
            </Link>
          </div>

          {/* Terminal */}
          <div className="term-enter bg-[var(--bg-card)] border border-[var(--border-medium)]
                          rounded-[var(--radius-lg)] max-w-[520px] overflow-hidden
                          shadow-[0_24px_64px_rgba(0,0,0,0.5),0_0_0_1px_rgba(255,255,255,0.05)]
                          transition-transform duration-300 hover:-translate-y-1
                          hover:shadow-[0_32px_80px_rgba(0,0,0,0.5),0_0_40px_rgba(0,245,160,0.08)]">
            <div className="flex items-center gap-1.5 px-4 py-3 bg-[var(--bg-elevated)] border-b border-[var(--border-subtle)]">
              <span className="w-3 h-3 rounded-full bg-[#ff5f57] cursor-pointer hover:brightness-110 transition-all" />
              <span className="w-3 h-3 rounded-full bg-[#febc2e] cursor-pointer hover:brightness-110 transition-all" />
              <span className="w-3 h-3 rounded-full bg-[#28c840] cursor-pointer hover:brightness-110 transition-all" />
              <span className="font-[var(--font-mono)] text-[12px] text-[var(--text-muted)] ml-2">ml_pipeline.py</span>
              <span className="ml-auto font-[var(--font-mono)] text-[10px] text-[var(--accent-green)] opacity-60">● running</span>
            </div>
            <div className="p-5 font-[var(--font-mono)] text-[13px] flex flex-col gap-2.5">
              {TERMINAL_LINES.map((line, i) => (
                <div key={i} className="flex gap-1 [animation:fadeInLeft_0.4s_ease_both]"
                     style={{ animationDelay: line.delay }}>
                  <span className={line.symCls}>{line.sym}</span>
                  <span className="text-[var(--text-secondary)]">{line.parts[0]}</span>
                  <span className="text-[var(--text-primary)]">{line.parts[1]}</span>
                  <span className="text-[var(--text-secondary)]">{line.parts[2]}</span>
                </div>
              ))}
              <span className="cursor-blink text-[var(--accent-green)]">_</span>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="relative z-[1] border-t border-[var(--border-subtle)] py-20 px-8 md:py-20 py-16 px-5 md:px-8">
          <div className="max-w-[1100px] mx-auto">
            <p className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.12em] text-[var(--accent-green)] mb-3">
              What's under the hood
            </p>
            <h2 className="font-[var(--font-display)] font-bold text-[var(--text-primary)] mb-12
                           text-[clamp(28px,4vw,40px)]">
              Intelligence at every layer
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
              {features.map(({ icon: Icon, label, desc, color }, i) => (
                <div key={i}
                  style={{ animationDelay: `${i * 0.1}s` }}
                  className="feature-card bg-[var(--bg-card)] border border-[var(--border-subtle)]
                             rounded-[var(--radius-lg)] p-7 [animation:featureCardIn_0.5s_ease_both] group">
                  <div className={`w-10 h-10 flex items-center justify-center mb-4 rounded-[var(--radius-md)]
                                   border ${iconDimCls[color]}
                                   transition-transform duration-300 group-hover:scale-110`}>
                    <Icon size={20} />
                  </div>
                  <h3 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)] mb-2">{label}</h3>
                  <p className="text-[14px] text-[var(--text-secondary)] leading-[1.6]">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Stats */}
        <section className="relative z-[1] border-t border-[var(--border-subtle)] py-[60px] px-8 bg-[var(--bg-secondary)]">
          <div className="max-w-[1100px] mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: '2 Platforms', label: 'LeetCode + Codeforces', color: 'green' },
              { value: 'SM-2',        label: 'Spaced Repetition Algorithm', color: 'blue' },
              { value: 'GNN',         label: 'Graph Neural Network Analysis', color: 'purple' },
              { value: '7-Day',       label: 'Personalized Practice Calendar', color: 'orange' },
            ].map(({ value, label, color }, i) => (
              <div key={i} className="stat-card text-center cursor-default"
                   style={{ animationDelay: `${i * 0.08}s` }}>
                <div className={`font-[var(--font-display)] text-[28px] font-extrabold mb-1.5 ${
                  color === 'green'  ? 'text-[var(--accent-green)]'  :
                  color === 'blue'   ? 'text-[var(--accent-blue)]'   :
                  color === 'purple' ? 'text-[var(--accent-purple)]' :
                                       'text-[var(--accent-orange)]'
                }`}>{value}</div>
                <div className="text-[13px] text-[var(--text-secondary)]">{label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Final CTA */}
        <section className="relative z-[1] text-center py-[100px] px-8 overflow-hidden">
          {/* Glow behind CTA */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                          w-[600px] h-[300px] rounded-full bg-[var(--accent-green)] opacity-[0.04] blur-[80px]
                          pointer-events-none" />
          <h2 className="relative font-[var(--font-display)] font-extrabold text-[var(--text-primary)] mb-4
                         text-[clamp(32px,5vw,56px)]">
            Ready to level up?
          </h2>
          <p className="relative text-[18px] text-[var(--text-secondary)] mb-9">
            Generate your first personalized roadmap in under 60 seconds.
          </p>
          <Link to="/auth?tab=register"
            className="primary-btn relative inline-flex items-center gap-2 px-7 py-3.5 rounded-[var(--radius-lg)]
                       text-[15px] font-semibold border border-[var(--accent-green)]
                       bg-[var(--accent-green)] text-[#080c14] font-[var(--font-body)]">
            <span className="relative z-[1]">Create Free Account</span>
            <ChevronRight size={18} className="relative z-[1]" />
          </Link>
        </section>

        {/* Footer */}
        <footer className="relative z-[1] border-t border-[var(--border-subtle)] py-6 px-8">
          <div className="max-w-[1100px] mx-auto flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-2.5 font-[var(--font-display)] font-bold text-[16px] text-[var(--text-primary)]">
              <div className="w-8 h-8 flex items-center justify-center rounded-[var(--radius-sm)]
                              bg-[var(--accent-green-dim)] border border-[var(--border-accent)] text-[var(--accent-green)]">
                <Code2 size={14} />
              </div>
              <span>CP Roadmap</span>
            </div>
            <span className="text-[13px] text-[var(--text-muted)]">
              Built for competitive programmers, by competitive programmers.
            </span>
          </div>
        </footer>
      </div>
    </>
  )
}