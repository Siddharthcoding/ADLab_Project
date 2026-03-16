import React, { useEffect, useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../utils/api'
import { Zap, History, TrendingUp, Clock, ChevronRight, AlertTriangle, BookOpen, ArrowUpRight } from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuth()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [visible, setVisible] = useState(false)
  const latest = history[0]

  useEffect(() => {
    api.get('/roadmap/history').then(r => setHistory(r.data)).catch(() => {}).finally(() => setLoading(false))
    setTimeout(() => setVisible(true), 50)
  }, [])

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return (
    <>
      <style>{`
        @keyframes slideUpFade {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes countUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmerLine {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .dash-enter { animation: slideUpFade 0.5s cubic-bezier(0.16,1,0.3,1) both; }
        .d1 { animation-delay: 0.05s; }
        .d2 { animation-delay: 0.12s; }
        .d3 { animation-delay: 0.19s; }
        .d4 { animation-delay: 0.26s; }
        .d5 { animation-delay: 0.33s; }
        .d6 { animation-delay: 0.40s; }
        .stat-val { animation: countUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.4s both; }
        .card-hover {
          transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
        }
        .card-hover:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08);
        }
        .history-row {
          transition: all 0.2s ease;
          border-left: 2px solid transparent;
        }
        .history-row:hover {
          border-left-color: var(--accent-green);
          padding-left: 14px;
          background: var(--bg-elevated);
        }
        .topic-tag { transition: all 0.2s ease; }
        .topic-tag:hover {
          background: var(--accent-green-dim);
          border-color: var(--accent-green);
          color: var(--accent-green);
          transform: translateY(-1px);
        }
        .gen-btn {
          position: relative;
          overflow: hidden;
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .gen-btn::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
          transform: translateX(-100%);
          transition: transform 0.6s ease;
        }
        .gen-btn:hover::after { transform: translateX(100%); }
        .gen-btn:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(0,245,160,0.35); }
      `}</style>

      <div className="flex flex-col gap-6">

        {/* Header */}
        <div className="dash-enter d1 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <p className="text-[14px] text-[var(--text-muted)] font-[var(--font-mono)] mb-1">{greeting},</p>
            <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)]">
              {user?.username} <span className="inline-block animate-[bounce_1s_ease_0.5s_1]">👋</span>
            </h1>
          </div>
          <Link to="/generate" className="gen-btn inline-flex items-center gap-2 px-5 py-[11px] whitespace-nowrap
                     bg-[var(--accent-green)] text-[#080c14] rounded-[var(--radius-md)]
                     text-[14px] font-semibold">
            <Zap size={16} className="relative z-[1]" />
            <span className="relative z-[1]">Generate Roadmap</span>
          </Link>
        </div>

        {/* Stats */}
        <div className="dash-enter d2 grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard icon={<History size={18} />}    label="Roadmaps Generated" value={loading ? '—' : history.length}                                color="blue"   loading={loading} />
          <StatCard icon={<BookOpen size={18} />}   label="Latest Problems"    value={loading ? '—' : latest?.problems?.length ?? 0}                 color="green"  loading={loading} />
          <StatCard icon={<TrendingUp size={18} />} label="User Level"         value={loading ? '—' : latest?.user_level ?? 'N/A'}                   color="orange" small loading={loading} />
          <StatCard icon={<Clock size={18} />}      label="Last Generated"     value={loading ? '—' : latest ? timeAgo(latest.created_at) : 'Never'} color="purple" small loading={loading} />
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

          {/* Quick Actions */}
          <div className="dash-enter d3 card-hover bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6
                          relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px]
                            bg-gradient-to-r from-transparent via-[var(--accent-blue)] to-transparent opacity-30" />
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">Quick Actions</h2>
            </div>
            <div className="flex flex-col gap-3">
              <ActionCard to="/generate" icon={<Zap size={20} />}    title="Generate New Roadmap" desc="Connect LeetCode & Codeforces to get your personalized plan" accent="green" />
              <ActionCard to="/history"  icon={<History size={20} />} title="View History"         desc="Browse all your previously generated roadmaps"               accent="blue"  />
            </div>
          </div>

          {/* Recent Roadmaps */}
          <div className="dash-enter d4 card-hover bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6
                          relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px]
                            bg-gradient-to-r from-transparent via-[var(--accent-green)] to-transparent opacity-30" />
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">Recent Roadmaps</h2>
              {history.length > 0 && (
                <Link to="/history" className="flex items-center gap-1 text-[12px] text-[var(--accent-green)]
                                               transition-all duration-200 hover:gap-2 group">
                  View all
                  <ChevronRight size={14} className="group-hover:translate-x-0.5 transition-transform duration-200" />
                </Link>
              )}
            </div>

            {loading ? (
              <div className="flex flex-col gap-2">
                {[1,2,3].map(i => <div key={i} className="skeleton h-[60px] rounded-[var(--radius-sm)]" />)}
              </div>
            ) : history.length === 0 ? (
              <div className="flex flex-col items-center justify-center text-center px-5 py-10 gap-2 text-[var(--text-muted)]">
                <div className="w-12 h-12 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)]
                                flex items-center justify-center mb-1">
                  <AlertTriangle size={22} className="text-[var(--text-muted)]" />
                </div>
                <p className="text-[15px] font-semibold text-[var(--text-secondary)]">No roadmaps yet</p>
                <span className="text-[13px]">Generate your first roadmap to get started</span>
                <Link to="/generate" className="mt-3 inline-block px-5 py-2 text-[14px] font-semibold
                           bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                           border border-[var(--border-accent)] rounded-[var(--radius-md)]
                           transition-all duration-200 hover:bg-[var(--accent-green)] hover:text-[#080c14]
                           hover:-translate-y-px">
                  Generate Now
                </Link>
              </div>
            ) : (
              <div className="flex flex-col gap-0.5">
                {history.slice(0, 5).map((item, idx) => (
                  <Link to={`/roadmap/${item.id}`} key={item.id}
                    className="history-row flex items-center gap-3.5 px-3 py-3 rounded-[var(--radius-md)] cursor-pointer"
                    style={{ animationDelay: `${0.4 + idx * 0.06}s` }}>
                    <div className="flex items-center gap-3 flex-1">
                      <span className="font-[var(--font-mono)] text-[12px] text-[var(--text-muted)] w-7 shrink-0">#{item.id}</span>
                      <div>
                        <div className="flex items-center gap-2 mb-[3px]">
                          <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full
                                           bg-[var(--accent-green-dim)] text-[var(--accent-green)]">
                            {item.user_level || 'Beginner'}
                          </span>
                          <span className="text-[12px] text-[var(--text-muted)]">{item.problems?.length || 0} problems</span>
                        </div>
                        <div className="text-[11px] text-[var(--text-muted)] font-[var(--font-mono)]">{formatDate(item.created_at)}</div>
                      </div>
                    </div>
                    <ArrowUpRight size={14} className="text-[var(--text-muted)] ml-auto shrink-0 opacity-0 group-hover:opacity-100
                                                        transition-opacity duration-200" />
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Weak Topics */}
        {latest?.weak_topics?.length > 0 && (
          <div className="dash-enter d5 card-hover bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6
                          relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-[1px]
                            bg-gradient-to-r from-transparent via-[var(--accent-orange)] to-transparent opacity-30" />
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">Latest Weak Topics</h2>
              <Link to={`/roadmap/${latest.id}`}
                className="flex items-center gap-1 text-[12px] text-[var(--accent-green)]
                           transition-all duration-200 hover:gap-2 group">
                View full roadmap
                <ChevronRight size={14} className="group-hover:translate-x-0.5 transition-transform duration-200" />
              </Link>
            </div>
            <div className="flex flex-wrap gap-2.5">
              {latest.weak_topics.slice(0, 10).map((topic, i) => (
                <div key={i} className="topic-tag flex items-center gap-1.5 px-3 py-1.5
                           bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                           rounded-full text-[13px] text-[var(--text-secondary)] cursor-default"
                  style={{ animationDelay: `${0.5 + i * 0.04}s` }}>
                  <span className="font-[var(--font-mono)] text-[10px] text-[var(--text-muted)]">{i + 1}</span>
                  {topic}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

const iconColors = {
  green:  'bg-[var(--accent-green-dim)]  text-[var(--accent-green)]',
  blue:   'bg-[var(--accent-blue-dim)]   text-[var(--accent-blue)]',
  orange: 'bg-[var(--accent-orange-dim)] text-[var(--accent-orange)]',
  purple: 'bg-[var(--accent-purple-dim)] text-[var(--accent-purple)]',
}

function StatCard({ icon, label, value, color, small, loading }) {
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-5
                    transition-all duration-300 hover:border-[var(--border-medium)] hover:-translate-y-1
                    hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)] group relative overflow-hidden cursor-default">
      {/* Hover glow */}
      <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300
                       bg-gradient-to-br ${
                         color === 'green'  ? 'from-[rgba(0,245,160,0.03)]' :
                         color === 'blue'   ? 'from-[rgba(59,130,246,0.03)]' :
                         color === 'orange' ? 'from-[rgba(249,115,22,0.03)]' :
                                             'from-[rgba(168,85,247,0.03)]'
                       } to-transparent`} />
      <div className={`w-9 h-9 rounded-[var(--radius-sm)] flex items-center justify-center mb-3
                       transition-transform duration-300 group-hover:scale-110 ${iconColors[color]}`}>
        {icon}
      </div>
      <div className="stat-val font-[var(--font-display)] font-bold text-[var(--text-primary)] leading-none mb-1.5"
           style={{ fontSize: small ? '18px' : '28px' }}>
        {loading ? <span className="skeleton inline-block w-12 h-6 rounded" /> : value}
      </div>
      <div className="text-[12px] text-[var(--text-muted)]">{label}</div>
    </div>
  )
}

const actionIconColors = {
  green: 'bg-[var(--accent-green-dim)] text-[var(--accent-green)]',
  blue:  'bg-[var(--accent-blue-dim)]  text-[var(--accent-blue)]',
}

function ActionCard({ to, icon, title, desc, accent }) {
  return (
    <Link to={to}
      className="flex items-center gap-4 p-4 rounded-[var(--radius-md)]
                 border border-[var(--border-subtle)] transition-all duration-250 group
                 hover:border-[var(--border-medium)] hover:bg-[var(--bg-elevated)]
                 hover:translate-x-1">
      <div className={`w-10 h-10 rounded-[var(--radius-md)] flex items-center justify-center shrink-0
                       transition-transform duration-300 group-hover:scale-110 ${actionIconColors[accent]}`}>
        {icon}
      </div>
      <div className="flex-1">
        <div className="text-[14px] font-semibold text-[var(--text-primary)] mb-[3px]">{title}</div>
        <div className="text-[12px] text-[var(--text-muted)]">{desc}</div>
      </div>
      <ChevronRight size={16} className="text-[var(--text-muted)] ml-auto shrink-0
                                          opacity-0 group-hover:opacity-100
                                          group-hover:translate-x-1
                                          transition-all duration-200" />
    </Link>
  )
}

function timeAgo(dateStr) {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}