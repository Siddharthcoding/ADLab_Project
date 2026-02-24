import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../utils/api'
import { Zap, History, TrendingUp, Clock, ChevronRight, AlertTriangle, BookOpen } from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuth()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const latest = history[0]

  useEffect(() => {
    api.get('/roadmap/history').then(r => setHistory(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return (
    /* .dashboard */
    <div className="flex flex-col gap-6">

      {/* .dashboard__header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          {/* .dashboard__greeting */}
          <p className="text-[14px] text-[var(--text-muted)] font-[var(--font-mono)] mb-1">
            {greeting},
          </p>
          {/* .dashboard__title */}
          <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)]">
            {user?.username} <span>👋</span>
          </h1>
        </div>

        {/* .btn-generate */}
        <Link
          to="/generate"
          className="inline-flex items-center gap-2 px-5 py-[11px] whitespace-nowrap
                     bg-[var(--accent-green)] text-[#080c14] rounded-[var(--radius-md)]
                     text-[14px] font-semibold transition-all duration-200
                     hover:bg-[#00d48c] hover:shadow-[0_0_20px_rgba(0,245,160,0.25)] hover:-translate-y-px"
        >
          <Zap size={16} />
          Generate Roadmap
        </Link>
      </div>

      {/* .dashboard__stats — 4 cols → 2 cols @900px */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={<History size={18} />}    label="Roadmaps Generated" value={loading ? '—' : history.length}                              color="blue"   />
        <StatCard icon={<BookOpen size={18} />}   label="Latest Problems"    value={loading ? '—' : latest?.problems?.length ?? 0}               color="green"  />
        <StatCard icon={<TrendingUp size={18} />} label="User Level"         value={loading ? '—' : latest?.user_level ?? 'N/A'}                 color="orange" small />
        <StatCard icon={<Clock size={18} />}      label="Last Generated"     value={loading ? '—' : latest ? timeAgo(latest.created_at) : 'Never'} color="purple" small />
      </div>

      {/* .dashboard__grid — 2 cols → 1 col @900px */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Quick Actions — .dashboard__card */}
        <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6">
          {/* .dashboard__card-header */}
          <div className="flex items-center justify-between mb-5">
            <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">
              Quick Actions
            </h2>
          </div>
          {/* .dashboard__actions */}
          <div className="flex flex-col gap-3">
            <ActionCard to="/generate" icon={<Zap size={20} />}    title="Generate New Roadmap" desc="Connect LeetCode & Codeforces to get your personalized plan" accent="green" />
            <ActionCard to="/history"  icon={<History size={20} />} title="View History"         desc="Browse all your previously generated roadmaps"               accent="blue"  />
          </div>
        </div>

        {/* Recent Roadmaps — .dashboard__card */}
        <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6">
          {/* .dashboard__card-header */}
          <div className="flex items-center justify-between mb-5">
            <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">
              Recent Roadmaps
            </h2>
            {history.length > 0 && (
              /* .dashboard__card-link */
              <Link
                to="/history"
                className="flex items-center gap-1 text-[12px] text-[var(--accent-green)]
                           transition-[gap] duration-200 hover:gap-1.5"
              >
                View all <ChevronRight size={14} />
              </Link>
            )}
          </div>

          {loading ? (
            /* .dashboard__loading */
            <div className="flex flex-col gap-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="skeleton h-[60px] rounded-[var(--radius-sm)]" />
              ))}
            </div>

          ) : history.length === 0 ? (
            /* .dashboard__empty */
            <div className="flex flex-col items-center justify-center text-center px-5 py-10 gap-2 text-[var(--text-muted)]">
              <AlertTriangle size={28} />
              <p className="text-[15px] font-semibold text-[var(--text-secondary)] mt-2">No roadmaps yet</p>
              <span className="text-[13px]">Generate your first roadmap to get started</span>
              {/* .dashboard__empty-btn */}
              <Link
                to="/generate"
                className="mt-3 inline-block px-5 py-2 text-[14px] font-semibold
                           bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                           border border-[var(--border-accent)] rounded-[var(--radius-md)]
                           transition-all duration-200
                           hover:bg-[var(--accent-green)] hover:text-[#080c14]"
              >
                Generate Now
              </Link>
            </div>

          ) : (
            /* .dashboard__history-list */
            <div className="flex flex-col gap-0.5">
              {history.slice(0, 5).map(item => (
                /* .dashboard__history-item */
                <Link
                  to={`/roadmap/${item.id}`}
                  key={item.id}
                  className="flex items-center gap-3.5 px-3 py-3 rounded-[var(--radius-md)]
                             cursor-pointer transition-colors duration-200
                             hover:bg-[var(--bg-elevated)]"
                >
                  {/* .dashboard__history-left */}
                  <div className="flex items-center gap-3 flex-1">
                    {/* .dashboard__history-id */}
                    <span className="font-[var(--font-mono)] text-[12px] text-[var(--text-muted)] w-7 shrink-0">
                      #{item.id}
                    </span>
                    <div>
                      {/* .dashboard__history-meta */}
                      <div className="flex items-center gap-2 mb-[3px]">
                        {/* .badge .badge--green */}
                        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full
                                         bg-[var(--accent-green-dim)] text-[var(--accent-green)]">
                          {item.user_level || 'Beginner'}
                        </span>
                        {/* .dashboard__history-count */}
                        <span className="text-[12px] text-[var(--text-muted)]">
                          {item.problems?.length || 0} problems
                        </span>
                      </div>
                      {/* .dashboard__history-date */}
                      <div className="text-[11px] text-[var(--text-muted)] font-[var(--font-mono)]">
                        {formatDate(item.created_at)}
                      </div>
                    </div>
                  </div>
                  {/* .dashboard__history-arrow */}
                  <ChevronRight size={16} className="text-[var(--text-muted)] ml-auto shrink-0" />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Latest Weak Topics — .dashboard__card .dashboard__card--full */}
      {latest?.weak_topics?.length > 0 && (
        <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-6">
          {/* .dashboard__card-header */}
          <div className="flex items-center justify-between mb-5">
            <h2 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">
              Latest Weak Topics
            </h2>
            <Link
              to={`/roadmap/${latest.id}`}
              className="flex items-center gap-1 text-[12px] text-[var(--accent-green)]
                         transition-[gap] duration-200 hover:gap-1.5"
            >
              View full roadmap <ChevronRight size={14} />
            </Link>
          </div>
          {/* .dashboard__topics */}
          <div className="flex flex-wrap gap-2.5">
            {latest.weak_topics.slice(0, 10).map((topic, i) => (
              /* .dashboard__topic-tag */
              <div
                key={i}
                className="flex items-center gap-1.5 px-3 py-1.5
                           bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                           rounded-full text-[13px] text-[var(--text-secondary)]
                           transition-all duration-200
                           hover:border-[var(--border-accent)] hover:text-[var(--accent-green)]"
              >
                {/* .dashboard__topic-rank */}
                <span className="font-[var(--font-mono)] text-[10px] text-[var(--text-muted)]">
                  {i + 1}
                </span>
                {topic}
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}

/* ── StatCard ── */
const iconColors = {
  green:  'bg-[var(--accent-green-dim)]  text-[var(--accent-green)]',
  blue:   'bg-[var(--accent-blue-dim)]   text-[var(--accent-blue)]',
  orange: 'bg-[var(--accent-orange-dim)] text-[var(--accent-orange)]',
  purple: 'bg-[var(--accent-purple-dim)] text-[var(--accent-purple)]',
}

function StatCard({ icon, label, value, color, small }) {
  return (
    /* .stat-card .stat-card--{color} */
    <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-lg)] p-5
                    transition-all duration-200 hover:border-[var(--border-medium)]">
      {/* .stat-card__icon */}
      <div className={`w-9 h-9 rounded-[var(--radius-sm)] flex items-center justify-center mb-3 ${iconColors[color]}`}>
        {icon}
      </div>
      {/* .stat-card__value */}
      <div
        className="font-[var(--font-display)] font-bold text-[var(--text-primary)] leading-none mb-1.5"
        style={{ fontSize: small ? '18px' : '28px' }}
      >
        {value}
      </div>
      {/* .stat-card__label */}
      <div className="text-[12px] text-[var(--text-muted)]">{label}</div>
    </div>
  )
}

/* ── ActionCard ── */
const actionIconColors = {
  green: 'bg-[var(--accent-green-dim)] text-[var(--accent-green)]',
  blue:  'bg-[var(--accent-blue-dim)]  text-[var(--accent-blue)]',
}

function ActionCard({ to, icon, title, desc, accent }) {
  return (
    /* .action-card .action-card--{accent} */
    <Link
      to={to}
      className="flex items-center gap-4 p-4 rounded-[var(--radius-md)]
                 border border-[var(--border-subtle)] transition-all duration-200
                 hover:border-[var(--border-medium)] hover:bg-[var(--bg-elevated)]"
    >
      {/* .action-card__icon */}
      <div className={`w-10 h-10 rounded-[var(--radius-md)] flex items-center justify-center shrink-0 ${actionIconColors[accent]}`}>
        {icon}
      </div>
      <div className="flex-1">
        {/* .action-card__title */}
        <div className="text-[14px] font-semibold text-[var(--text-primary)] mb-[3px]">{title}</div>
        {/* .action-card__desc */}
        <div className="text-[12px] text-[var(--text-muted)]">{desc}</div>
      </div>
      {/* .action-card__arrow */}
      <ChevronRight size={16} className="text-[var(--text-muted)] ml-auto shrink-0" />
    </Link>
  )
}

/* ── Helpers ── */
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