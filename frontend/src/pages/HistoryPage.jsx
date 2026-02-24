import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../utils/api'
import { ChevronRight, Zap, Clock, TrendingUp, Search } from 'lucide-react'

export default function HistoryPage() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch]   = useState('')

  useEffect(() => {
    api.get('/roadmap/history').then(r => setHistory(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [])

  const filtered = history.filter(h =>
    !search || h.user_level?.toLowerCase().includes(search.toLowerCase()) || String(h.id).includes(search)
  )

  return (
    /* .history */
    <div className="flex flex-col gap-6">

      {/* .history__header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          {/* .history__title */}
          <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)] mb-1.5">
            Roadmap History
          </h1>
          {/* .history__sub */}
          <p className="text-[14px] text-[var(--text-secondary)]">
            All your previously generated roadmaps in one place.
          </p>
        </div>

        {/* .btn-generate — same as Dashboard */}
        <Link
          to="/generate"
          className="inline-flex items-center gap-2 px-5 py-[11px] whitespace-nowrap
                     bg-[var(--accent-green)] text-[#080c14] rounded-[var(--radius-md)]
                     text-[14px] font-semibold transition-all duration-200
                     hover:bg-[#00d48c] hover:shadow-[0_0_20px_rgba(0,245,160,0.25)] hover:-translate-y-px"
        >
          <Zap size={15} />
          New Roadmap
        </Link>
      </div>

      {/* Search — .history__search-wrap */}
      <div className="relative max-w-[360px]">
        {/* .history__search-icon */}
        <Search
          size={15}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none"
        />
        {/* .form-input .history__search — pl-[38px] overrides normal padding */}
        <input
          type="text"
          placeholder="Search by ID or level..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full pl-[38px] pr-3.5 py-[11px]
                     bg-[var(--bg-secondary)] border border-[var(--border-medium)]
                     rounded-[var(--radius-md)] text-[14px] text-[var(--text-primary)]
                     placeholder:text-[var(--text-muted)]
                     outline-none transition-all duration-200
                     focus:border-[var(--accent-green)] focus:shadow-[0_0_0_3px_var(--accent-green-dim)]"
        />
      </div>

      {/* States */}
      {loading ? (
        /* .history__list (skeleton) */
        <div className="flex flex-col gap-2.5">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="skeleton h-[100px] rounded-[var(--radius-lg)]" />
          ))}
        </div>

      ) : filtered.length === 0 ? (
        /* .history__empty */
        <div className="flex flex-col items-center justify-center text-center px-5 py-20 gap-3 text-[var(--text-muted)]">
          <Zap size={32} className="text-[var(--text-muted)]" />
          <p className="text-[16px] text-[var(--text-secondary)]">
            {search ? 'No results found.' : "You haven't generated any roadmaps yet."}
          </p>
          {!search && (
            /* .dashboard__empty-btn — reused style */
            <Link
              to="/generate"
              className="inline-block mt-1 px-5 py-2 text-[14px] font-semibold
                         bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                         border border-[var(--border-accent)] rounded-[var(--radius-md)]
                         transition-all duration-200
                         hover:bg-[var(--accent-green)] hover:text-[#080c14]"
            >
              Generate Your First Roadmap
            </Link>
          )}
        </div>

      ) : (
        /* .history__list */
        <div className="flex flex-col gap-2.5">
          {filtered.map((item, i) => (
            <HistoryCard key={item.id} item={item} index={i} />
          ))}
        </div>
      )}

    </div>
  )
}

/* ── HistoryCard ── */
function HistoryCard({ item, index }) {
  const weakTopics = item.weak_topics?.slice(0, 4) || []

  return (
    /* .history-card */
    <Link
      to={`/roadmap/${item.id}`}
      style={{ animationDelay: `${index * 0.05}s` }}
      className="flex items-center gap-5 p-5
                 bg-[var(--bg-card)] border border-[var(--border-subtle)]
                 rounded-[var(--radius-lg)]
                 transition-all duration-200
                 [animation:fadeIn_0.4s_ease_both]
                 hover:border-[var(--border-medium)] hover:bg-[var(--bg-card-hover)]
                 hover:-translate-y-px hover:shadow-[var(--shadow-card)]"
    >
      {/* .history-card__left */}
      <div className="flex items-center gap-5 flex-1">

        {/* .history-card__id — hidden on mobile @600px */}
        <div className="hidden sm:block font-[var(--font-mono)] text-[20px] font-bold
                        text-[var(--text-muted)] w-12 text-right shrink-0">
          #{item.id}
        </div>

        {/* .history-card__body */}
        <div className="flex-1">

          {/* .history-card__top */}
          <div className="flex items-center gap-3 mb-2.5 flex-wrap">

            {/* .badge .badge--green */}
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full
                             bg-[var(--accent-green-dim)] text-[var(--accent-green)]">
              {item.user_level || 'Beginner'}
            </span>

            {/* .history-card__count */}
            <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
              <Zap size={11} /> {item.problems?.length || 0} problems
            </span>

            {/* .history-card__date */}
            <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
              <Clock size={11} />
              {new Date(item.created_at).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric',
              })}
            </span>

            {/* .history-card__penalty */}
            {item.contest_penalty != null && (
              <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
                <TrendingUp size={11} /> penalty: {item.contest_penalty.toFixed(3)}
              </span>
            )}
          </div>

          {/* .history-card__topics */}
          {weakTopics.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {weakTopics.map((t, j) => (
                /* .history-card__topic */
                <span
                  key={j}
                  className="text-[11px] px-2.5 py-[3px]
                             bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                             rounded-full text-[var(--text-muted)]"
                >
                  {t}
                </span>
              ))}
              {item.weak_topics?.length > 4 && (
                /* .history-card__topic .history-card__topic--more */
                <span className="text-[11px] px-2.5 py-[3px]
                                 bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                                 rounded-full text-[var(--text-muted)] opacity-60">
                  +{item.weak_topics.length - 4} more
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* .history-card__arrow */}
      <ChevronRight size={16} className="text-[var(--text-muted)] shrink-0" />
    </Link>
  )
}