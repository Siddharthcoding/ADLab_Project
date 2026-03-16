import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../utils/api'
import { ChevronRight, Zap, Clock, TrendingUp, Search, X } from 'lucide-react'

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
    <>
      <style>{`
        @keyframes slideUpFade {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .h-enter { animation: slideUpFade 0.45s cubic-bezier(0.16,1,0.3,1) both; }
        .h1 { animation-delay: 0.04s; }
        .h2 { animation-delay: 0.1s; }
        .h3 { animation-delay: 0.16s; }
        .history-card-wrap {
          transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
          border-left: 3px solid transparent;
        }
        .history-card-wrap:hover {
          transform: translateX(4px);
          border-left-color: var(--accent-green);
          box-shadow: 0 4px 24px rgba(0,0,0,0.35), -4px 0 12px rgba(0,245,160,0.07);
        }
        .search-clear { transition: all 0.2s ease; }
        .search-clear:hover { color: var(--text-primary); transform: scale(1.1); }
        .gen-btn-hist {
          position: relative; overflow: hidden;
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .gen-btn-hist::after {
          content: '';
          position: absolute; inset: 0;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
          transform: translateX(-100%);
        }
        .gen-btn-hist:hover::after { transform: translateX(100%); transition: transform 0.6s ease; }
        .gen-btn-hist:hover { transform: translateY(-2px); box-shadow: 0 0 28px rgba(0,245,160,0.35); }
        .empty-icon { animation: slideUpFade 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s both; }
      `}</style>

      <div className="flex flex-col gap-6">

        {/* Header */}
        <div className="h-enter h1 flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h1 className="font-[var(--font-display)] text-[28px] font-extrabold text-[var(--text-primary)] mb-1.5">
              Roadmap History
            </h1>
            <p className="text-[14px] text-[var(--text-secondary)]">
              All your previously generated roadmaps in one place.
            </p>
          </div>
          <Link to="/generate" className="gen-btn-hist inline-flex items-center gap-2 px-5 py-[11px] whitespace-nowrap
                     bg-[var(--accent-green)] text-[#080c14] rounded-[var(--radius-md)]
                     text-[14px] font-semibold">
            <Zap size={15} className="relative z-[1]" />
            <span className="relative z-[1]">New Roadmap</span>
          </Link>
        </div>

        {/* Search */}
        <div className="h-enter h2 relative max-w-[360px]">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none
                                       transition-colors duration-200" />
          <input
            type="text" placeholder="Search by ID or level..."
            value={search} onChange={e => setSearch(e.target.value)}
            className="w-full pl-[38px] pr-9 py-[11px]
                       bg-[var(--bg-secondary)] border border-[var(--border-medium)]
                       rounded-[var(--radius-md)] text-[14px] text-[var(--text-primary)]
                       placeholder:text-[var(--text-muted)]
                       outline-none transition-all duration-200
                       focus:border-[var(--accent-green)] focus:shadow-[0_0_0_3px_var(--accent-green-dim),0_0_20px_rgba(0,245,160,0.1)]
                       hover:border-[rgba(255,255,255,0.18)]" />
          {search && (
            <button onClick={() => setSearch('')}
              className="search-clear absolute right-3 top-1/2 -translate-y-1/2
                         text-[var(--text-muted)] bg-transparent p-0.5">
              <X size={13} />
            </button>
          )}
          {/* Active filter count badge */}
          {search && filtered.length > 0 && (
            <span className="absolute -top-2 -right-2 text-[10px] font-bold font-[var(--font-mono)]
                             bg-[var(--accent-green)] text-[#080c14] rounded-full w-5 h-5
                             flex items-center justify-center">
              {filtered.length}
            </span>
          )}
        </div>

        {/* List */}
        <div className="h-enter h3">
          {loading ? (
            <div className="flex flex-col gap-2.5">
              {[1,2,3,4,5].map(i => <div key={i} className="skeleton h-[100px] rounded-[var(--radius-lg)]" />)}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center px-5 py-20 gap-3 text-[var(--text-muted)]">
              <div className="empty-icon w-16 h-16 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)]
                              flex items-center justify-center mb-1">
                <Zap size={28} className="text-[var(--text-muted)]" />
              </div>
              <p className="text-[16px] text-[var(--text-secondary)] font-medium">
                {search ? 'No results found.' : "You haven't generated any roadmaps yet."}
              </p>
              <p className="text-[13px]">
                {search ? `No roadmaps match "${search}"` : 'Generate your first roadmap to get started.'}
              </p>
              {!search && (
                <Link to="/generate"
                  className="inline-block mt-1 px-5 py-2 text-[14px] font-semibold
                             bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                             border border-[var(--border-accent)] rounded-[var(--radius-md)]
                             transition-all duration-200 hover:bg-[var(--accent-green)] hover:text-[#080c14]
                             hover:-translate-y-px">
                  Generate Your First Roadmap
                </Link>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-2.5">
              {filtered.map((item, i) => <HistoryCard key={item.id} item={item} index={i} />)}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

function HistoryCard({ item, index }) {
  const weakTopics = item.weak_topics?.slice(0, 4) || []

  return (
    <Link to={`/roadmap/${item.id}`}
      style={{ animationDelay: `${index * 0.05}s` }}
      className="history-card-wrap flex items-center gap-5 p-5
                 bg-[var(--bg-card)] border border-[var(--border-subtle)]
                 rounded-[var(--radius-lg)]
                 [animation:slideUpFade_0.4s_cubic-bezier(0.16,1,0.3,1)_both]
                 group">

      <div className="flex items-center gap-5 flex-1">
        {/* ID */}
        <div className="hidden sm:flex flex-col items-center gap-1 w-14 shrink-0">
          <span className="font-[var(--font-mono)] text-[20px] font-bold text-[var(--text-muted)]
                           group-hover:text-[var(--accent-green)] transition-colors duration-300">
            #{item.id}
          </span>
        </div>

        {/* Body */}
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2.5 flex-wrap">
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full
                             bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                             border border-[rgba(0,245,160,0.2)]">
              {item.user_level || 'Beginner'}
            </span>
            <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
              <Zap size={11} className="text-[var(--accent-green)] opacity-70" />
              {item.problems?.length || 0} problems
            </span>
            <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
              <Clock size={11} />
              {new Date(item.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
            </span>
            {item.contest_penalty != null && (
              <span className="flex items-center gap-1 text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
                <TrendingUp size={11} /> penalty: {item.contest_penalty.toFixed(3)}
              </span>
            )}
          </div>
          {weakTopics.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {weakTopics.map((t, j) => (
                <span key={j} className="text-[11px] px-2.5 py-[3px]
                               bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                               rounded-full text-[var(--text-muted)]
                               transition-colors duration-200
                               group-hover:border-[rgba(255,255,255,0.1)]">
                  {t}
                </span>
              ))}
              {item.weak_topics?.length > 4 && (
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

      <ChevronRight size={16}
        className="text-[var(--text-muted)] shrink-0
                   group-hover:text-[var(--accent-green)]
                   group-hover:translate-x-1
                   transition-all duration-200" />
    </Link>
  )
}