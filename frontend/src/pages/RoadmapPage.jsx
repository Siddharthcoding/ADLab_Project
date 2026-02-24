import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import api from '../utils/api'
import {
  ExternalLink, Brain, AlertTriangle, Clock,
  TrendingUp, ChevronLeft, Zap, Network, Tag
} from 'lucide-react'

export default function RoadmapPage() {
  const { id } = useParams()
  const [roadmap, setRoadmap]   = useState(null)
  const [loading, setLoading]   = useState(true)
  const [activeTab, setActiveTab] = useState('problems')

  useEffect(() => {
    api.get(`/roadmap/${id}`).then(r => setRoadmap(r.data)).catch(() => {}).finally(() => setLoading(false))
  }, [id])

  if (loading) return <LoadingSkeleton />
  if (!roadmap) return (
    /* .roadmap-error */
    <div className="p-10 text-center text-[var(--text-muted)]">Roadmap not found.</div>
  )

  const tabs = [
    { id: 'problems',  label: `Problems (${roadmap.problems?.length || 0})` },
    { id: 'session',   label: 'Session Plan',    show: roadmap.session_plan?.length > 0 },
    { id: 'calendar',  label: '7-Day Calendar',  show: roadmap.daily_calendar?.length > 0 },
    { id: 'retention', label: 'Retention',       show: !!roadmap.retention_data },
    { id: 'gnn',       label: 'GNN Gaps',        show: roadmap.gnn_data?.hidden_gaps?.length > 0 },
  ].filter(t => t.show !== false)

  return (
    /* .roadmap */
    <div className="flex flex-col gap-5">

      {/* .roadmap__header */}
      <div className="flex flex-col gap-2.5">

        {/* .roadmap__back */}
        <Link
          to="/history"
          className="inline-flex items-center gap-1.5 w-fit
                     text-[13px] text-[var(--text-muted)] hover:text-[var(--text-secondary)]
                     transition-colors duration-200"
        >
          <ChevronLeft size={16} /> Back to History
        </Link>

        {/* .roadmap__meta */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* .roadmap__title */}
          <h1 className="font-[var(--font-display)] text-[26px] font-extrabold text-[var(--text-primary)]">
            Roadmap #{id}
          </h1>
          {/* .badge .badge--green */}
          <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full
                           bg-[var(--accent-green-dim)] text-[var(--accent-green)]">
            {roadmap.user_level || 'Beginner'}
          </span>
        </div>

        {/* .roadmap__stats-row */}
        <div className="flex gap-3 flex-wrap">
          <MetaBadge icon={<Zap size={13} />}       label={`${roadmap.problems?.length || 0} problems`} />
          <MetaBadge icon={<TrendingUp size={13} />} label={`Penalty: ${roadmap.contest_penalty?.toFixed(3) || 'N/A'}`} />
          <MetaBadge icon={<Clock size={13} />}      label={new Date(roadmap.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} />
        </div>
      </div>

      {/* .roadmap__weak */}
      {roadmap.weak_topics?.length > 0 && (
        <div className="bg-[rgba(249,115,22,0.05)] border border-[rgba(249,115,22,0.2)]
                        rounded-[var(--radius-lg)] px-5 py-4">
          {/* .roadmap__section-title */}
          <div className="flex items-center gap-2 text-[13px] font-semibold text-[var(--text-secondary)] mb-3">
            <AlertTriangle size={15} className="text-[var(--accent-orange)]" />
            Weak Topics Detected
          </div>
          {/* .roadmap__topics */}
          <div className="flex flex-wrap gap-2">
            {roadmap.weak_topics.map((t, i) => (
              /* .roadmap__topic-chip */
              <span key={i} className="flex items-center gap-1.5 px-3 py-[5px]
                                       bg-[var(--accent-orange-dim)] border border-[rgba(249,115,22,0.2)]
                                       rounded-full text-[12px] text-[var(--accent-orange)]">
                {/* .roadmap__topic-rank */}
                <span className="font-[var(--font-mono)] text-[10px] text-[var(--accent-orange)] opacity-70">
                  {i + 1}
                </span>
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* .roadmap__insights */}
      {roadmap.ml_insights && (
        <div className="flex gap-3 px-5 py-4
                        bg-[var(--accent-purple-dim)] border border-[rgba(168,85,247,0.2)]
                        rounded-[var(--radius-lg)] text-[14px] text-[var(--text-secondary)] leading-[1.6]">
          <Brain size={15} className="text-[var(--accent-purple)] shrink-0 mt-0.5" />
          <p>{roadmap.ml_insights}</p>
        </div>
      )}

      {/* .roadmap__tabs */}
      <div className="flex gap-1 flex-wrap bg-[var(--bg-card)] border border-[var(--border-subtle)]
                      rounded-[var(--radius-lg)] p-1.5">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={[
              'px-4 py-2 rounded-[var(--radius-md)] text-[13px] font-medium whitespace-nowrap',
              'transition-all duration-200',
              /* mobile: smaller padding */
              'sm:px-4 sm:py-2 px-3 py-[7px] sm:text-[13px] text-[12px]',
              activeTab === t.id
                ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)]'
                : 'bg-none text-[var(--text-muted)] hover:text-[var(--text-secondary)]',
            ].join(' ')}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* .roadmap__content */}
      <div>
        {activeTab === 'problems'  && <ProblemsTab  problems={roadmap.problems} />}
        {activeTab === 'session'   && <SessionTab   session={roadmap.session_plan} />}
        {activeTab === 'calendar'  && <CalendarTab  calendar={roadmap.daily_calendar} />}
        {activeTab === 'retention' && <RetentionTab data={roadmap.retention_data} />}
        {activeTab === 'gnn'       && <GNNTab       data={roadmap.gnn_data} />}
      </div>

    </div>
  )
}

/* ── MetaBadge ── .roadmap__meta-badge */
function MetaBadge({ icon, label }) {
  return (
    <div className="flex items-center gap-1.5 px-3 py-[5px]
                    bg-[var(--bg-card)] border border-[var(--border-subtle)]
                    rounded-full text-[12px] text-[var(--text-secondary)] font-[var(--font-mono)]">
      {icon}
      <span>{label}</span>
    </div>
  )
}

/* ══ PROBLEMS TAB ══ */
function ProblemsTab({ problems }) {
  const [filter, setFilter] = useState('all')
  const filtered = filter === 'all' ? problems : problems.filter(p => p.source === filter)

  return (
    /* .tab-problems */
    <div className="flex flex-col gap-4">

      {/* .tab-problems__filters */}
      <div className="flex gap-2">
        {['all', 'LeetCode', 'Codeforces'].map(s => (
          /* .tab-filter [.tab-filter--active] */
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={[
              'px-4 py-1.5 rounded-full text-[13px] border transition-all duration-200',
              filter === s
                ? 'bg-[var(--accent-green-dim)] border-[var(--border-accent)] text-[var(--accent-green)]'
                : 'bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)] hover:border-[var(--border-medium)] hover:text-[var(--text-secondary)]',
            ].join(' ')}
          >
            {s}
          </button>
        ))}
      </div>

      {/* .tab-problems__list */}
      <div className="flex flex-col gap-2">
        {filtered.map((p, i) => <ProblemCard key={i} problem={p} rank={i + 1} />)}
      </div>
    </div>
  )
}

/* ── ProblemCard ── */
const diffBadgeCls = {
  green:  'bg-[var(--accent-green-dim)]       text-[var(--accent-green)]',
  blue:   'bg-[var(--accent-blue-dim)]        text-[var(--accent-blue)]',
  orange: 'bg-[var(--accent-orange-dim)]      text-[var(--accent-orange)]',
  red:    'bg-[rgba(239,68,68,0.15)]          text-[var(--accent-red)]',
}

function ProblemCard({ problem: p, rank }) {
  const isLC = p.source === 'LeetCode'
  const diffKey = p.difficulty <= 800 ? 'green' : p.difficulty <= 1400 ? 'blue' : p.difficulty <= 2000 ? 'orange' : 'red'

  return (
    /* .problem-card */
    <div className="flex gap-3 bg-[var(--bg-card)] border border-[var(--border-subtle)]
                    rounded-[var(--radius-lg)] p-4
                    transition-all duration-200 hover:border-[var(--border-medium)]">

      {/* .problem-card__rank */}
      <div className="font-[var(--font-mono)] text-[12px] text-[var(--text-muted)]
                      w-6 pt-0.5 shrink-0 text-right">
        {rank}
      </div>

      {/* .problem-card__body */}
      <div className="flex-1 flex flex-col gap-2">

        {/* .problem-card__top */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          {/* .problem-card__name-row */}
          <div className="flex items-center gap-2.5 flex-1">
            {/* .source-badge --lc / --cf */}
            <span className={[
              'text-[10px] font-bold px-2 py-[3px] rounded-[4px] font-[var(--font-mono)] shrink-0',
              isLC
                ? 'bg-[rgba(255,161,22,0.15)] text-[#FFA116]'
                : 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)]',
            ].join(' ')}>
              {isLC ? 'LC' : 'CF'}
            </span>
            {/* .problem-card__name */}
            <span className="text-[14px] font-semibold text-[var(--text-primary)]">{p.name}</span>
          </div>

          {/* .problem-card__right */}
          <div className="flex items-center gap-2 shrink-0">
            {/* .diff-badge */}
            <span className={`text-[11px] font-semibold px-2.5 py-[3px] rounded-full font-[var(--font-mono)] ${diffBadgeCls[diffKey]}`}>
              {p.difficulty}
            </span>
            {/* .problem-card__link */}
            {p.link && (
              <a href={p.link} target="_blank" rel="noopener noreferrer"
                 className="text-[var(--text-muted)] hover:text-[var(--accent-green)] p-1 transition-colors duration-200">
                <ExternalLink size={13} />
              </a>
            )}
          </div>
        </div>

        {/* .problem-card__insight */}
        {p.ml_explanation && (
          <div className="flex items-center gap-1.5 text-[12px] text-[var(--text-muted)] italic">
            <Brain size={11} className="shrink-0" />
            {p.ml_explanation}
          </div>
        )}

        {/* .problem-card__tags */}
        <div className="flex flex-wrap gap-1.5">
          {p.tags?.slice(0, 5).map((tag, i) => (
            <span key={i} className="text-[11px] px-2 py-[3px] bg-[var(--bg-elevated)] rounded-[4px] text-[var(--text-muted)]">
              {tag}
            </span>
          ))}
        </div>

        {/* .problem-card__metrics */}
        <div className="flex gap-2 flex-wrap">
          <MetricPill label="Priority"  value={`${(p.ml_priority * 100).toFixed(0)}%`} />
          <MetricPill label="Forgotten" value={`${(p.forgetting_urgency * 100).toFixed(0)}%`} orange />
          <MetricPill label="Est."      value={`${p.est_minutes}m`} />
        </div>
      </div>
    </div>
  )
}

/* ── MetricPill ── .metric-pill */
function MetricPill({ label, value, orange }) {
  return (
    <div className="flex gap-1 text-[11px] px-2 py-[3px] rounded-[4px]
                    bg-[var(--bg-secondary)] border border-[var(--border-subtle)]">
      <span className="text-[var(--text-muted)]">{label}</span>
      <span className={`font-semibold font-[var(--font-mono)] ${orange ? 'text-[var(--accent-orange)]' : 'text-[var(--text-secondary)]'}`}>
        {value}
      </span>
    </div>
  )
}

/* ══ SESSION TAB ══ */
function SessionTab({ session }) {
  if (!session?.length) return <EmptyTab message="No session plan available." />
  return (
    <div>
      {/* .tab-session__desc */}
      <p className="text-[14px] text-[var(--text-secondary)] leading-[1.6] mb-4">
        Optimal problem ordering for your practice session based on SM-2 spaced repetition:
      </p>
      {/* .tab-session__list */}
      <div className="flex flex-col gap-2">
        {session.map((p, i) => (
          /* .session-item */
          <div key={i} className="flex items-center gap-4 px-4 py-3.5
                                   bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                   rounded-[var(--radius-md)] transition-all duration-200
                                   hover:border-[var(--border-medium)]">
            {/* .session-item__order */}
            <div className="w-7 h-7 rounded-full shrink-0 flex items-center justify-center
                            bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                            text-[12px] font-[var(--font-mono)] text-[var(--accent-green)]">
              {i + 1}
            </div>
            {/* .session-item__body */}
            <div className="flex-1">
              {/* .session-item__name */}
              <div className="text-[14px] font-semibold text-[var(--text-primary)] mb-1">{p.name}</div>
              {/* .session-item__meta */}
              <div className="flex items-center gap-2">
                <span className={[
                  'text-[10px] font-bold px-2 py-[3px] rounded-[4px] font-[var(--font-mono)]',
                  p.source === 'LeetCode'
                    ? 'bg-[rgba(255,161,22,0.15)] text-[#FFA116]'
                    : 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)]',
                ].join(' ')}>
                  {p.source}
                </span>
                {/* .session-item__reason */}
                <span className="text-[12px] text-[var(--text-muted)]">{p.session_reason}</span>
              </div>
            </div>
            {/* .session-item__time */}
            <div className="flex items-center gap-1 font-[var(--font-mono)] text-[12px] text-[var(--text-muted)] shrink-0">
              <Clock size={12} /> {p.est_minutes}m
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ══ CALENDAR TAB ══ */
function CalendarTab({ calendar }) {
  if (!calendar?.length) return <EmptyTab message="No calendar available." />
  return (
    /* .tab-calendar — grid auto-fill → 1 col on mobile */
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {calendar.map((day, i) => (
        /* .calendar-day */
        <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                rounded-[var(--radius-lg)] p-5
                                transition-all duration-200 hover:border-[var(--border-medium)]">
          {/* .calendar-day__header */}
          <div className="flex items-center gap-2.5 mb-2.5 flex-wrap">
            {/* .calendar-day__num */}
            <span className="font-[var(--font-mono)] text-[12px] font-semibold px-2.5 py-1
                             bg-[var(--bg-elevated)] rounded-full text-[var(--text-muted)]">
              Day {day.day}
            </span>
            {/* .calendar-day__label */}
            <span className="text-[14px] font-semibold text-[var(--text-primary)] flex-1">{day.label}</span>
            {/* .calendar-day__roi */}
            {day.roi && (
              <span className="font-[var(--font-mono)] text-[12px] text-[var(--accent-green)]">
                +{day.roi.toFixed(1)} pts/hr
              </span>
            )}
          </div>
          {/* .calendar-day__goal */}
          <p className="text-[13px] text-[var(--text-secondary)] mb-3 leading-[1.5]">{day.goal}</p>
          {/* .calendar-day__topics */}
          {day.focus_topics?.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <Tag size={11} className="text-[var(--text-muted)] shrink-0" />
              {day.focus_topics.slice(0, 4).map((t, j) => (
                /* .calendar-day__topic */
                <span key={j} className="text-[11px] px-2 py-[3px] bg-[var(--accent-blue-dim)] rounded-full text-[var(--accent-blue)]">
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

/* ══ RETENTION TAB ══ */
function RetentionTab({ data }) {
  if (!data) return <EmptyTab message="No retention data available." />
  const atRisk = data.at_risk || []
  return (
    <div>
      {/* .tab-section-title */}
      <h3 className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)] mb-4">
        At-Risk Topics (Need Review)
      </h3>
      {atRisk.length === 0 ? (
        <p className="text-[14px] text-[var(--text-muted)]">No at-risk topics. Great job!</p>
      ) : (
        /* .retention-list */
        <div className="flex flex-col gap-4">
          {atRisk.map((item, i) => {
            const barColor = item.retention > 0.5
              ? 'bg-[var(--accent-blue)]'
              : item.retention > 0.2
                ? 'bg-[var(--accent-orange)]'
                : 'bg-[var(--accent-red)]'
            return (
              /* .retention-item */
              <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                      rounded-[var(--radius-md)] p-4">
                {/* .retention-item__top */}
                <div className="flex justify-between mb-2.5">
                  <span className="text-[14px] font-semibold text-[var(--text-primary)]">{item.tag}</span>
                  <span className="text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
                    {item.last_seen_days?.toFixed(0)} days ago
                  </span>
                </div>
                {/* .retention-bar */}
                <div className="h-1.5 bg-[var(--bg-elevated)] rounded-full overflow-hidden mb-1.5">
                  {/* .retention-bar__fill */}
                  <div
                    className={`h-full rounded-full transition-[width] duration-1000 ${barColor}`}
                    style={{ width: `${item.retention * 100}%` }}
                  />
                </div>
                {/* .retention-item__pct */}
                <span className="text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
                  {(item.retention * 100).toFixed(0)}% retained
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/* ══ GNN TAB ══ */
function GNNTab({ data }) {
  if (!data?.hidden_gaps?.length) return <EmptyTab message="No hidden gaps detected. Your knowledge graph looks solid!" />
  return (
    <div>
      {/* .tab-gnn__intro */}
      <div className="flex items-start gap-2.5 px-4 py-3.5 mb-5
                      bg-[var(--accent-purple-dim)] border border-[rgba(168,85,247,0.2)]
                      rounded-[var(--radius-md)]
                      text-[14px] text-[var(--text-secondary)] leading-[1.6]">
        <Network size={16} className="text-[var(--accent-purple)] shrink-0 mt-0.5" />
        <p>Our Graph Neural Network analyzed your prerequisite relationships and found these hidden gaps:</p>
      </div>

      {/* .gnn-list */}
      <div className="flex flex-col gap-3">
        {data.hidden_gaps.map((gap, i) => (
          /* .gnn-item */
          <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                   rounded-[var(--radius-lg)] p-5">
            {/* .gnn-item__header */}
            <div className="mb-3.5">
              {/* .gnn-item__topic */}
              <span className="font-[var(--font-display)] text-[16px] font-bold text-[var(--text-primary)]">
                {gap.topic}
              </span>
            </div>

            {/* .gnn-item__metrics */}
            <div className="flex gap-6 mb-3.5 flex-wrap">
              <div className="flex flex-col gap-[3px]">
                {/* .gnn-metric__label */}
                <span className="text-[11px] text-[var(--text-muted)] uppercase tracking-[0.05em]">
                  Apparent Retention
                </span>
                {/* .gnn-metric__val */}
                <span className="font-[var(--font-mono)] text-[22px] font-bold text-[var(--accent-green)]">
                  {(gap.apparent_retention * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex flex-col gap-[3px]">
                <span className="text-[11px] text-[var(--text-muted)] uppercase tracking-[0.05em]">
                  True Confidence
                </span>
                <span className="font-[var(--font-mono)] text-[22px] font-bold text-[var(--accent-red)]">
                  {(gap.true_confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* .gnn-item__prereqs */}
            {gap.weak_prerequisites?.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[12px] text-[var(--text-muted)]">Weak Prerequisites:</span>
                {gap.weak_prerequisites.map((p, j) => (
                  /* .gnn-prereq-chip */
                  <span key={j} className="text-[12px] px-2.5 py-[3px]
                                           bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)]
                                           rounded-full text-[var(--accent-red)]">
                    {p}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── EmptyTab ── .tab-empty */
function EmptyTab({ message }) {
  return (
    <div className="p-10 text-center text-[14px] text-[var(--text-muted)]">{message}</div>
  )
}

/* ── LoadingSkeleton ── */
function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-5">
      <div className="skeleton h-20 rounded-[12px]" />
      <div className="skeleton h-[60px] rounded-[12px]" />
      <div className="flex flex-col gap-3">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="skeleton h-20 rounded-[10px]" />
        ))}
      </div>
    </div>
  )
}