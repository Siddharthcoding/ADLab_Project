import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import api from '../utils/api'
import toast from 'react-hot-toast'
import {
  ExternalLink, Brain, AlertTriangle, Clock,
  TrendingUp, ChevronLeft, Zap, Network, Tag, CheckCircle2,
  Play, Trophy, Calendar as CalendarIcon, Filter, X,
  Download, Share2, BarChart3, Check
} from 'lucide-react'

export default function RoadmapPage() {
  const { id } = useParams()
  const [roadmap, setRoadmap] = useState(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('problems')
  const [completedProblems, setCompletedProblems] = useState(new Set())

  useEffect(() => {
    loadRoadmap()
  }, [id])

  const loadRoadmap = async () => {
    try {
      const res = await api.get(`/roadmap/${id}`)
      setRoadmap(res.data)
      setCompletedProblems(new Set(res.data.completed_problems || []))
    } catch (err) {
      toast.error('Failed to load roadmap')
    } finally {
      setLoading(false)
    }
  }

  const toggleProblemCompletion = async (problemIndex) => {
    const newCompleted = new Set(completedProblems)
    const isCompleted = newCompleted.has(problemIndex)
    
    if (isCompleted) {
      newCompleted.delete(problemIndex)
    } else {
      newCompleted.add(problemIndex)
    }
    
    setCompletedProblems(newCompleted)
    
    try {
      await api.post(`/roadmap/${id}/toggle-problem`, {
        problem_index: problemIndex,
        completed: !isCompleted
      })
      toast.success(isCompleted ? 'Problem unmarked' : 'Problem completed! 🎉')
    } catch (err) {
      // Revert on error
      setCompletedProblems(completedProblems)
      toast.error('Failed to update progress')
    }
  }

  const exportRoadmap = () => {
    const data = {
      roadmap_id: roadmap.id,
      created: roadmap.created_at,
      level: roadmap.user_level,
      problems: roadmap.problems,
      weak_topics: roadmap.weak_topics,
      progress: `${completedProblems.size}/${roadmap.problems.length}`
    }
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `roadmap-${id}.json`
    a.click()
    toast.success('Roadmap exported!')
  }

  const shareRoadmap = () => {
    const url = window.location.href
    navigator.clipboard.writeText(url)
    toast.success('Link copied to clipboard!')
  }

  if (loading) return <LoadingSkeleton />
  if (!roadmap) return <div className="p-10 text-center text-[var(--text-muted)]">Roadmap not found.</div>

  const progress = roadmap.problems.length > 0 
    ? (completedProblems.size / roadmap.problems.length) * 100 
    : 0

  const tabs = [
    { id: 'problems', label: `Problems (${roadmap.problems?.length || 0})`, icon: Zap },
    { id: 'session', label: 'Session Plan', icon: Play, show: roadmap.session_plan?.length > 0 },
    { id: 'calendar', label: '7-Day Calendar', icon: CalendarIcon, show: roadmap.daily_calendar?.length > 0 },
    { id: 'retention', label: 'Retention', icon: TrendingUp, show: !!roadmap.retention_data },
    { id: 'gnn', label: 'GNN Gaps', icon: Network, show: roadmap.gnn_data?.hidden_gaps?.length > 0 },
    { id: 'stats', label: 'Statistics', icon: BarChart3 },
  ].filter(t => t.show !== false)

  return (
    <>
      <style>{`
        @keyframes slideUpFade { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes tabUnderline { from{width:0;opacity:0} to{width:80%;opacity:1} }
        @keyframes shimmer { 0%{background-position:200% center} 100%{background-position:-200% center} }
        @keyframes progressFill { from{width:0} }
        @keyframes checkPop { 0%{transform:scale(0)} 50%{transform:scale(1.2)} 100%{transform:scale(1)} }
        .rm-enter { animation: slideUpFade 0.45s cubic-bezier(0.16,1,0.3,1) both; }
        .r1{animation-delay:0.04s} .r2{animation-delay:0.1s} .r3{animation-delay:0.16s}
        .r4{animation-delay:0.22s} .r5{animation-delay:0.28s}
        .problem-card-wrap { 
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1); 
          border-left: 3px solid transparent; 
        }
        .problem-card-wrap:hover { 
          transform: translateX(4px) translateY(-2px); 
          border-left-color: var(--accent-green);
          box-shadow: 0 8px 32px rgba(0,0,0,0.4), -3px 0 12px rgba(0,245,160,0.1);
        }
        .problem-card-completed {
          opacity: 0.6;
          border-left-color: var(--accent-green) !important;
          background: linear-gradient(90deg, rgba(0,245,160,0.05), transparent);
        }
        .solve-btn {
          position: relative;
          overflow: hidden;
          transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        }
        .solve-btn::before {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
          background-size: 200% 100%;
          transform: translateX(-100%);
        }
        .solve-btn:hover::before {
          animation: shimmer 1s ease-in-out;
        }
        .solve-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 0 24px rgba(0,245,160,0.4);
        }
        .checkbox-wrapper {
          transition: all 0.2s ease;
        }
        .checkbox-wrapper:hover {
          transform: scale(1.1);
        }
        .checkbox-checked {
          animation: checkPop 0.3s cubic-bezier(0.16,1,0.3,1);
        }
        .progress-bar {
          animation: progressFill 1s cubic-bezier(0.16,1,0.3,1);
        }
        .tag-pill {
          transition: all 0.2s ease;
        }
        .tag-pill:hover {
          transform: translateY(-2px);
          background: var(--accent-blue-dim);
          border-color: var(--accent-blue);
          color: var(--accent-blue);
        }
        .session-item-wrap { transition: all 0.2s ease; border-left: 3px solid transparent; }
        .session-item-wrap:hover { border-left-color: var(--accent-green); background: var(--bg-elevated); }
        .tab-btn { transition: all 0.2s cubic-bezier(0.4,0,0.2,1); }
        .tab-btn:hover:not(.tab-active) { background: rgba(255,255,255,0.04); color: var(--text-secondary); }
        .tab-active { background: var(--bg-elevated) !important; color: var(--text-primary) !important; }
        .meta-badge { transition: all 0.2s ease; }
        .meta-badge:hover { border-color: rgba(255,255,255,0.15); background: var(--bg-elevated); transform: translateY(-1px); }
        .retention-bar-fill { transition: width 1.2s cubic-bezier(0.4,0,0.2,1); }
        .chip-hover { transition: all 0.2s ease; }
        .chip-hover:hover { transform: translateY(-1px); }
        .filter-btn { transition: all 0.2s ease; }
        .filter-btn:hover:not(.filter-active) { border-color: rgba(255,255,255,0.15); transform: translateY(-1px); }
        .filter-active { background: var(--accent-green-dim) !important; border-color: var(--border-accent) !important; color: var(--accent-green) !important; }
        .metric-badge {
          transition: all 0.2s ease;
        }
        .metric-badge:hover {
          transform: scale(1.05);
          background: var(--bg-elevated);
        }
        .action-btn {
          transition: all 0.2s ease;
        }
        .action-btn:hover {
          transform: translateY(-2px);
        }
      `}</style>

      <div className="flex flex-col gap-5">

        {/* Header */}
        <div className="rm-enter r1 flex flex-col gap-3">
          <Link to="/history"
            className="inline-flex items-center gap-1.5 w-fit text-[13px] text-[var(--text-muted)]
                       hover:text-[var(--text-secondary)] transition-all duration-200 group">
            <ChevronLeft size={16} className="group-hover:-translate-x-1 transition-transform duration-200" />
            Back to History
          </Link>
          
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3">
              <h1 className="font-[var(--font-display)] text-[26px] font-extrabold text-[var(--text-primary)]">
                Roadmap #{id}
              </h1>
              <span className="text-[12px] font-bold px-3 py-1.5 rounded-full
                               bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                               border border-[rgba(0,245,160,0.3)]
                               shadow-[0_0_12px_rgba(0,245,160,0.15)]">
                {roadmap.user_level || 'Beginner'}
              </span>
            </div>
            
            <div className="flex gap-2">
              <button onClick={exportRoadmap}
                className="action-btn flex items-center gap-2 px-4 py-2 rounded-lg
                           bg-[var(--bg-card)] border border-[var(--border-subtle)]
                           text-[13px] font-semibold text-[var(--text-secondary)]
                           hover:border-[var(--accent-blue)] hover:text-[var(--accent-blue)]">
                <Download size={14} />
                Export
              </button>
              <button onClick={shareRoadmap}
                className="action-btn flex items-center gap-2 px-4 py-2 rounded-lg
                           bg-[var(--bg-card)] border border-[var(--border-subtle)]
                           text-[13px] font-semibold text-[var(--text-secondary)]
                           hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]">
                <Share2 size={14} />
                Share
              </button>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-[13px]">
              <span className="text-[var(--text-secondary)] font-medium">
                Progress: {completedProblems.size} / {roadmap.problems.length} problems
              </span>
              <span className="text-[var(--accent-green)] font-bold font-[var(--font-mono)]">
                {progress.toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-[var(--bg-elevated)] rounded-full overflow-hidden border border-[var(--border-subtle)]">
              <div className="progress-bar h-full bg-gradient-to-r from-[var(--accent-green)] to-[#00d48c]
                              rounded-full transition-all duration-1000 ease-out
                              shadow-[0_0_12px_rgba(0,245,160,0.5)]"
                   style={{ width: `${progress}%` }} />
            </div>
          </div>

          <div className="flex gap-2 flex-wrap">
            <MetaBadge icon={<Zap size={14} />} label={`${roadmap.problems?.length || 0} problems`} accent="green" />
            <MetaBadge icon={<CheckCircle2 size={14} />} label={`${completedProblems.size} completed`} accent="green" />
            <MetaBadge icon={<TrendingUp size={14} />} label={`Penalty: ${roadmap.contest_penalty?.toFixed(3) || 'N/A'}`} accent="orange" />
            <MetaBadge icon={<Clock size={14} />} label={new Date(roadmap.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} accent="blue" />
          </div>
        </div>

        {/* Weak topics, ML insights, Tabs - same as before */}
        {roadmap.weak_topics?.length > 0 && (
          <div className="rm-enter r2 bg-gradient-to-br from-[rgba(249,115,22,0.08)] to-[rgba(249,115,22,0.02)]
                          border border-[rgba(249,115,22,0.25)]
                          rounded-[var(--radius-xl)] px-6 py-5 relative overflow-hidden
                          shadow-[0_4px_24px_rgba(249,115,22,0.1)]">
            <div className="absolute top-0 left-0 right-0 h-[2px]
                            bg-gradient-to-r from-transparent via-[var(--accent-orange)] to-transparent opacity-60" />
            <div className="flex items-center gap-2.5 text-[14px] font-bold text-[var(--text-primary)] mb-4">
              <div className="w-8 h-8 rounded-lg bg-[var(--accent-orange-dim)] border border-[rgba(249,115,22,0.3)]
                              flex items-center justify-center">
                <AlertTriangle size={16} className="text-[var(--accent-orange)]" />
              </div>
              Weak Topics Detected
            </div>
            <div className="flex flex-wrap gap-2.5">
              {roadmap.weak_topics.map((t, i) => (
                <span key={i}
                  className="chip-hover flex items-center gap-2 px-4 py-2
                             bg-[var(--accent-orange-dim)] border border-[rgba(249,115,22,0.3)]
                             rounded-lg text-[13px] font-medium text-[var(--accent-orange)] cursor-default
                             shadow-[0_2px_8px_rgba(249,115,22,0.1)]"
                  style={{ animationDelay: `${0.2 + i * 0.04}s` }}>
                  <span className="w-5 h-5 rounded-full bg-[rgba(249,115,22,0.2)] flex items-center justify-center
                                   font-[var(--font-mono)] text-[10px] font-bold">{i + 1}</span>
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}

        {roadmap.ml_insights && (
          <div className="rm-enter r3 flex gap-3.5 px-6 py-5
                          bg-gradient-to-br from-[rgba(168,85,247,0.08)] to-[rgba(168,85,247,0.02)]
                          border border-[rgba(168,85,247,0.25)]
                          rounded-[var(--radius-xl)] text-[14px] text-[var(--text-secondary)] leading-[1.7]
                          relative overflow-hidden shadow-[0_4px_24px_rgba(168,85,247,0.1)]">
            <div className="absolute top-0 left-0 right-0 h-[2px]
                            bg-gradient-to-r from-transparent via-[var(--accent-purple)] to-transparent opacity-60" />
            <div className="w-8 h-8 rounded-lg bg-[var(--accent-purple-dim)] border border-[rgba(168,85,247,0.3)]
                            flex items-center justify-center shrink-0">
              <Brain size={16} className="text-[var(--accent-purple)]" />
            </div>
            <p className="flex-1">{roadmap.ml_insights}</p>
          </div>
        )}

        {/* Tabs */}
        <div className="rm-enter r4 flex gap-2 flex-wrap bg-[var(--bg-card)] border border-[var(--border-subtle)]
                        rounded-[var(--radius-xl)] p-2 relative shadow-[0_2px_12px_rgba(0,0,0,0.2)]">
          {tabs.map(t => {
            const Icon = t.icon
            return (
              <button key={t.id} onClick={() => setActiveTab(t.id)}
                className={[
                  'tab-btn px-5 py-2.5 rounded-[var(--radius-lg)] text-[13px] font-semibold whitespace-nowrap',
                  'flex items-center gap-2 relative',
                  activeTab === t.id ? 'tab-active' : 'text-[var(--text-muted)]',
                ].join(' ')}>
                <Icon size={15} className={activeTab === t.id ? 'text-[var(--accent-green)]' : ''} />
                {t.label}
                {activeTab === t.id && (
                  <span className="absolute inset-x-[10%] bottom-[2px] h-[2px] rounded-full
                                   bg-[var(--accent-green)] opacity-70
                                   [animation:tabUnderline_0.3s_ease_both]" />
                )}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="rm-enter r5">
          {activeTab === 'problems' && <ProblemsTab problems={roadmap.problems} completedProblems={completedProblems} onToggle={toggleProblemCompletion} />}
          {activeTab === 'session' && <SessionTab session={roadmap.session_plan} />}
          {activeTab === 'calendar' && <CalendarTab calendar={roadmap.daily_calendar} />}
          {activeTab === 'retention' && <RetentionTab data={roadmap.retention_data} />}
          {activeTab === 'gnn' && <GNNTab data={roadmap.gnn_data} />}
          {activeTab === 'stats' && <StatisticsTab roadmap={roadmap} completedProblems={completedProblems} />}
        </div>
      </div>
    </>
  )
}

// Helper components continued in next message due to length...

function MetaBadge({ icon, label, accent = 'blue' }) {
  const colors = {
    green: 'bg-[var(--accent-green-dim)] border-[rgba(0,245,160,0.3)] text-[var(--accent-green)]',
    blue: 'bg-[var(--accent-blue-dim)] border-[rgba(59,130,246,0.3)] text-[var(--accent-blue)]',
    orange: 'bg-[var(--accent-orange-dim)] border-[rgba(249,115,22,0.3)] text-[var(--accent-orange)]',
  }
  
  return (
    <div className={`meta-badge flex items-center gap-2 px-3 py-2
                    border rounded-lg text-[12px] font-semibold font-[var(--font-mono)]
                    cursor-default ${colors[accent]}`}>
      {icon}<span>{label}</span>
    </div>
  )
}

/* ══ PROBLEMS TAB ══ */
function ProblemsTab({ problems, completedProblems, onToggle }) {
  const [filter, setFilter] = useState('all')
  const [tagFilter, setTagFilter] = useState(null)
  const [showCompleted, setShowCompleted] = useState(true)
  
  let filtered = filter === 'all' ? problems : problems.filter(p => p.source === filter)
  if (tagFilter) {
    filtered = filtered.filter(p => p.tags?.includes(tagFilter))
  }
  if (!showCompleted) {
    filtered = filtered.filter((_, i) => !completedProblems.has(i))
  }

  return (
    <div className="flex flex-col gap-5">
      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <div className="flex gap-2">
          {['all', 'LeetCode', 'Codeforces'].map(s => (
            <button key={s} onClick={() => setFilter(s)}
              className={`filter-btn px-4 py-2 rounded-lg text-[13px] font-semibold border ${
                filter === s 
                  ? 'filter-active' 
                  : 'bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)]'
              }`}>
              {s}
              {s !== 'all' && (
                <span className="ml-2 text-[11px] font-[var(--font-mono)] opacity-70">
                  ({problems?.filter(p => p.source === s).length || 0})
                </span>
              )}
            </button>
          ))}
        </div>

        <button onClick={() => setShowCompleted(!showCompleted)}
          className={`filter-btn px-4 py-2 rounded-lg text-[13px] font-semibold border ${
            showCompleted
              ? 'bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)]'
              : 'bg-[var(--accent-orange-dim)] border-[var(--accent-orange)] text-[var(--accent-orange)]'
          }`}>
          {showCompleted ? 'Hide Completed' : 'Show All'}
        </button>
        
        {tagFilter && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--accent-blue-dim)]
                          border border-[var(--accent-blue)] rounded-lg text-[12px] text-[var(--accent-blue)]">
            <Filter size={12} />
            <span>Tag: {tagFilter}</span>
            <button onClick={() => setTagFilter(null)}
              className="ml-1 hover:bg-[rgba(59,130,246,0.2)] rounded p-0.5 transition-colors">
              <X size={12} />
            </button>
          </div>
        )}
        
        <span className="ml-auto text-[13px] text-[var(--text-muted)] font-[var(--font-mono)]">
          Showing {filtered.length} of {problems?.length || 0} problems
        </span>
      </div>

      {/* Problems */}
      <div className="flex flex-col gap-3">
        {filtered.map((p, i) => {
          const originalIndex = problems.indexOf(p)
          return (
            <ProblemCard 
              key={originalIndex} 
              problem={p} 
              rank={originalIndex + 1}
              onTagClick={setTagFilter}
              isCompleted={completedProblems.has(originalIndex)}
              onToggleComplete={() => onToggle(originalIndex)}
            />
          )
        })}
      </div>
    </div>
  )
}

const diffBadgeCls = {
  green: 'bg-[var(--accent-green-dim)] text-[var(--accent-green)] border-[rgba(0,245,160,0.3)]',
  blue: 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)] border-[rgba(59,130,246,0.3)]',
  orange: 'bg-[var(--accent-orange-dim)] text-[var(--accent-orange)] border-[rgba(249,115,22,0.3)]',
  red: 'bg-[rgba(239,68,68,0.15)] text-[var(--accent-red)] border-[rgba(239,68,68,0.3)]',
}

function ProblemCard({ problem: p, rank, onTagClick, isCompleted, onToggleComplete }) {
  const isLC = p.source === 'LeetCode'
  const diffKey = p.difficulty <= 800 ? 'green' : p.difficulty <= 1400 ? 'blue' : p.difficulty <= 2000 ? 'orange' : 'red'

  return (
    <div className={`problem-card-wrap flex gap-4 bg-[var(--bg-card)] border border-[var(--border-subtle)]
                    rounded-[var(--radius-xl)] p-5 group relative overflow-hidden
                    ${isCompleted ? 'problem-card-completed' : ''}`}>
      
      {/* Completion Checkbox */}
      <div className="checkbox-wrapper shrink-0">
        <button
          onClick={onToggleComplete}
          className={`w-10 h-10 rounded-lg border-2 flex items-center justify-center
                     transition-all duration-300 ${
            isCompleted 
              ? 'bg-[var(--accent-green)] border-[var(--accent-green)] checkbox-checked' 
              : 'bg-[var(--bg-elevated)] border-[var(--border-subtle)] hover:border-[var(--accent-green)]'
          }`}>
          {isCompleted && <Check size={20} className="text-[#080c14]" />}
        </button>
      </div>

      {/* Rank badge */}
      <div className="w-10 h-10 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)]
                      flex items-center justify-center shrink-0
                      font-[var(--font-mono)] text-[14px] font-bold text-[var(--text-muted)]
                      group-hover:border-[var(--accent-green)] group-hover:text-[var(--accent-green)]
                      transition-all duration-300">
        {rank}
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col gap-3">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2.5 mb-2">
              <span className={[
                'text-[11px] font-bold px-2.5 py-1 rounded-md font-[var(--font-mono)] shrink-0',
                isLC ? 'bg-[rgba(255,161,22,0.2)] text-[#FFA116] border border-[rgba(255,161,22,0.3)]' 
                     : 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)] border border-[rgba(59,130,246,0.3)]',
              ].join(' ')}>{isLC ? 'LeetCode' : 'Codeforces'}</span>
              <span className={`text-[12px] font-bold px-3 py-1 rounded-md font-[var(--font-mono)] border ${diffBadgeCls[diffKey]}`}>
                {p.difficulty}
              </span>
              {isCompleted && (
                <span className="text-[11px] font-bold px-2.5 py-1 rounded-md
                                 bg-[var(--accent-green-dim)] text-[var(--accent-green)]
                                 border border-[rgba(0,245,160,0.3)]
                                 flex items-center gap-1">
                  <CheckCircle2 size={12} /> Completed
                </span>
              )}
            </div>
            <h3 className={`text-[16px] font-bold mb-2 ${
              isCompleted ? 'text-[var(--text-muted)] line-through' : 'text-[var(--text-primary)]'
            }`}>{p.name}</h3>
            
            {/* ML Explanation */}
            {p.ml_explanation && (
              <div className="flex items-start gap-2 mb-3 px-3 py-2 bg-[var(--accent-purple-dim)]
                              border border-[rgba(168,85,247,0.2)] rounded-lg">
                <Brain size={14} className="shrink-0 text-[var(--accent-purple)] mt-0.5" />
                <p className="text-[13px] text-[var(--text-secondary)] leading-[1.5] italic">
                  {p.ml_explanation}
                </p>
              </div>
            )}

            {/* Tags */}
            <div className="flex flex-wrap gap-2 mb-3">
              {p.tags?.slice(0, 6).map((tag, i) => (
                <button
                  key={i}
                  onClick={() => onTagClick(tag)}
                  className="tag-pill text-[12px] px-3 py-1.5 bg-[var(--bg-elevated)]
                             border border-[var(--border-subtle)] rounded-lg
                             text-[var(--text-secondary)] font-medium cursor-pointer">
                  <Tag size={10} className="inline mr-1.5" />
                  {tag}
                </button>
              ))}
              {p.tags?.length > 6 && (
                <span className="text-[12px] px-3 py-1.5 text-[var(--text-muted)]">
                  +{p.tags.length - 6} more
                </span>
              )}
            </div>

            {/* Metrics */}
            <div className="flex gap-3 flex-wrap">
              <MetricPill label="ML Priority" value={`${(p.ml_priority * 100).toFixed(0)}%`} color="green" />
              <MetricPill label="Forgotten" value={`${(p.forgetting_urgency * 100).toFixed(0)}%`} color="orange" />
              <MetricPill label="Est. Time" value={`${p.est_minutes}m`} color="blue" />
            </div>
          </div>

          {/* Solve Button */}
          {p.link && (
            <a
              href={p.link}
              target="_blank"
              rel="noopener noreferrer"
              className="solve-btn flex items-center gap-2 px-6 py-3 whitespace-nowrap
                         bg-[var(--accent-green)] text-[#080c14]
                         rounded-[var(--radius-lg)] text-[14px] font-bold
                         shadow-[0_4px_16px_rgba(0,245,160,0.2)]
                         shrink-0">
              <Play size={16} className="relative z-[1]" />
              <span className="relative z-[1]">Solve</span>
              <ExternalLink size={14} className="relative z-[1]" />
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

function MetricPill({ label, value, color = 'blue' }) {
  const colors = {
    green: 'border-[rgba(0,245,160,0.3)] bg-[var(--accent-green-dim)]',
    blue: 'border-[rgba(59,130,246,0.3)] bg-[var(--accent-blue-dim)]',
    orange: 'border-[rgba(249,115,22,0.3)] bg-[var(--accent-orange-dim)]',
  }
  
  const textColors = {
    green: 'text-[var(--accent-green)]',
    blue: 'text-[var(--accent-blue)]',
    orange: 'text-[var(--accent-orange)]',
  }

  return (
    <div className={`metric-badge flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[12px] ${colors[color]}`}>
      <span className="text-[var(--text-muted)] font-medium">{label}</span>
      <span className={`font-bold font-[var(--font-mono)] ${textColors[color]}`}>{value}</span>
    </div>
  )
}

/* ══ STATISTICS TAB ══ */
function StatisticsTab({ roadmap, completedProblems }) {
  const problems = roadmap.problems || []
  
  // Calculate statistics
  const totalProblems = problems.length
  const completedCount = completedProblems.size
  const remainingCount = totalProblems - completedCount
  const progressPercent = totalProblems > 0 ? (completedCount / totalProblems) * 100 : 0
  
  // By difficulty
  const byDifficulty = {
    easy: problems.filter(p => p.difficulty <= 800).length,
    medium: problems.filter(p => p.difficulty > 800 && p.difficulty <= 1400).length,
    hard: problems.filter(p => p.difficulty > 1400 && p.difficulty <= 2000).length,
    expert: problems.filter(p => p.difficulty > 2000).length,
  }
  
  const completedByDiff = {
    easy: problems.filter((p, i) => p.difficulty <= 800 && completedProblems.has(i)).length,
    medium: problems.filter((p, i) => p.difficulty > 800 && p.difficulty <= 1400 && completedProblems.has(i)).length,
    hard: problems.filter((p, i) => p.difficulty > 1400 && p.difficulty <= 2000 && completedProblems.has(i)).length,
    expert: problems.filter((p, i) => p.difficulty > 2000 && completedProblems.has(i)).length,
  }
  
  // By platform
  const byPlatform = {
    leetcode: problems.filter(p => p.source === 'LeetCode').length,
    codeforces: problems.filter(p => p.source === 'Codeforces').length,
  }
  
  const completedByPlatform = {
    leetcode: problems.filter((p, i) => p.source === 'LeetCode' && completedProblems.has(i)).length,
    codeforces: problems.filter((p, i) => p.source === 'Codeforces' && completedProblems.has(i)).length,
  }
  
  // Top tags
  const tagCounts = {}
  problems.forEach(p => {
    p.tags?.forEach(tag => {
      tagCounts[tag] = (tagCounts[tag] || 0) + 1
    })
  })
  const topTags = Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
  
  // Estimated time
  const totalMinutes = problems.reduce((sum, p) => sum + (p.est_minutes || 0), 0)
  const completedMinutes = problems
    .filter((_, i) => completedProblems.has(i))
    .reduce((sum, p) => sum + (p.est_minutes || 0), 0)
  const remainingMinutes = totalMinutes - completedMinutes
  
  const formatTime = (mins) => {
    const hours = Math.floor(mins / 60)
    const minutes = mins % 60
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Problems"
          value={totalProblems}
          icon={<Zap size={20} />}
          color="blue"
        />
        <StatCard
          title="Completed"
          value={completedCount}
          subtitle={`${progressPercent.toFixed(1)}%`}
          icon={<CheckCircle2 size={20} />}
          color="green"
        />
        <StatCard
          title="Remaining"
          value={remainingCount}
          icon={<AlertTriangle size={20} />}
          color="orange"
        />
        <StatCard
          title="Est. Time Left"
          value={formatTime(remainingMinutes)}
          subtitle={`${formatTime(totalMinutes)} total`}
          icon={<Clock size={20} />}
          color="purple"
        />
      </div>

      {/* Difficulty Breakdown */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                      rounded-[var(--radius-xl)] p-6">
        <h3 className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)] mb-5
                       flex items-center gap-2">
          <TrendingUp size={20} className="text-[var(--accent-blue)]" />
          Difficulty Breakdown
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <ProgressBar
            label="Easy (≤800)"
            completed={completedByDiff.easy}
            total={byDifficulty.easy}
            color="green"
          />
          <ProgressBar
            label="Medium (801-1400)"
            completed={completedByDiff.medium}
            total={byDifficulty.medium}
            color="blue"
          />
          <ProgressBar
            label="Hard (1401-2000)"
            completed={completedByDiff.hard}
            total={byDifficulty.hard}
            color="orange"
          />
          <ProgressBar
            label="Expert (2000+)"
            completed={completedByDiff.expert}
            total={byDifficulty.expert}
            color="red"
          />
        </div>
      </div>

      {/* Platform Breakdown */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                      rounded-[var(--radius-xl)] p-6">
        <h3 className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)] mb-5
                       flex items-center gap-2">
          <Network size={20} className="text-[var(--accent-purple)]" />
          Platform Distribution
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <ProgressBar
            label="LeetCode"
            completed={completedByPlatform.leetcode}
            total={byPlatform.leetcode}
            color="orange"
            showIcon="LC"
          />
          <ProgressBar
            label="Codeforces"
            completed={completedByPlatform.codeforces}
            total={byPlatform.codeforces}
            color="blue"
            showIcon="CF"
          />
        </div>
      </div>

      {/* Top Tags */}
      <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                      rounded-[var(--radius-xl)] p-6">
        <h3 className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)] mb-5
                       flex items-center gap-2">
          <Tag size={20} className="text-[var(--accent-green)]" />
          Most Common Topics
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {topTags.map(([tag, count], i) => (
            <div key={tag}
                 className="flex flex-col gap-1 p-3 bg-[var(--bg-elevated)]
                            border border-[var(--border-subtle)] rounded-lg
                            hover:border-[var(--accent-blue)] transition-all duration-200"
                 style={{ animationDelay: `${i * 0.05}s` }}>
              <span className="text-[13px] font-semibold text-[var(--text-primary)] truncate">
                {tag}
              </span>
              <span className="text-[20px] font-bold text-[var(--accent-blue)] font-[var(--font-mono)]">
                {count}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Performance Insights */}
      {roadmap.contest_penalty !== null && (
        <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                        rounded-[var(--radius-xl)] p-6">
          <h3 className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)] mb-4
                         flex items-center gap-2">
            <Trophy size={20} className="text-[var(--accent-orange)]" />
            Performance Metrics
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="flex flex-col gap-2 p-4 bg-[var(--bg-elevated)]
                            border border-[var(--border-subtle)] rounded-lg">
              <span className="text-[13px] text-[var(--text-muted)] uppercase tracking-wider">
                Contest Penalty
              </span>
              <span className="text-[28px] font-bold text-[var(--accent-orange)] font-[var(--font-mono)]">
                {roadmap.contest_penalty.toFixed(3)}
              </span>
            </div>
            <div className="flex flex-col gap-2 p-4 bg-[var(--bg-elevated)]
                            border border-[var(--border-subtle)] rounded-lg">
              <span className="text-[13px] text-[var(--text-muted)] uppercase tracking-wider">
                Skill Level
              </span>
              <span className="text-[28px] font-bold text-[var(--accent-green)] font-[var(--font-display)]">
                {roadmap.user_level}
              </span>
            </div>
            <div className="flex flex-col gap-2 p-4 bg-[var(--bg-elevated)]
                            border border-[var(--border-subtle)] rounded-lg">
              <span className="text-[13px] text-[var(--text-muted)] uppercase tracking-wider">
                Weak Topics
              </span>
              <span className="text-[28px] font-bold text-[var(--accent-red)] font-[var(--font-mono)]">
                {roadmap.weak_topics?.length || 0}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ title, value, subtitle, icon, color }) {
  const colors = {
    blue: 'bg-[var(--accent-blue-dim)] border-[rgba(59,130,246,0.3)] text-[var(--accent-blue)]',
    green: 'bg-[var(--accent-green-dim)] border-[rgba(0,245,160,0.3)] text-[var(--accent-green)]',
    orange: 'bg-[var(--accent-orange-dim)] border-[rgba(249,115,22,0.3)] text-[var(--accent-orange)]',
    purple: 'bg-[var(--accent-purple-dim)] border-[rgba(168,85,247,0.3)] text-[var(--accent-purple)]',
  }

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                    rounded-[var(--radius-xl)] p-5
                    hover:border-[var(--border-medium)] hover:-translate-y-1
                    transition-all duration-300 shadow-[0_2px_12px_rgba(0,0,0,0.2)]">
      <div className={`w-12 h-12 rounded-lg border flex items-center justify-center mb-3 ${colors[color]}`}>
        {icon}
      </div>
      <div className="text-[13px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
        {title}
      </div>
      <div className="text-[32px] font-bold text-[var(--text-primary)] font-[var(--font-mono)] leading-none mb-1">
        {value}
      </div>
      {subtitle && (
        <div className="text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
          {subtitle}
        </div>
      )}
    </div>
  )
}

function ProgressBar({ label, completed, total, color, showIcon }) {
  const percent = total > 0 ? (completed / total) * 100 : 0
  
  const colors = {
    green: 'bg-[var(--accent-green)]',
    blue: 'bg-[var(--accent-blue)]',
    orange: 'bg-[#FFA116]',
    red: 'bg-[var(--accent-red)]',
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[14px] font-semibold text-[var(--text-primary)] flex items-center gap-2">
          {showIcon && (
            <span className={`text-[10px] font-bold px-2 py-1 rounded-md font-[var(--font-mono)] ${
              showIcon === 'LC' 
                ? 'bg-[rgba(255,161,22,0.2)] text-[#FFA116] border border-[rgba(255,161,22,0.3)]'
                : 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)] border border-[rgba(59,130,246,0.3)]'
            }`}>
              {showIcon}
            </span>
          )}
          {label}
        </span>
        <span className="text-[13px] text-[var(--text-muted)] font-[var(--font-mono)]">
          {completed}/{total}
        </span>
      </div>
      <div className="h-2 bg-[var(--bg-elevated)] rounded-full overflow-hidden border border-[var(--border-subtle)]">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${colors[color]}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="text-[12px] text-[var(--text-muted)] font-[var(--font-mono)]">
        {percent.toFixed(1)}% complete
      </span>
    </div>
  )
}

/* ══ SESSION TAB ══ */
function SessionTab({ session }) {
  if (!session?.length) return <EmptyTab message="No session plan available." />
  return (
    <div>
      <div className="flex items-center gap-3 mb-5 px-5 py-4 bg-[var(--accent-blue-dim)]
                      border border-[rgba(59,130,246,0.3)] rounded-[var(--radius-lg)]">
        <Play size={18} className="text-[var(--accent-blue)]" />
        <p className="text-[14px] text-[var(--text-secondary)] leading-[1.6]">
          Optimal problem ordering for your practice session based on SM-2 spaced repetition algorithm
        </p>
      </div>
      <div className="flex flex-col gap-3">
        {session.map((p, i) => (
          <div key={i} className="session-item-wrap flex items-center gap-4 px-5 py-4
                                   bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                   rounded-[var(--radius-lg)] group">
            <div className="w-10 h-10 rounded-lg shrink-0 flex items-center justify-center
                            bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                            text-[14px] font-[var(--font-mono)] font-bold text-[var(--accent-green)]
                            transition-transform duration-200 group-hover:scale-110
                            shadow-[0_2px_8px_rgba(0,245,160,0.2)]">
              {i + 1}
            </div>
            <div className="flex-1">
              <div className="text-[15px] font-bold text-[var(--text-primary)] mb-2">{p.name}</div>
              <div className="flex items-center gap-3 flex-wrap">
                <span className={[
                  'text-[11px] font-bold px-2.5 py-1 rounded-md font-[var(--font-mono)]',
                  p.source === 'LeetCode' 
                    ? 'bg-[rgba(255,161,22,0.2)] text-[#FFA116] border border-[rgba(255,161,22,0.3)]' 
                    : 'bg-[var(--accent-blue-dim)] text-[var(--accent-blue)] border border-[rgba(59,130,246,0.3)]',
                ].join(' ')}>{p.source}</span>
                <span className="text-[13px] text-[var(--text-secondary)]">{p.session_reason}</span>
              </div>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg
                            bg-[var(--bg-elevated)] border border-[var(--border-subtle)]
                            font-[var(--font-mono)] text-[13px] text-[var(--text-muted)] shrink-0">
              <Clock size={14} /> {p.est_minutes}m
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
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {calendar.map((day, i) => (
        <div key={i}
          className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                     rounded-[var(--radius-xl)] p-6
                     transition-all duration-250 hover:border-[var(--border-medium)]
                     hover:-translate-y-2 hover:shadow-[0_12px_36px_rgba(0,0,0,0.4)]
                     relative overflow-hidden group"
          style={{ animationDelay: `${i * 0.06}s` }}>
          <div className="absolute top-0 left-0 right-0 h-[3px]
                          bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)]
                          opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          <div className="flex items-center gap-3 mb-3 flex-wrap">
            <span className="w-10 h-10 rounded-lg bg-[var(--accent-green-dim)] border border-[var(--border-accent)]
                             flex items-center justify-center
                             font-[var(--font-mono)] text-[14px] font-bold text-[var(--accent-green)]">
              {day.day}
            </span>
            <span className="text-[15px] font-bold text-[var(--text-primary)] flex-1">{day.label}</span>
            {day.roi && (
              <span className="px-3 py-1.5 rounded-lg bg-[var(--accent-green-dim)]
                               border border-[rgba(0,245,160,0.3)]
                               font-[var(--font-mono)] text-[12px] font-bold text-[var(--accent-green)]">
                +{day.roi.toFixed(1)} pts/hr
              </span>
            )}
          </div>
          <p className="text-[14px] text-[var(--text-secondary)] mb-4 leading-[1.6]">{day.goal}</p>
          {day.focus_topics?.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              <Tag size={12} className="text-[var(--text-muted)] shrink-0" />
              {day.focus_topics.slice(0, 4).map((t, j) => (
                <span key={j} className="text-[12px] px-3 py-1.5 bg-[var(--accent-blue-dim)]
                                         border border-[rgba(59,130,246,0.3)]
                                         rounded-lg text-[var(--accent-blue)] font-medium">
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
  const [revealed, setRevealed] = useState(false)
  useEffect(() => { setTimeout(() => setRevealed(true), 100) }, [])
  const atRisk = data.at_risk || []

  return (
    <div>
      <h3 className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)] mb-5
                     flex items-center gap-2">
        <TrendingUp size={20} className="text-[var(--accent-orange)]" />
        At-Risk Topics (Need Review)
      </h3>
      {atRisk.length === 0 ? (
        <div className="flex items-center gap-3 px-5 py-4 bg-[var(--accent-green-dim)]
                        border border-[rgba(0,245,160,0.3)] rounded-[var(--radius-lg)]
                        text-[15px] font-semibold text-[var(--accent-green)]">
          <CheckCircle2 size={18} /> No at-risk topics. Great job!
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {atRisk.map((item, i) => {
            const barColor = item.retention > 0.5 ? 'bg-[var(--accent-blue)]' :
                             item.retention > 0.2 ? 'bg-[var(--accent-orange)]' : 'bg-[var(--accent-red)]'
            const glowColor = item.retention > 0.5 ? 'rgba(59,130,246,0.5)' :
                              item.retention > 0.2 ? 'rgba(249,115,22,0.5)' : 'rgba(239,68,68,0.5)'
            return (
              <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                      rounded-[var(--radius-lg)] p-5
                                      transition-all duration-200 hover:border-[var(--border-medium)]
                                      hover:shadow-[0_4px_20px_rgba(0,0,0,0.3)]"
                   style={{ animationDelay: `${i * 0.08}s` }}>
                <div className="flex justify-between mb-3">
                  <span className="text-[16px] font-bold text-[var(--text-primary)]">{item.tag}</span>
                  <span className="text-[13px] text-[var(--text-muted)] font-[var(--font-mono)]">
                    {item.last_seen_days?.toFixed(0)} days ago
                  </span>
                </div>
                <div className="h-2 bg-[var(--bg-elevated)] rounded-full overflow-hidden mb-2
                                border border-[var(--border-subtle)]">
                  <div
                    className={`retention-bar-fill h-full rounded-full ${barColor}`}
                    style={{
                      width: revealed ? `${item.retention * 100}%` : '0%',
                      boxShadow: revealed ? `0 0 12px ${glowColor}` : 'none',
                      transitionDelay: `${i * 0.08}s`,
                    }}
                  />
                </div>
                <span className="text-[13px] font-bold text-[var(--text-muted)] font-[var(--font-mono)]">
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
      <div className="flex items-start gap-3 px-5 py-4 mb-6
                      bg-[var(--accent-purple-dim)] border border-[rgba(168,85,247,0.3)]
                      rounded-[var(--radius-lg)] text-[14px] text-[var(--text-secondary)] leading-[1.7]
                      relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-[2px]
                        bg-gradient-to-r from-transparent via-[var(--accent-purple)] to-transparent opacity-50" />
        <div className="w-8 h-8 rounded-lg bg-[rgba(168,85,247,0.2)] border border-[rgba(168,85,247,0.3)]
                        flex items-center justify-center shrink-0">
          <Network size={16} className="text-[var(--accent-purple)]" />
        </div>
        <p>Our Graph Neural Network analyzed your prerequisite relationships and found these hidden knowledge gaps:</p>
      </div>
      <div className="flex flex-col gap-4">
        {data.hidden_gaps.map((gap, i) => (
          <div key={i} className="bg-[var(--bg-card)] border border-[var(--border-subtle)]
                                   rounded-[var(--radius-xl)] p-6
                                   transition-all duration-200 hover:border-[var(--border-medium)]
                                   hover:-translate-y-1 hover:shadow-[0_8px_32px_rgba(0,0,0,0.4)]
                                   relative overflow-hidden group">
            <div className="absolute top-0 left-0 bottom-0 w-[4px]
                            bg-gradient-to-b from-[var(--accent-red)] to-transparent
                            opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-l-[var(--radius-xl)]" />
            <div className="mb-4">
              <span className="font-[var(--font-display)] text-[18px] font-bold text-[var(--text-primary)]">
                {gap.topic}
              </span>
            </div>
            <div className="flex gap-8 mb-4 flex-wrap">
              <div className="flex flex-col gap-1">
                <span className="text-[12px] text-[var(--text-muted)] uppercase tracking-[0.08em] font-semibold">
                  Apparent Retention
                </span>
                <span className="font-[var(--font-mono)] text-[28px] font-bold text-[var(--accent-green)]">
                  {(gap.apparent_retention * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[12px] text-[var(--text-muted)] uppercase tracking-[0.08em] font-semibold">
                  True Confidence
                </span>
                <span className="font-[var(--font-mono)] text-[28px] font-bold text-[var(--accent-red)]">
                  {(gap.true_confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            {gap.weak_prerequisites?.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[13px] text-[var(--text-muted)] font-semibold">Weak Prerequisites:</span>
                {gap.weak_prerequisites.map((p, j) => (
                  <span key={j} className="chip-hover text-[13px] px-3 py-1.5
                                           bg-[rgba(239,68,68,0.15)] border border-[rgba(239,68,68,0.3)]
                                           rounded-lg text-[var(--accent-red)] font-medium cursor-default">
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

function EmptyTab({ message }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center gap-4">
      <div className="w-16 h-16 rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)]
                      flex items-center justify-center text-[var(--text-muted)]">
        <Trophy size={28} />
      </div>
      <p className="text-[15px] text-[var(--text-muted)] font-medium">{message}</p>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-5">
      <div className="skeleton h-24 rounded-[16px]" />
      <div className="skeleton h-[80px] rounded-[16px]" />
      <div className="flex flex-col gap-4">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="skeleton h-32 rounded-[16px]" style={{ animationDelay: `${i * 0.08}s` }} />
        ))}
      </div>
    </div>
  )
}