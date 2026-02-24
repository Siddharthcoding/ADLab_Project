import React, { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import {
  LayoutDashboard, Zap, History, LogOut, Menu, X,
  Code2, ChevronRight, User
} from 'lucide-react'

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/generate',  label: 'Generate',  icon: Zap },
  { path: '/history',   label: 'History',   icon: History },
]

export default function Layout({ children }) {
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [collapsed,   setCollapsed]   = useState(false)
  const [mobileOpen,  setMobileOpen]  = useState(false)

  const handleLogout = () => { logout(); navigate('/') }

  return (
    <div className="flex min-h-screen relative">

      {/* ── Sidebar ── */}
      <aside
        className={[
          // base
          'fixed left-0 top-0 bottom-0 z-[100]',
          'flex flex-col min-h-screen',
          'bg-[var(--bg-secondary)] border-r border-[var(--border-subtle)]',
          'overflow-hidden transition-[width] duration-300',
          // desktop width
          collapsed ? 'w-[60px]' : 'w-[220px]',
          // mobile: hidden off-screen unless open
          mobileOpen
            ? 'translate-x-0'
            : '-translate-x-full md:translate-x-0',
          'transition-transform md:transition-[width] duration-300',
        ].join(' ')}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-5 border-b border-[var(--border-subtle)] min-h-[64px]">
          <Link
            to="/dashboard"
            className="flex items-center gap-2.5 whitespace-nowrap overflow-hidden"
          >
            {/* Logo icon */}
            <div className="w-8 h-8 flex items-center justify-center rounded-[var(--radius-sm)] bg-[var(--accent-green-dim)] border border-[var(--border-accent)] text-[var(--accent-green)] shrink-0">
              <Code2 size={18} />
            </div>
            {/* Logo text */}
            <span
              className={[
                'font-[var(--font-display)] font-bold text-[15px] text-[var(--text-primary)]',
                'transition-opacity duration-200',
                collapsed ? 'opacity-0 w-0' : 'opacity-100',
              ].join(' ')}
            >
              CP Roadmap
            </span>
          </Link>

          {/* Collapse button — hidden on mobile */}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className={[
              'hidden md:flex items-center p-1 rounded bg-none shrink-0',
              'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]',
              'transition-all duration-200',
              collapsed ? 'rotate-180' : '',
            ].join(' ')}
          >
            <ChevronRight size={14} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-3 flex flex-col gap-0.5">
          {navItems.map(({ path, label, icon: Icon }) => {
            const active = location.pathname === path
            return (
              <Link
                key={path}
                to={path}
                onClick={() => setMobileOpen(false)}
                className={[
                  'flex items-center gap-2.5 px-2.5 py-2.5',
                  'rounded-[var(--radius-md)] text-[14px] font-medium',
                  'whitespace-nowrap overflow-hidden relative',
                  'transition-all duration-200',
                  active
                    ? 'bg-[var(--accent-green-dim)] text-[var(--accent-green)]'
                    : 'text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]',
                ].join(' ')}
              >
                <Icon size={16} className="shrink-0" />

                <span
                  className={[
                    'transition-all duration-200',
                    collapsed ? 'opacity-0 w-0' : 'opacity-100',
                  ].join(' ')}
                >
                  {label}
                </span>

                {/* Active indicator bar */}
                {active && (
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 w-[3px] h-[60%] bg-[var(--accent-green)] rounded-l-[2px]" />
                )}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="px-2 py-3 border-t border-[var(--border-subtle)] flex flex-col gap-1">
          {/* User info */}
          <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-[var(--radius-md)] overflow-hidden">
            <div className="w-8 h-8 rounded-full shrink-0 flex items-center justify-center bg-[var(--accent-blue-dim)] border border-[var(--accent-blue)] text-[var(--accent-blue)]">
              <User size={14} />
            </div>
            <div
              className={[
                'flex flex-col overflow-hidden transition-all duration-200',
                collapsed ? 'opacity-0 w-0' : 'opacity-100',
              ].join(' ')}
            >
              <span className="text-[13px] font-semibold text-[var(--text-primary)] truncate">
                {user?.username}
              </span>
              <span className="text-[11px] text-[var(--text-muted)] truncate">
                {user?.email}
              </span>
            </div>
          </div>

          {/* Logout */}
          <button
            onClick={handleLogout}
            title="Logout"
            className={[
              'flex items-center gap-2.5 px-2.5 py-[9px] w-full',
              'rounded-[var(--radius-md)] text-[14px] font-medium',
              'whitespace-nowrap overflow-hidden bg-none',
              'text-[var(--text-muted)] hover:bg-red-500/10 hover:text-red-400',
              'transition-all duration-200',
            ].join(' ')}
          >
            <LogOut size={15} className="shrink-0" />
            <span
              className={[
                'transition-all duration-200',
                collapsed ? 'opacity-0 w-0' : 'opacity-100',
              ].join(' ')}
            >
              Logout
            </span>
          </button>
        </div>
      </aside>

      {/* ── Mobile overlay ── */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-[90] bg-black/50 backdrop-blur-sm md:hidden"
        />
      )}

      {/* ── Main content ── */}
      <main
        className={[
          'flex-1 min-h-screen flex flex-col',
          'transition-[margin-left] duration-300',
          // desktop offset matches sidebar width
          collapsed ? 'md:ml-[60px]' : 'md:ml-[220px]',
          // mobile: no offset
          'ml-0',
        ].join(' ')}
      >
        {/* Mobile topbar */}
        <div className="md:hidden flex items-center justify-between px-5 py-3.5 bg-[var(--bg-secondary)] border-b border-[var(--border-subtle)] sticky top-0 z-50">
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="bg-none text-[var(--text-secondary)] p-1"
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

          <Link
            to="/dashboard"
            className="flex items-center gap-2 font-[var(--font-display)] font-bold text-[15px] text-[var(--text-primary)]"
          >
            <Code2 size={16} />
            <span>CP Roadmap</span>
          </Link>

          {/* Mobile avatar */}
          <div className="w-7 h-7 rounded-full flex items-center justify-center bg-[var(--accent-blue-dim)] border border-[var(--accent-blue)] text-[var(--accent-blue)]">
            <User size={12} />
          </div>
        </div>

        {/* Page content */}
        <div className="px-8 py-8 max-w-[1200px] w-full mx-auto md:px-8 px-4 md:py-8 py-5" style={{ animation: 'fadeIn 0.4s ease' }}>
          {children}
        </div>
      </main>
    </div>
  )
}