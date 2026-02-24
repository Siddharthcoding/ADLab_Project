# CP Roadmap — Frontend

Modern React frontend for the CP Roadmap Generator API.

## Tech Stack

- **React 18** + **Vite** — fast development and builds
- **React Router v6** — client-side routing
- **Axios** — API requests with proxy to backend
- **React Hot Toast** — toast notifications
- **Lucide React** — icons
- **Framer Motion** — animations (available for future use)

## Design System

- **Fonts**: Syne (display) + DM Sans (body) + DM Mono (code)
- **Theme**: Dark terminal aesthetic with neon green accent
- **CSS Custom Properties** for consistent theming throughout

## Setup

### Prerequisites
- Node.js 18+
- Backend running at `http://localhost:8000`

### Install & Run

```bash
npm install
npm run dev
```

App will be at `http://localhost:5173`

The Vite dev server proxies all `/api` requests to `http://localhost:8000`.

### Build for Production

```bash
npm run build
npm run preview
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/auth` | Login / Register |
| `/dashboard` | Overview, stats, recent roadmaps |
| `/generate` | Generate new roadmap form |
| `/roadmap/:id` | Full roadmap detail view |
| `/history` | All roadmaps list |

## Project Structure

```
src/
├── context/
│   └── AuthContext.jsx     # Auth state (JWT token, user)
├── utils/
│   └── api.js              # Axios instance with proxy config
├── components/
│   ├── Layout.jsx          # Sidebar + main layout
│   └── Layout.css
├── pages/
│   ├── LandingPage.jsx     # Public homepage
│   ├── AuthPage.jsx        # Login/Register tabs
│   ├── Dashboard.jsx       # Main dashboard
│   ├── GeneratePage.jsx    # Roadmap generation with progress
│   ├── RoadmapPage.jsx     # Tabbed roadmap detail view
│   └── HistoryPage.jsx     # History list
├── App.jsx                 # Router + auth guard
├── main.jsx
└── index.css               # Global styles + design tokens
```

## Notes

- The backend API must have CORS configured to allow `http://localhost:5173`
- Add to your backend `BACKEND_CORS_ORIGINS`: `"http://localhost:5173"`
- The roadmap detail page calls `GET /api/v1/roadmap/{id}` — make sure this endpoint exists in your backend

## CORS Setup

In your `docker-compose.yml`, update the backend environment to include the frontend origin:

```yaml
backend:
  environment:
    - BACKEND_CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```
