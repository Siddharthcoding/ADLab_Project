# CP Roadmap Generator - FastAPI Backend

Complete production-ready backend with PostgreSQL and ML integration.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OR Python 3.11+ with PostgreSQL locally

### Option 1: Docker (Recommended)
```bash
# 1. Copy YOUR ML code to the app/ folder
cp -r /your/project/ml app/
cp -r /your/project/api app/
cp -r /your/project/analysis app/
cp -r /your/project/preprocess app/
cp -r /your/project/recommender app/
cp /your/project/data/leetcode_questions.csv data/

# 2. Setup environment
cp .env.example .env
# Edit .env and set a secure password and secret key

# 3. Start everything
docker-compose up --build

# 4. Access API
# Swagger Docs: http://localhost:8000/api/v1/docs
# ReDoc: http://localhost:8000/api/v1/redoc
```

### Option 2: Local Development
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy your ML code (same as Docker option 1)

# 4. Setup PostgreSQL and Redis locally

# 5. Setup environment
cp .env.example .env
# Edit DATABASE_URL to point to your local PostgreSQL

# 6. Run migrations
alembic upgrade head

# 7. Start server
uvicorn app.main:app --reload --port 8000
```

## 📡 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login (returns JWT token)

### Users
- `GET /api/v1/users/me` - Get current user info

### Roadmap
- `POST /api/v1/roadmap/generate` - Generate ML-powered roadmap
- `GET /api/v1/roadmap/history` - Get user's roadmap history
- `GET /api/v1/roadmap/{id}` - Get specific roadmap

## 🧪 Testing the API
```bash
# 1. Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"testuser","password":"test123"}'

# 2. Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=test123"

# Copy the access_token from response

# 3. Generate roadmap
curl -X POST http://localhost:8000/api/v1/roadmap/generate \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "leetcode_username": "SiddharthKumarMishra",
    "codeforces_handle": "siddharthkumarmishra",
    "session_hours": 3.0
  }'

# 4. Get roadmap history
curl -X GET http://localhost:8000/api/v1/roadmap/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 📁 Project Structure
```
app/
├── ml/              # Copy your ml/ folder here
├── api/             # Copy your api/ folder here (leetcode_api.py, codeforces_api.py)
├── analysis/        # Copy your analysis/ folder here
├── preprocess/      # Copy your preprocess/ folder here
├── recommender/     # Copy your recommender/ folder here
├── main.py          # FastAPI application
├── core/            # Configuration, security, dependencies
├── db/              # Database setup and session management
├── models/          # SQLAlchemy database models
├── schemas/         # Pydantic request/response schemas
├── api/v1/          # API route handlers
└── services/        # ML service integration layer
```

## 💾 Database Schema

### Users Table
- id, email, username, hashed_password
- is_active, is_superuser
- created_at, updated_at

### User Profiles Table
- user_id (FK to users)
- leetcode_username, codeforces_handle
- lc_stats, cf_stats (JSON)
- weak_topics (JSON)

### Roadmaps Table
- user_id (FK to users)
- **problems** (JSON) - Top 50 ML-ranked problems
- **session_plan** (JSON) - Thompson Sampling session
- **daily_calendar** (JSON) - 7-day contest prep
- **retention_data** (JSON) - Forgetting curve analysis
- **gnn_data** (JSON) - Knowledge graph insights
- **weak_topics** (JSON) - Detected weaknesses
- **ml_insights** (Text) - Summary insights
- created_at, is_active

## 🔧 Environment Variables

Edit `.env` file:
```env
# Database
POSTGRES_PASSWORD=your_secure_password

# Security (IMPORTANT!)
SECRET_KEY=your-32-plus-character-secret-key

# Optional
POSTGRES_USER=postgres
POSTGRES_DB=cp_roadmap
```

## 🐛 Troubleshooting

### ML modules not loading
```bash
# Make sure you copied all folders
ls app/ml/
ls app/api/
ls app/analysis/
ls app/preprocess/
ls app/recommender/
ls data/
```

### Database connection error
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres
```

### Import errors
```bash
# Make sure __init__.py exists in all directories
find app -name __init__.py
```

## 🚀 Production Deployment

1. Generate a strong SECRET_KEY:
```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. Set strong POSTGRES_PASSWORD in `.env`

3. Set `DEBUG=False` in `.env`

4. Use managed PostgreSQL (AWS RDS, Google Cloud SQL, etc.)

5. Deploy backend container to cloud (AWS ECS, Google Cloud Run, etc.)

6. Set up HTTPS with SSL certificate

7. Configure proper CORS origins

## 📊 API Documentation

After starting the server:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## 💡 How It Works

1. User registers and logs in
2. User provides LeetCode/Codeforces handles
3. Backend fetches submission data from APIs
4. Your ML pipeline runs:
   - Forgetting curve analysis
   - GNN knowledge graph
   - Session planning
   - Retention forecasting
5. Results stored in PostgreSQL
6. User can retrieve past roadmaps anytime

## 📄 License

MIT License