"""
ML Service - Integrates your complete ML pipeline with FastAPI
This is the main integration point between your ML code and the FastAPI backend.

INSTRUCTIONS:
1. Copy your folders to app/:
   - app/ml/ (your ml folder)
   - app/api/ (leetcode_api.py, codeforces_api.py)
   - app/analysis/ (weakness_analysis.py, contest_analysis.py)
   - app/preprocess/ (leetcode_catalog.py, codeforces_catalog.py, normalize.py)
   - app/recommender/ (roadmap.py)
   - data/leetcode_questions.csv

2. This service will automatically:
   - Import your modules
   - Fetch platform data
   - Run your complete ML pipeline
   - Store results in PostgreSQL
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.models.roadmap import Roadmap
from app.models.profile import UserProfile
import pandas as pd
from datetime import datetime
import sys
import os

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import your ML modules
try:
    from app.api.leetcode_api import (
        fetch_leetcode_submissions,
        fetch_leetcode_solved_stats,
        fetch_leetcode_contests
    )
    from app.api.codeforces_api import (
        fetch_cf_submissions,
        fetch_cf_contests
    )
    from app.preprocess.leetcode_catalog import load_leetcode_catalog
    from app.preprocess.codeforces_catalog import load_cf_catalog
    from app.preprocess.normalize import normalize_cf, normalize_lc
    from app.analysis.weakness_analysis import detect_weak_topics
    from app.analysis.contest_analysis import contest_penalty_cf, contest_penalty_lc
    from app.recommender.roadmap import generate_roadmap
    from app.ml.pipeline import MLPipeline
    ML_AVAILABLE = True
    print("✅ All ML modules loaded successfully")
except ImportError as e:
    print(f"⚠️  ML modules not available: {e}")
    print("Please copy your ml/, api/, analysis/, preprocess/, recommender/ folders to app/")
    ML_AVAILABLE = False


class MLService:
    """
    Service class that wraps your complete ML pipeline.
    Handles fetching data, running ML, and storing results.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def generate_complete_roadmap(
        self,
        user: User,
        leetcode_username: str | None,
        codeforces_handle: str | None,
        session_hours: float
    ) -> Roadmap:
        """
        Main method - generates complete roadmap with all ML features.
        
        This method:
        1. Fetches data from LeetCode and Codeforces
        2. Loads problem catalogs
        3. Detects weak topics
        4. Runs your complete ML pipeline
        5. Extracts session plans, retention data, GNN insights
        6. Saves everything to database
        
        Returns: Roadmap object with all data
        """
        
        if not ML_AVAILABLE:
            print("⚠️  ML modules not available, returning fallback roadmap")
            return await self._generate_fallback_roadmap(user, leetcode_username, codeforces_handle)
        
        print(f"\n{'='*70}")
        print(f"🚀 Generating ML-powered roadmap for user: {user.username}")
        print(f"   LeetCode: {leetcode_username or 'Not provided'}")
        print(f"   Codeforces: {codeforces_handle or 'Not provided'}")
        print(f"   Session hours: {session_hours}")
        print(f"{'='*70}\n")
        
        # ===================================================================
        # STEP 1: Fetch platform data
        # ===================================================================
        print("📥 Step 1: Fetching platform data...")
        lc_subs, lc_total_solved, lc_contests = await self._fetch_leetcode_data(leetcode_username)
        cf_subs, cf_total_solved, cf_contests = await self._fetch_codeforces_data(codeforces_handle)
        
        print(f"   ✓ LeetCode: {lc_total_solved} solved, {len(lc_subs)} submissions")
        print(f"   ✓ Codeforces: {cf_total_solved} solved, {len(cf_subs)} submissions")
        
        # ===================================================================
        # STEP 2: Load problem catalogs
        # ===================================================================
        print("\n📚 Step 2: Loading problem catalogs...")
        lc_df = normalize_lc(load_leetcode_catalog()) if leetcode_username else pd.DataFrame()
        cf_df = normalize_cf(load_cf_catalog()) if codeforces_handle else pd.DataFrame()
        
        print(f"   ✓ LeetCode catalog: {len(lc_df)} problems")
        print(f"   ✓ Codeforces catalog: {len(cf_df)} problems")
        
        # ===================================================================
        # STEP 3: Detect weak topics
        # ===================================================================
        print("\n🔍 Step 3: Analyzing weak topics...")
        solved_lc = set(s.get("titleSlug") for s in lc_subs if s.get("statusDisplay") == "Accepted")
        solved_cf = set()
        for s in cf_subs:
            if s.get("verdict") == "OK":
                p = s.get("problem", {})
                if p.get("contestId") and p.get("index"):
                    solved_cf.add(f"{p['contestId']}{p['index']}")
        
        weak_topics = detect_weak_topics(
            cf_df=cf_df,
            lc_df=lc_df,
            solved_cf=solved_cf,
            solved_lc=solved_lc,
            cf_contests=cf_contests,
            lc_contests=lc_contests,
            lc_total_solved=lc_total_solved,
            threshold=0.35,
            min_attempts=8,
            adaptive_min=True
        )
        
        print(f"   ✓ Detected weak topics: {weak_topics[:5]}")
        
        # ===================================================================
        # STEP 4: Calculate contest penalty
        # ===================================================================
        print("\n🏆 Step 4: Calculating contest performance...")
        cf_penalty = contest_penalty_cf(cf_contests) if cf_contests else 0.5
        lc_penalty = contest_penalty_lc(lc_contests) if lc_contests else 0.5
        contest_penalty = round((cf_penalty + lc_penalty) / 2, 3)
        
        print(f"   ✓ Contest penalty: {contest_penalty}")
        
        # ===================================================================
        # STEP 5: Generate base roadmap
        # ===================================================================
        print("\n🗺️  Step 5: Generating base roadmap...")
        base_roadmap = generate_roadmap(
            lc_df=lc_df,
            cf_df=cf_df,
            weak_topics=weak_topics,
            solved_lc=solved_lc,
            solved_cf=solved_cf,
            contest_penalty=contest_penalty,
            max_items=50,
            balance_platforms=(cf_total_solved > 0)
        )
        
        print(f"   ✓ Generated {len(base_roadmap)} base problems")
        
        # ===================================================================
        # STEP 6: Run ML pipeline (YOUR MAIN ML CODE)
        # ===================================================================
        print("\n🤖 Step 6: Running ML pipeline...")
        ml = MLPipeline()
        enhanced_roadmap = ml.run(
            lc_subs=lc_subs,
            cf_subs=cf_subs,
            lc_df=lc_df,
            cf_df=cf_df,
            roadmap=base_roadmap,
            weak_topics=weak_topics,
            contest_penalty=contest_penalty,
            lc_total_solved=lc_total_solved,
            cf_total_solved=cf_total_solved
        )
        
        print(f"   ✓ ML pipeline complete, enhanced {len(enhanced_roadmap)} problems")
        
        # ===================================================================
        # STEP 7: Calculate user Elo
        # ===================================================================
        total_solved = lc_total_solved + cf_total_solved
        elo_base = min(800 + total_solved // 3, 2200)
        user_elo = max(800, int(elo_base * (1.0 - contest_penalty * 0.3)))
        
        # ===================================================================
        # STEP 8: Extract ML insights
        # ===================================================================
        print("\n📊 Step 8: Extracting ML insights...")
        
        # Session plan
        session_plan_data = None
        daily_calendar_data = None
        if ml.session_planner:
            try:
                session_plan_data = ml.session_planner.build_session_plan(hours=session_hours)
                daily_calendar_data = ml.session_planner.build_contest_calendar()
                print(f"   ✓ Session plan: {len(session_plan_data) if session_plan_data else 0} problems")
            except Exception as e:
                print(f"   ⚠️  Session planner error: {e}")
        
        # Retention data
        retention_data = None
        if ml.retention_report:
            retention_data = {
                "at_risk": [
                    {"tag": e["tag"], "retention": e["retention"], "last_seen_days": e["last_seen_days"]}
                    for e in ml.retention_report.at_risk[:10]
                ],
                "strong": [
                    {"tag": e["tag"], "retention": e["retention"]}
                    for e in ml.retention_report.strong[:10]
                ]
            }
            print(f"   ✓ Retention data: {len(ml.retention_report.at_risk)} at-risk topics")
        
        # GNN data
        gnn_data = None
        if ml.gnn_confidences and ml.knowledge_graph:
            hidden_gaps = ml.knowledge_graph.detect_hidden_gaps()
            gnn_data = {
                "confidences": ml.gnn_confidences,
                "hidden_gaps": [
                    {
                        "topic": g.topic,
                        "apparent_retention": g.apparent_retention,
                        "true_confidence": g.true_confidence,
                        "weak_prerequisites": g.weak_prerequisites
                    }
                    for g in hidden_gaps[:5]
                ]
            }
            print(f"   ✓ GNN data: {len(hidden_gaps)} hidden gaps detected")
        
        # ML insights summary text
        ml_insights_text = self._generate_insights_text(ml, user_elo, weak_topics)
        
        # ===================================================================
        # STEP 9: Save to database
        # ===================================================================
        print("\n💾 Step 9: Saving to database...")
        
        roadmap = Roadmap(
            user_id=user.id,
            problems=enhanced_roadmap[:50],  # Top 50 problems
            weak_topics=weak_topics,
            user_level=self._get_user_level(total_solved),
            contest_penalty=contest_penalty,
            session_plan=session_plan_data,
            daily_calendar=[
                {
                    "day": p.day,
                    "label": p.label,
                    "focus_topics": p.focus_topics,
                    "goal": p.session_goal,
                    "roi": p.expected_retention_gain
                } for p in (daily_calendar_data or [])
            ],
            retention_data=retention_data,
            gnn_data=gnn_data,
            ml_insights=ml_insights_text
        )
        
        self.db.add(roadmap)
        await self.db.commit()
        await self.db.refresh(roadmap)
        
        print(f"   ✓ Roadmap saved with ID: {roadmap.id}")
        
        # ===================================================================
        # STEP 10: Update user profile cache
        # ===================================================================
        result = await self.db.execute(
            select(UserProfile).where(UserProfile.user_id == user.id)
        )
        profile = result.scalar_one_or_none()
        if profile:
            profile.leetcode_username = leetcode_username
            profile.codeforces_handle = codeforces_handle
            profile.weak_topics = weak_topics
            profile.lc_stats = {"total_solved": lc_total_solved}
            profile.cf_stats = {"total_solved": cf_total_solved}
            await self.db.commit()
            print(f"   ✓ User profile updated")
        
        print(f"\n{'='*70}")
        print(f"✅ ROADMAP GENERATION COMPLETE")
        print(f"   Generated {len(enhanced_roadmap)} problems")
        print(f"   User level: {self._get_user_level(total_solved)}")
        print(f"   Weak topics: {len(weak_topics)}")
        print(f"{'='*70}\n")
        
        return roadmap
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    async def _fetch_leetcode_data(self, username: str | None):
        """Fetch LeetCode submissions and stats"""
        if not username:
            return [], 0, []
        try:
            print(f"      Fetching LeetCode data for {username}...")
            subs = fetch_leetcode_submissions(username, limit=500)
            total, stats = fetch_leetcode_solved_stats(username)
            contests = fetch_leetcode_contests(username)
            return subs, total, contests
        except Exception as e:
            print(f"      ⚠️  Error fetching LeetCode data: {e}")
            return [], 0, []
    
    async def _fetch_codeforces_data(self, handle: str | None):
        """Fetch Codeforces submissions and stats"""
        if not handle:
            return [], 0, []
        try:
            print(f"      Fetching Codeforces data for {handle}...")
            subs = fetch_cf_submissions(handle)
            total = sum(1 for s in subs if s.get("verdict") == "OK")
            contests = fetch_cf_contests(handle)
            return subs, total, contests
        except Exception as e:
            print(f"      ⚠️  Error fetching Codeforces data: {e}")
            return [], 0, []
    
    async def _generate_fallback_roadmap(self, user: User, lc_username: str | None, cf_handle: str | None):
        """Generate a basic roadmap if ML modules are not available"""
        roadmap = Roadmap(
            user_id=user.id,
            problems=[
                {"name": "Two Sum", "difficulty": 800, "tags": ["array", "hash table"], "source": "LeetCode"},
                {"name": "Add Two Numbers", "difficulty": 1200, "tags": ["linked list", "math"], "source": "LeetCode"},
                {"name": "Valid Parentheses", "difficulty": 800, "tags": ["string", "stack"], "source": "LeetCode"},
            ],
            weak_topics=["array", "hash table", "string"],
            user_level="Intermediate",
            contest_penalty=0.5,
            ml_insights="Fallback roadmap - ML modules not loaded"
        )
        self.db.add(roadmap)
        await self.db.commit()
        await self.db.refresh(roadmap)
        return roadmap
    
    @staticmethod
    def _get_user_level(total_solved: int) -> str:
        """Determine user skill level from solve count"""
        if total_solved == 0:
            return "Absolute Beginner"
        if total_solved < 25:
            return "Beginner"
        if total_solved < 100:
            return "Intermediate"
        if total_solved < 300:
            return "Advanced"
        return "Expert"
    
    @staticmethod
    def _generate_insights_text(ml, user_elo: int, weak_topics: list) -> str:
        """Generate summary text of ML insights"""
        insights = []
        
        # At-risk topics
        if ml.retention_report and ml.retention_report.at_risk:
            at_risk = ml.retention_report.at_risk[:3]
            insights.append(f"At-risk: {', '.join([e['tag'] for e in at_risk])}")
        
        # GNN weakest
        if ml.gnn_confidences:
            sorted_conf = sorted(ml.gnn_confidences.items(), key=lambda x: x[1])
            weakest = sorted_conf[:3]
            insights.append(f"GNN weakest: {', '.join([t for t, c in weakest])}")
        
        # User Elo
        insights.append(f"Elo: ~{user_elo}")
        
        # Weak topics
        if weak_topics:
            insights.append(f"Weak: {', '.join(weak_topics[:5])}")
        
        return " | ".join(insights)