import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

print("=" * 80)
print("🚀 CP ROADMAP API - COMPLETE TEST")
print("=" * 80)

# 1. Login
print("\n1️⃣  LOGIN")
print("-" * 80)
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={
        "username": "testuser",
        "password": "test123"
    }
)

if response.status_code != 200:
    print(f"❌ Login failed: {response.status_code}")
    print(response.text)
    exit(1)

token = response.json()["access_token"]
print(f"✅ Login successful!")
print(f"📋 Token: {token[:60]}...")

# 2. Test Authentication
print("\n2️⃣  AUTHENTICATION TEST")
print("-" * 80)
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{BASE_URL}/users/me", headers=headers)

if response.status_code != 200:
    print(f"❌ Authentication failed: {response.status_code}")
    print(response.text)
    exit(1)

user = response.json()
print(f"✅ Authenticated successfully!")
print(f"👤 User: {user['username']}")
print(f"📧 Email: {user['email']}")
print(f"🆔 ID: {user['id']}")

# 3. Generate Roadmap
print("\n3️⃣  ROADMAP GENERATION")
print("-" * 80)
print("⏳ Starting ML pipeline (this takes 30-60 seconds)...")
print("   - Fetching LeetCode & Codeforces data")
print("   - Loading problem catalogs")
print("   - Running ML analysis")
print("   - Generating personalized roadmap")

start_time = time.time()

response = requests.post(
    f"{BASE_URL}/roadmap/generate",
    json={
        "leetcode_username": "SiddharthKumarMishra",
        "codeforces_handle": "siddharthkumarmishra",
        "session_hours": 3
    },
    headers=headers,
    timeout=180
)

elapsed = time.time() - start_time

if response.status_code != 200:
    print(f"\n❌ Roadmap generation failed: {response.status_code}")
    print(response.text)
    exit(1)

roadmap = response.json()
print(f"\n✅ Roadmap generated successfully in {elapsed:.1f} seconds!")

# Display Results
print("\n" + "=" * 80)
print("📊 ROADMAP ANALYSIS")
print("=" * 80)

print(f"\n🎓 User Profile:")
print(f"   Level: {roadmap['user_level']}")
print(f"   Contest Penalty: {roadmap['contest_penalty']:.3f}")
print(f"   Total Problems: {len(roadmap['problems'])}")

print(f"\n⚠️  Weak Topics:")
for i, topic in enumerate(roadmap['weak_topics'][:8], 1):
    print(f"   {i}. {topic}")

print(f"\n🎯 TOP 10 RECOMMENDED PROBLEMS:")
print("-" * 80)
for i, p in enumerate(roadmap['problems'][:10], 1):
    source_icon = "🔷" if p['source'] == "LeetCode" else "🔶"
    print(f"\n{i}. {source_icon} {p['name']}")
    print(f"   Difficulty: {p['difficulty']} | Source: {p['source']}")
    print(f"   Tags: {', '.join(p['tags'][:4])}")
    if p.get('ml_explanation'):
        print(f"   💡 ML Insight: {p['ml_explanation']}")
    if p.get('link'):
        print(f"   🔗 {p['link']}")

# Session Plan
if roadmap.get('session_plan'):
    session = roadmap['session_plan']
    if isinstance(session, list) and len(session) > 0:
        print(f"\n⚡ OPTIMAL {roadmap.get('session_hours', 3)}-HOUR SESSION PLAN:")
        print("-" * 80)
        print(f"   Problems in session: {len(session)}")
        for i, sp in enumerate(session[:5], 1):
            print(f"   {i}. {sp.get('name', 'Unknown')} ({sp.get('est_minutes', 20)} min)")

# Daily Calendar
if roadmap.get('daily_calendar'):
    calendar = roadmap['daily_calendar']
    if len(calendar) > 0:
        print(f"\n📅 7-DAY PRACTICE CALENDAR:")
        print("-" * 80)
        for day in calendar[:3]:
            print(f"   Day {day['day']} - {day['label']}")
            print(f"   🎯 {day['goal']}")
            print(f"   📌 Focus: {', '.join(day['focus_topics'][:3])}")
            if 'roi' in day:
                print(f"   📈 ROI: +{day['roi']:.1f} rating points/hr")
            print()

# Retention Data
if roadmap.get('retention_data'):
    retention = roadmap['retention_data']
    if retention.get('at_risk'):
        print(f"\n⚠️  AT-RISK TOPICS (Need Review):")
        print("-" * 80)
        for topic in retention['at_risk'][:5]:
            print(f"   • {topic['tag']}: {topic['retention']:.0%} retention "
                  f"(last seen {topic['last_seen_days']:.0f} days ago)")

# GNN Data
if roadmap.get('gnn_data') and roadmap['gnn_data'].get('hidden_gaps'):
    print(f"\n🕸️  HIDDEN MASTERY GAPS (GNN Analysis):")
    print("-" * 80)
    for gap in roadmap['gnn_data']['hidden_gaps'][:3]:
        print(f"   Topic: '{gap['topic']}'")
        print(f"   Apparent Retention: {gap['apparent_retention']:.0%}")
        print(f"   True Confidence: {gap['true_confidence']:.0%}")
        print(f"   Weak Prerequisites: {', '.join(gap['weak_prerequisites'])}")
        print()

# ML Insights
if roadmap.get('ml_insights'):
    print(f"\n🧠 ML INSIGHTS:")
    print("-" * 80)
    print(f"   {roadmap['ml_insights']}")

# 4. Get History
print("\n4️⃣  ROADMAP HISTORY")
print("-" * 80)
response = requests.get(f"{BASE_URL}/roadmap/history", headers=headers)
history = response.json()
print(f"✅ Total roadmaps generated: {len(history)}")
for h in history:
    print(f"   • ID: {h['id']} | Created: {h['created_at'][:19]}")

print("\n" + "=" * 80)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\n💡 Your ML-powered CP Roadmap API is fully functional!")
print("   You can now integrate this with your frontend or use it via API calls.")