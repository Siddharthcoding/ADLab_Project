# api/leetcode_api.py

import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import time

def fetch_leetcode_submissions(username, limit=500):  # Increased limit
    """
    Fetch recent LeetCode submissions (up to 500).
    """
    url = "https://leetcode.com/graphql"
    
    query = {
        "query": """
        query recentSubmissions($username: String!, $limit: Int!) {
          recentSubmissionList(username: $username, limit: $limit) {
            titleSlug
            statusDisplay
          }
        }
        """,
        "variables": {"username": username, "limit": limit}
    }
    
    try:
        response = requests.post(url, json=query, timeout=15)  # Increased timeout
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("errors"):
            error_msg = data["errors"][0].get("message", "Unknown error")
            print(f"⚠️  LeetCode API Error: {error_msg}")
            return []
        
        submissions = data.get("data", {}).get("recentSubmissionList")
        
        if submissions is None:
            print(f"⚠️  LeetCode user '{username}' not found or has no submissions")
            return []
        
        return submissions
        
    except ConnectionError:
        print("❌ Cannot connect to LeetCode. Please check your internet connection.")
        return []
    except Timeout:
        print("⏱️  LeetCode API timeout. The service might be slow.")
        return []
    except RequestException as e:
        print(f"❌ Error fetching LeetCode submissions: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error with LeetCode API: {str(e)}")
        return []

def fetch_leetcode_solved_stats(username):
    url = "https://leetcode.com/graphql"
    query = {
        "query": """
        query($username: String!) {
          matchedUser(username: $username) {
            submitStats: submitStatsGlobal {
              acSubmissionNum {
                difficulty
                count
              }
            }
          }
        }
        """,
        "variables": {"username": username}
    }
    
    try:
        response = requests.post(url, json=query, timeout=15)
        data = response.json()
        stats = data.get("data", {}).get("matchedUser", {}).get("submitStats", {}).get("acSubmissionNum", [])
        
        # ✅ FIXED: Proper difficulty mapping
        diff_map = {}
        total = 0
        for stat in stats:
            diff = stat.get("difficulty", "").lower()
            count = int(stat.get("count", 0))
            diff_map[diff] = count
            total += count
            
        easy = diff_map.get("easy", 0)
        medium = diff_map.get("medium", 0) 
        hard = diff_map.get("hard", 0)
        
        return total, [easy, medium, hard]
    except:
        return 0, [0, 0, 0]


def fetch_leetcode_contests(username):
    """
    Fetch LeetCode contest history (unchanged).
    """
    url = "https://leetcode.com/graphql"
    
    query = {
        "query": """
        query userContestRankingInfo($username: String!) {
          userContestRankingHistory(username: $username) {
            rating
            contest {
              title
            }
          }
        }
        """,
        "variables": {"username": username}
    }
    
    try:
        response = requests.post(url, json=query, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("errors"):
            return []
        
        contests = data.get("data", {}).get("userContestRankingHistory")
        return contests or []
        
    except:
        return []
