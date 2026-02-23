# api/codeforces_api.py

import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import time

def fetch_cf_submissions(handle):
    """
    Fetch Codeforces submissions with pagination (up to 10k).
    """
    url = f"https://codeforces.com/api/user.status?handle={handle}&from=1&count=10000"
    
    for attempt in range(3):  # Retry logic
        try:
            response = requests.get(url, timeout=20)  # Increased timeout
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                print(f"⚠️  Codeforces API Error: {data.get('comment', 'Unknown error')}")
                return []
            
            print(f"✓ Fetched {len(data.get('result', []))} Codeforces submissions")
            return data.get("result", [])
            
        except Timeout:
            print(f"⏱️  Codeforces API timeout (attempt {attempt+1}/3)")
            time.sleep(2)
        except ConnectionError:
            print("❌ Cannot connect to Codeforces.")
            return []
        except RequestException as e:
            print(f"❌ Error: {str(e)}")
            return []
    
    print("⏱️  Codeforces API failed after 3 retries.")
    return []

def fetch_cf_contests(handle):
    """
    Fetch Codeforces contest history (unchanged but improved timeout).
    """
    url = f"https://codeforces.com/api/user.rating?handle={handle}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "OK":
            return []
        
        return data.get("result", [])
        
    except:
        return []

def fetch_cf_problemset():
    """Unchanged."""
    url = "https://codeforces.com/api/problemset.problems"
    
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != "OK":
            return []
        
        return data.get("result", {}).get("problems", [])
        
    except:
        return []
