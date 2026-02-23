from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class RoadmapGenerate(BaseModel):
    leetcode_username: Optional[str] = None
    codeforces_handle: Optional[str] = None
    session_hours: float = 2.0

class RoadmapResponse(BaseModel):
    id: int
    problems: List[Dict[str, Any]]
    weak_topics: Optional[List[str]]
    user_level: Optional[str]
    contest_penalty: Optional[float]
    session_plan: Optional[List[Dict[str, Any]]]  # fixed: List not Dict
    daily_calendar: Optional[List[Dict[str, Any]]]
    retention_data: Optional[Dict[str, Any]]
    gnn_data: Optional[Dict[str, Any]]
    ml_insights: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True