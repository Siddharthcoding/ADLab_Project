from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, Float, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base  # change this line
class Roadmap(Base):
    __tablename__ = "roadmaps"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    problems = Column(JSON, nullable=False)
    weak_topics = Column(JSON, nullable=True)
    user_level = Column(String(50), nullable=True)
    contest_penalty = Column(Float, nullable=True)
    
    session_plan = Column(JSON, nullable=True)
    daily_calendar = Column(JSON, nullable=True)
    retention_data = Column(JSON, nullable=True)
    gnn_data = Column(JSON, nullable=True)
    
    ml_insights = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="roadmaps")