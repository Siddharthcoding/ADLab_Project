from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.db.session import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.roadmap import Roadmap
from app.schemas.roadmap import RoadmapGenerate, RoadmapResponse, ProblemToggle
from app.services.ml_service import MLService

router = APIRouter()

@router.post("/generate", response_model=RoadmapResponse)
async def generate_roadmap(
    request: RoadmapGenerate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        ml_service = MLService(db)
        roadmap = await ml_service.generate_complete_roadmap(
            user=current_user,
            leetcode_username=request.leetcode_username,
            codeforces_handle=request.codeforces_handle,
            session_hours=request.session_hours
        )
        return roadmap
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate roadmap: {str(e)}")

@router.get("/history", response_model=List[RoadmapResponse])
async def get_roadmap_history(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Roadmap)
        .where(Roadmap.user_id == current_user.id)
        .order_by(Roadmap.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

@router.get("/{roadmap_id}", response_model=RoadmapResponse)
async def get_roadmap(
    roadmap_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Roadmap).where(
            Roadmap.id == roadmap_id,
            Roadmap.user_id == current_user.id
        )
    )
    roadmap = result.scalar_one_or_none()
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    return roadmap

# NEW: Toggle problem completion
@router.post("/{roadmap_id}/toggle-problem", response_model=RoadmapResponse)
async def toggle_problem_completion(
    roadmap_id: int,
    toggle: ProblemToggle,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Roadmap).where(
            Roadmap.id == roadmap_id,
            Roadmap.user_id == current_user.id
        )
    )
    roadmap = result.scalar_one_or_none()
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    
    completed = roadmap.completed_problems or []
    
    if toggle.completed:
        if toggle.problem_index not in completed:
            completed.append(toggle.problem_index)
    else:
        if toggle.problem_index in completed:
            completed.remove(toggle.problem_index)
    
    roadmap.completed_problems = completed
    roadmap.progress_percentage = (len(completed) / len(roadmap.problems)) * 100 if roadmap.problems else 0
    
    await db.commit()
    await db.refresh(roadmap)
    return roadmap