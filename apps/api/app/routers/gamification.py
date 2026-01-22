"""
Gamification API Endpoints
XP, levels, achievements, streaks, and skill trees
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from app.core.database import get_db
from app.models.user import User
from app.models.gamification import UserAchievement, UserStats, DailyActivity
from app.models.spaced_repetition import Concept
from app.models.assessment import UserConceptMastery
from app.gamification import GamificationEngine
from pydantic import BaseModel

router = APIRouter()


class XPAwardRequest(BaseModel):
    """Request to award XP"""
    user_id: int
    action: str
    multiplier: float = 1.0
    is_first_time: bool = False


@router.post("/xp/award")
async def award_xp(
    request: XPAwardRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Award XP to a user for an action

    Updates user level and checks for achievements
    """
    # Get user
    result = await db.execute(select(User).where(User.id == request.user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate XP
    xp_earned = GamificationEngine.award_xp(
        request.action,
        request.multiplier,
        request.is_first_time
    )

    # Add bonus XP from streaks
    bonus_xp = GamificationEngine.calculate_bonus_xp(
        xp_earned,
        user.streak_days,
        is_perfect_week=False  # TODO: Check if perfect week
    )

    total_xp = xp_earned + bonus_xp

    # Update user XP and level
    old_level = user.level
    user.total_xp += total_xp
    user.level = GamificationEngine.calculate_level(user.total_xp)
    user.last_activity_date = datetime.now()

    # Update streak
    new_streak, is_active_today = GamificationEngine.calculate_streak(
        user.last_activity_date,
        user.streak_days
    )

    if not is_active_today and new_streak > user.streak_days:
        # Increment streak (first activity today)
        user.streak_days = user.streak_days + 1
    elif new_streak == 0 and user.streak_days > 0:
        # Streak broken
        user.streak_days = 1  # Start new streak

    # Check for level up
    level_up = user.level > old_level

    await db.commit()
    await db.refresh(user)

    # Calculate progress to next level
    xp_needed, current_level, progress = GamificationEngine.xp_to_next_level(user.total_xp)

    return {
        "user_id": user.id,
        "xp_earned": xp_earned,
        "bonus_xp": bonus_xp,
        "total_xp_earned": total_xp,
        "total_xp": user.total_xp,
        "level": user.level,
        "level_up": level_up,
        "xp_to_next_level": xp_needed,
        "progress_percentage": progress,
        "streak_days": user.streak_days,
    }


@router.get("/profile/{user_id}")
async def get_user_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get user gamification profile"""
    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get or create stats
    stats_result = await db.execute(
        select(UserStats).where(UserStats.user_id == user_id)
    )
    stats = stats_result.scalar_one_or_none()

    if not stats:
        stats = UserStats(user_id=user_id)
        db.add(stats)
        await db.commit()
        await db.refresh(stats)

    # Get achievements
    achievements_result = await db.execute(
        select(UserAchievement).where(UserAchievement.user_id == user_id)
    )
    achievements = achievements_result.scalars().all()

    # Calculate level info
    xp_needed, current_level, progress = GamificationEngine.xp_to_next_level(user.total_xp)

    return {
        "user_id": user.id,
        "username": user.username,
        "level": user.level,
        "total_xp": user.total_xp,
        "xp_to_next_level": xp_needed,
        "level_progress": progress,
        "streak_days": user.streak_days,
        "last_activity": user.last_activity_date.isoformat() if user.last_activity_date else None,
        "stats": {
            "modules_completed": stats.modules_completed,
            "concepts_mastered": stats.concepts_mastered,
            "reviews_completed": stats.reviews_completed,
            "courses_completed": stats.courses_completed,
            "chat_messages": stats.chat_messages,
            "total_study_time_minutes": stats.total_study_time_minutes,
        },
        "achievements": [
            {
                "id": ach.achievement_id,
                "name": ach.name,
                "rarity": ach.rarity,
                "unlocked_at": ach.unlocked_at.isoformat(),
            }
            for ach in achievements
        ],
        "achievement_count": len(achievements),
    }


@router.get("/achievements")
async def get_achievements(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all achievements and user progress"""
    # Get user stats
    stats_result = await db.execute(
        select(UserStats).where(UserStats.user_id == user_id)
    )
    stats = stats_result.scalar_one_or_none()

    # Get user
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()

    if not stats or not user:
        # Return default
        stats_dict = {}
        user_dict = {"streak_days": 0}
    else:
        stats_dict = {
            "modules_completed": stats.modules_completed,
            "concepts_mastered": stats.concepts_mastered,
            "streak_days": user.streak_days,
            "reviews_completed": stats.reviews_completed,
            "perfect_streak": stats.perfect_streak,
            "chat_messages": stats.chat_messages,
            "courses_started": stats.courses_started,
            "courses_completed": stats.courses_completed,
        }

    # Get unlocked achievements
    unlocked_result = await db.execute(
        select(UserAchievement).where(UserAchievement.user_id == user_id)
    )
    unlocked = unlocked_result.scalars().all()
    unlocked_ids = [ach.achievement_id for ach in unlocked]

    # Check for newly unlocked achievements
    newly_unlocked = GamificationEngine.check_achievements(stats_dict, unlocked_ids)

    # Award newly unlocked achievements
    for achievement in newly_unlocked:
        user_achievement = UserAchievement(
            user_id=user_id,
            achievement_id=achievement.id,
            name=achievement.name,
            description=achievement.description,
            achievement_type=achievement.achievement_type.value,
            xp_reward=achievement.xp_reward,
            rarity=achievement.rarity,
        )
        db.add(user_achievement)

        # Award XP
        if user:
            user.total_xp += achievement.xp_reward

    if newly_unlocked:
        await db.commit()

    # Get all achievements with progress
    all_achievements = []
    for achievement in GamificationEngine.ACHIEVEMENTS:
        is_unlocked = achievement.id in unlocked_ids or any(
            a.id == achievement.id for a in newly_unlocked
        )

        # Calculate progress
        progress = 0
        if achievement.requirement:
            req_key = list(achievement.requirement.keys())[0]
            req_value = achievement.requirement[req_key]
            current_value = stats_dict.get(req_key, 0)
            progress = min(100, (current_value / req_value) * 100) if req_value > 0 else 0

        all_achievements.append({
            "id": achievement.id,
            "name": achievement.name,
            "description": achievement.description,
            "type": achievement.achievement_type.value,
            "xp_reward": achievement.xp_reward,
            "icon": achievement.icon,
            "rarity": achievement.rarity,
            "is_unlocked": is_unlocked,
            "progress": progress,
        })

    return {
        "user_id": user_id,
        "total_achievements": len(GamificationEngine.ACHIEVEMENTS),
        "unlocked_count": len(unlocked_ids) + len(newly_unlocked),
        "newly_unlocked": [
            {
                "id": ach.id,
                "name": ach.name,
                "xp_reward": ach.xp_reward,
            }
            for ach in newly_unlocked
        ],
        "achievements": all_achievements,
    }


@router.get("/skill-tree")
async def get_skill_tree(
    user_id: int,
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get skill tree for a course"""
    from app.worker.app.services.graph_service import GraphService

    # Get concepts for course
    concepts_result = await db.execute(
        select(Concept).where(Concept.course_id == course_id)
    )
    concepts = concepts_result.scalars().all()

    # Get user masteries
    mastery_result = await db.execute(
        select(UserConceptMastery).where(
            and_(
                UserConceptMastery.user_id == user_id,
                UserConceptMastery.concept_id.in_([c.id for c in concepts])
            )
        )
    )
    masteries = mastery_result.scalars().all()
    mastery_map = {m.concept_id: m.mastery_level for m in masteries}

    # Get prerequisites from knowledge graph
    graph_service = GraphService()
    prerequisites = {}

    for concept in concepts:
        prereqs = graph_service.get_concept_prerequisites(course_id, concept.name)
        # Map concept names back to IDs
        prereq_ids = []
        for prereq_name in prereqs:
            matching_concepts = [c for c in concepts if c.name == prereq_name]
            if matching_concepts:
                prereq_ids.append(matching_concepts[0].id)
        prerequisites[concept.id] = prereq_ids

    graph_service.close()

    # Build skill tree
    concept_dicts = [{"id": c.id, "name": c.name} for c in concepts]
    skill_nodes = GamificationEngine.build_skill_tree(
        concept_dicts,
        mastery_map,
        prerequisites
    )

    return {
        "user_id": user_id,
        "course_id": course_id,
        "nodes": [
            {
                "concept_id": node.concept_id,
                "concept_name": node.concept_name,
                "mastery_level": node.mastery_level,
                "is_unlocked": node.is_unlocked,
                "is_mastered": node.mastery_level >= 0.95,
                "prerequisites": node.prerequisites,
                "children": node.children,
            }
            for node in skill_nodes
        ],
    }


@router.get("/leaderboard")
async def get_leaderboard(
    course_id: Optional[int] = None,
    time_period: str = "all_time",  # all_time, weekly, monthly
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get leaderboard rankings"""
    # Get top users by XP
    result = await db.execute(
        select(User)
        .order_by(User.total_xp.desc())
        .limit(limit)
    )
    users = result.scalars().all()

    leaderboard = []
    for rank, user in enumerate(users, 1):
        leaderboard.append({
            "rank": rank,
            "user_id": user.id,
            "username": user.username,
            "level": user.level,
            "total_xp": user.total_xp,
            "streak_days": user.streak_days,
        })

    return {
        "time_period": time_period,
        "total_users": len(leaderboard),
        "leaderboard": leaderboard,
    }
