"""
Gamification Engine
XP, levels, streaks, achievements, and skill trees based on Octalysis framework
"""
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
import math


class AchievementType(str, Enum):
    """Types of achievements"""
    MILESTONE = "milestone"  # Course/concept milestones
    STREAK = "streak"  # Daily/weekly streaks
    MASTERY = "mastery"  # Mastery achievements
    SOCIAL = "social"  # Social achievements
    EXPLORER = "explorer"  # Exploration achievements


class Achievement(BaseModel):
    """Achievement definition"""
    id: str
    name: str
    description: str
    achievement_type: AchievementType
    xp_reward: int
    icon: str
    requirement: Dict  # Flexible requirement definition
    rarity: str = "common"  # common, rare, epic, legendary


class SkillNode(BaseModel):
    """Skill tree node representing a concept"""
    concept_id: int
    concept_name: str
    mastery_level: float
    is_unlocked: bool
    prerequisites: List[int]
    children: List[int]
    xp_to_unlock: int = 0


class GamificationEngine:
    """
    Gamification system based on Octalysis framework

    Core drives:
    1. Epic Meaning: Skill trees, learning paths
    2. Development & Accomplishment: XP, levels, achievements
    3. Empowerment: Choice in learning path
    4. Ownership: Progress tracking
    5. Social Influence: Leaderboards (optional)
    6. Scarcity: Limited-time challenges
    7. Unpredictability: Random rewards
    8. Avoidance: Streak maintenance
    """

    # Age Groups (Research-based stages)
    AGE_EARLY = "early_childhood"  # 3-7
    AGE_MIDDLE = "middle_childhood" # 8-12
    AGE_ADOLESCENT = "adolescence" # 13-18
    AGE_ADULT = "adult" # 18+

    # XP configuration
    BASE_XP = 100  # XP for level 1
    LEVEL_MULTIPLIER = 1.5  # Exponential growth

    # XP rewards
    XP_REWARDS = {
        "review_card": 10,
        "review_card_first_time": 20,
        "module_complete": 50,
        "concept_mastered": 100,
        "course_complete": 500,
        "daily_streak": 25,
        "weekly_streak": 100,
        "perfect_week": 200,
        "chat_interaction": 5,
        "video_complete": 30,
    }

    # Achievements
    ACHIEVEMENTS = [
        Achievement(
            id="first_steps",
            name="First Steps",
            description="Complete your first module",
            achievement_type=AchievementType.MILESTONE,
            xp_reward=50,
            icon="🎯",
            requirement={"modules_completed": 1},
        ),
        Achievement(
            id="knowledge_seeker",
            name="Knowledge Seeker",
            description="Complete 10 modules",
            achievement_type=AchievementType.MILESTONE,
            xp_reward=200,
            icon="📚",
            requirement={"modules_completed": 10},
        ),
        Achievement(
            id="week_warrior",
            name="Week Warrior",
            description="Maintain a 7-day learning streak",
            achievement_type=AchievementType.STREAK,
            xp_reward=150,
            icon="🔥",
            requirement={"streak_days": 7},
        ),
        Achievement(
            id="month_master",
            name="Month Master",
            description="Maintain a 30-day learning streak",
            achievement_type=AchievementType.STREAK,
            xp_reward=500,
            icon="⭐",
            requirement={"streak_days": 30},
            rarity="epic",
        ),
        Achievement(
            id="concept_crusher",
            name="Concept Crusher",
            description="Master 10 concepts",
            achievement_type=AchievementType.MASTERY,
            xp_reward=300,
            icon="💎",
            requirement={"concepts_mastered": 10},
        ),
        Achievement(
            id="perfect_recall",
            name="Perfect Recall",
            description="Get 50 reviews correct in a row",
            achievement_type=AchievementType.MASTERY,
            xp_reward=250,
            icon="🧠",
            requirement={"perfect_streak": 50},
        ),
        Achievement(
            id="curious_mind",
            name="Curious Mind",
            description="Ask 100 questions in chat",
            achievement_type=AchievementType.EXPLORER,
            xp_reward=100,
            icon="❓",
            requirement={"chat_messages": 100},
        ),
        Achievement(
            id="speed_learner",
            name="Speed Learner",
            description="Complete a course in under 7 days",
            achievement_type=AchievementType.MILESTONE,
            xp_reward=400,
            icon="⚡",
            requirement={"course_completion_days": 7},
            rarity="rare",
        ),
    ]

    @staticmethod
    def get_level_curve(age_group: str) -> float:
        """
        Research indicates different tolerance for progression speed.
        Early childhood needs faster initial progression (higher feedback density).
        """
        if age_group == GamificationEngine.AGE_EARLY:
            return 1.2 # Faster leveling
        elif age_group == GamificationEngine.AGE_ADOLESCENT:
            return 1.8 # Slower, more meaningful leveling
        return 1.5

    @staticmethod
    def calculate_level(total_xp: int, age_group: str = "adult") -> int:
        """
        Calculate level from total XP using age-weighted formula.
        
        Formula: Level = floor(1 + (XP / BASE_XP) ^ (1/multiplier))
        """
        multiplier = GamificationEngine.get_level_curve(age_group)
        if total_xp <= 0:
            return 1
        return math.floor(1 + (total_xp / GamificationEngine.BASE_XP) ** (1/multiplier))

    @staticmethod
    def xp_for_level(level: int, age_group: str = "adult") -> int:
        """
        Calculate total XP needed to reach a level
        """
        multiplier = GamificationEngine.get_level_curve(age_group)
        return int(GamificationEngine.BASE_XP * (level - 1) ** multiplier)

    @staticmethod
    def xp_to_next_level(current_xp: int, age_group: str = "adult") -> Tuple[int, int, float]:
        """
        Calculate XP needed for next level
        """
        current_level = GamificationEngine.calculate_level(current_xp, age_group)
        next_level_xp = GamificationEngine.xp_for_level(current_level + 1, age_group)
        current_level_xp = GamificationEngine.xp_for_level(current_level, age_group)

        xp_needed = max(0, next_level_xp - current_xp)
        xp_in_level = max(0, current_xp - current_level_xp)
        xp_for_current_level = max(1, next_level_xp - current_level_xp)

        progress = (xp_in_level / xp_for_current_level) * 100

        return xp_needed, current_level, progress

    @staticmethod
    def calculate_streak(
        last_activity_date: Optional[datetime],
        current_streak: int,
        streak_shields_available: int = 0
    ) -> Tuple[int, bool, bool]:
        """
        Calculate current streak with "Streak Shield" (Research-backed loss aversion protection).
        """
        if not last_activity_date:
            return 0, False, False

        now = datetime.now()
        today = now.date()
        last_date = last_activity_date.date()

        # Check if activity today
        if last_date == today:
            return current_streak, True, False

        # Check if yesterday (streak continues)
        if last_date == today - timedelta(days=1):
            return current_streak, False, False

        # Streak potentially broken - check for shield
        if streak_shields_available > 0:
            # If they missed yesterday (last_date was 2 days ago), shield triggers today
            if last_date == today - timedelta(days=2):
                return current_streak, False, True

        # Streak broken
        return 0, False, False

    @staticmethod
    def award_xp(
        action: str,
        multiplier: float = 1.0,
        is_first_time: bool = False
    ) -> int:
        """
        Award XP for an action
        """
        base_xp = GamificationEngine.XP_REWARDS.get(action, 5)

        if is_first_time and action == "review_card":
            base_xp = GamificationEngine.XP_REWARDS["review_card_first_time"]

        return int(base_xp * multiplier)

    @staticmethod
    def check_achievements(
        user_stats: Dict,
        unlocked_achievement_ids: List[str]
    ) -> List[Achievement]:
        """
        Check which new achievements are unlocked
        """
        newly_unlocked = []

        for achievement in GamificationEngine.ACHIEVEMENTS:
            # Skip if already unlocked
            if achievement.id in unlocked_achievement_ids:
                continue

            # Check requirement
            requirement_met = True
            for key, required_value in achievement.requirement.items():
                user_value = user_stats.get(key, 0)

                if isinstance(required_value, int):
                    if user_value < required_value:
                        requirement_met = False
                        break

            if requirement_met:
                newly_unlocked.append(achievement)

        return newly_unlocked

    @staticmethod
    def build_skill_tree(
        concepts: List[Dict],
        user_masteries: Dict[int, float],
        prerequisites: Dict[int, List[int]]
    ) -> List[SkillNode]:
        """
        Build skill tree from concepts
        """
        skill_nodes = []

        for concept in concepts:
            concept_id = concept["id"]
            mastery = user_masteries.get(concept_id, 0.0)
            prereqs = prerequisites.get(concept_id, [])

            # Check if unlocked (prerequisites mastered)
            is_unlocked = True
            if prereqs:
                for prereq_id in prereqs:
                    prereq_mastery = user_masteries.get(prereq_id, 0.0)
                    if prereq_mastery < 0.7:  # 70% mastery threshold
                        is_unlocked = False
                        break

            # Find children (concepts that depend on this one)
            children = [
                cid for cid, cprereqs in prerequisites.items()
                if concept_id in cprereqs
            ]

            skill_node = SkillNode(
                concept_id=concept_id,
                concept_name=concept["name"],
                mastery_level=mastery,
                is_unlocked=is_unlocked,
                prerequisites=prereqs,
                children=children,
            )

            skill_nodes.append(skill_node)

        return skill_nodes

    @staticmethod
    def get_leaderboard_rank(
        user_xp: int,
        all_users_xp: List[int]
    ) -> Tuple[int, int, float]:
        """
        Calculate leaderboard ranking
        """
        sorted_xp = sorted(all_users_xp, reverse=True)

        try:
            rank = sorted_xp.index(user_xp) + 1
        except ValueError:
            rank = len(sorted_xp) + 1

        total_users = len(sorted_xp)
        percentile = ((total_users - rank) / total_users) * 100 if total_users > 0 else 0

        return rank, total_users, percentile

    @staticmethod
    def calculate_bonus_xp(
        base_xp: int,
        streak_days: int,
        is_perfect_week: bool = False
    ) -> int:
        """
        Calculate bonus XP from streaks and achievements
        """
        bonus = 0

        # Streak bonus (1% per day, max 50%)
        streak_bonus = min(0.5, streak_days * 0.01)
        bonus += int(base_xp * streak_bonus)

        # Perfect week bonus
        if is_perfect_week:
            bonus += int(base_xp * 0.25)

        return bonus