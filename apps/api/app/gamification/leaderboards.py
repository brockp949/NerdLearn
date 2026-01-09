"""
Social Features - Leaderboard System
Implements competitive and collaborative social features

Research basis:
- Octalysis Framework Core Drive #5: Social Influence & Relatedness
- SDT (Self-Determination Theory): Relatedness need
- Research shows leaderboards can increase engagement by 20-30%
- But also risk demotivating bottom performers

Key features:
- Multiple leaderboard types (global, friends, skill-based)
- Time-based resets (weekly, monthly, all-time)
- Percentile rankings to avoid demotivation
- Privacy controls for sensitive users
"""
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math


class LeaderboardType(str, Enum):
    """Types of leaderboards"""
    GLOBAL = "global"          # All users
    FRIENDS = "friends"        # User's connections
    COURSE = "course"          # Within a course
    SKILL = "skill"            # Specific skill/concept
    STREAK = "streak"          # Streak-based
    WEEKLY = "weekly"          # Weekly XP
    MONTHLY = "monthly"        # Monthly XP
    ALL_TIME = "all_time"      # All-time XP


class LeaderboardVisibility(str, Enum):
    """User's visibility preference"""
    PUBLIC = "public"          # Visible to all
    FRIENDS_ONLY = "friends"   # Only to friends
    ANONYMOUS = "anonymous"    # Show as "Anonymous Learner"
    HIDDEN = "hidden"          # Don't appear on leaderboards


@dataclass
class LeaderboardEntry:
    """Single entry in a leaderboard"""
    user_id: int
    display_name: str
    score: float
    rank: int
    percentile: float  # 0-100
    change_since_last: int  # Rank change (+/- positions)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Privacy
    is_anonymous: bool = False
    avatar_url: Optional[str] = None


@dataclass
class LeaderboardConfig:
    """Configuration for a leaderboard"""
    leaderboard_id: str
    leaderboard_type: LeaderboardType
    name: str
    description: str
    reset_period_days: Optional[int] = None  # None = never resets
    max_entries: int = 100
    min_activity_threshold: int = 1  # Minimum actions to appear
    show_percentile: bool = True
    allow_opt_out: bool = True


@dataclass
class UserSocialProfile:
    """User's social/leaderboard profile"""
    user_id: int
    display_name: str
    visibility: LeaderboardVisibility
    friends: List[int] = field(default_factory=list)
    blocked_users: List[int] = field(default_factory=list)

    # Stats
    total_xp: int = 0
    weekly_xp: int = 0
    monthly_xp: int = 0
    current_streak: int = 0
    longest_streak: int = 0

    # Per-course stats
    course_xp: Dict[int, int] = field(default_factory=dict)

    # Achievements shared
    shared_achievements: List[str] = field(default_factory=list)


class LeaderboardManager:
    """
    Manages all leaderboard functionality

    Features:
    - Multiple leaderboard types
    - Automatic ranking and percentile calculation
    - Privacy-respecting display
    - Demotivation prevention (percentile focus)
    """

    def __init__(self):
        self.users: Dict[int, UserSocialProfile] = {}
        self.leaderboards: Dict[str, LeaderboardConfig] = {}
        self.leaderboard_data: Dict[str, List[LeaderboardEntry]] = {}

        # Initialize default leaderboards
        self._init_default_leaderboards()

    def _init_default_leaderboards(self):
        """Initialize default leaderboard configurations"""
        defaults = [
            LeaderboardConfig(
                leaderboard_id="weekly_global",
                leaderboard_type=LeaderboardType.WEEKLY,
                name="Weekly Champions",
                description="Top learners this week",
                reset_period_days=7,
            ),
            LeaderboardConfig(
                leaderboard_id="monthly_global",
                leaderboard_type=LeaderboardType.MONTHLY,
                name="Monthly Masters",
                description="Top learners this month",
                reset_period_days=30,
            ),
            LeaderboardConfig(
                leaderboard_id="all_time",
                leaderboard_type=LeaderboardType.ALL_TIME,
                name="Hall of Fame",
                description="All-time top learners",
            ),
            LeaderboardConfig(
                leaderboard_id="streak_masters",
                leaderboard_type=LeaderboardType.STREAK,
                name="Streak Masters",
                description="Longest learning streaks",
            ),
        ]

        for config in defaults:
            self.leaderboards[config.leaderboard_id] = config
            self.leaderboard_data[config.leaderboard_id] = []

    def register_user(
        self,
        user_id: int,
        display_name: str,
        visibility: LeaderboardVisibility = LeaderboardVisibility.PUBLIC,
    ) -> UserSocialProfile:
        """Register or update user social profile"""
        if user_id in self.users:
            profile = self.users[user_id]
            profile.display_name = display_name
            profile.visibility = visibility
        else:
            profile = UserSocialProfile(
                user_id=user_id,
                display_name=display_name,
                visibility=visibility,
            )
            self.users[user_id] = profile

        return profile

    def update_user_stats(
        self,
        user_id: int,
        xp_earned: int = 0,
        course_id: Optional[int] = None,
        streak_days: Optional[int] = None,
    ):
        """Update user stats for leaderboards"""
        if user_id not in self.users:
            self.register_user(user_id, f"User_{user_id}")

        profile = self.users[user_id]

        # Update XP
        profile.total_xp += xp_earned
        profile.weekly_xp += xp_earned
        profile.monthly_xp += xp_earned

        if course_id:
            profile.course_xp[course_id] = profile.course_xp.get(course_id, 0) + xp_earned

        # Update streak
        if streak_days is not None:
            profile.current_streak = streak_days
            profile.longest_streak = max(profile.longest_streak, streak_days)

        # Rebuild affected leaderboards
        self._rebuild_leaderboards()

    def add_friend(self, user_id: int, friend_id: int):
        """Add a friend connection"""
        if user_id in self.users and friend_id in self.users:
            if friend_id not in self.users[user_id].friends:
                self.users[user_id].friends.append(friend_id)
            if user_id not in self.users[friend_id].friends:
                self.users[friend_id].friends.append(user_id)

    def remove_friend(self, user_id: int, friend_id: int):
        """Remove a friend connection"""
        if user_id in self.users and friend_id in self.users[user_id].friends:
            self.users[user_id].friends.remove(friend_id)
        if friend_id in self.users and user_id in self.users[friend_id].friends:
            self.users[friend_id].friends.remove(user_id)

    def get_leaderboard(
        self,
        leaderboard_id: str,
        requesting_user_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get leaderboard entries

        Args:
            leaderboard_id: Which leaderboard
            requesting_user_id: User requesting (for privacy filtering)
            limit: Max entries to return
            offset: Starting position

        Returns:
            Leaderboard data with entries
        """
        if leaderboard_id not in self.leaderboards:
            return {"error": "Leaderboard not found"}

        config = self.leaderboards[leaderboard_id]
        entries = self.leaderboard_data.get(leaderboard_id, [])

        # Filter for privacy
        filtered_entries = []
        for entry in entries:
            user = self.users.get(entry.user_id)
            if not user:
                continue

            # Check visibility
            if user.visibility == LeaderboardVisibility.HIDDEN:
                continue
            elif user.visibility == LeaderboardVisibility.FRIENDS_ONLY:
                if requesting_user_id and requesting_user_id in user.friends:
                    filtered_entries.append(entry)
                elif entry.user_id == requesting_user_id:
                    filtered_entries.append(entry)
            elif user.visibility == LeaderboardVisibility.ANONYMOUS:
                # Create anonymous entry
                anon_entry = LeaderboardEntry(
                    user_id=entry.user_id,
                    display_name="Anonymous Learner",
                    score=entry.score,
                    rank=entry.rank,
                    percentile=entry.percentile,
                    change_since_last=entry.change_since_last,
                    is_anonymous=True,
                )
                filtered_entries.append(anon_entry)
            else:
                filtered_entries.append(entry)

        # Apply pagination
        paginated = filtered_entries[offset:offset + limit]

        # Find requesting user's position
        user_position = None
        if requesting_user_id:
            for entry in entries:
                if entry.user_id == requesting_user_id:
                    user_position = {
                        "rank": entry.rank,
                        "percentile": entry.percentile,
                        "score": entry.score,
                    }
                    break

        return {
            "leaderboard_id": leaderboard_id,
            "name": config.name,
            "description": config.description,
            "type": config.leaderboard_type.value,
            "total_participants": len(entries),
            "entries": [
                {
                    "rank": e.rank,
                    "display_name": e.display_name,
                    "score": e.score,
                    "percentile": round(e.percentile, 1),
                    "change": e.change_since_last,
                    "is_anonymous": e.is_anonymous,
                    "avatar_url": e.avatar_url,
                    "is_you": e.user_id == requesting_user_id,
                }
                for e in paginated
            ],
            "your_position": user_position,
        }

    def get_friends_leaderboard(
        self,
        user_id: int,
        metric: str = "weekly_xp",
    ) -> Dict[str, Any]:
        """Get leaderboard among user's friends"""
        if user_id not in self.users:
            return {"error": "User not found"}

        user = self.users[user_id]
        friend_ids = user.friends + [user_id]  # Include self

        # Get scores for friends
        entries = []
        for fid in friend_ids:
            if fid not in self.users:
                continue

            friend = self.users[fid]

            # Get score based on metric
            if metric == "weekly_xp":
                score = friend.weekly_xp
            elif metric == "monthly_xp":
                score = friend.monthly_xp
            elif metric == "total_xp":
                score = friend.total_xp
            elif metric == "streak":
                score = friend.current_streak
            else:
                score = friend.total_xp

            entries.append({
                "user_id": fid,
                "display_name": friend.display_name,
                "score": score,
            })

        # Sort by score
        entries.sort(key=lambda x: x["score"], reverse=True)

        # Add ranks
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
            entry["is_you"] = entry["user_id"] == user_id

        return {
            "leaderboard_type": "friends",
            "metric": metric,
            "entries": entries,
            "your_rank": next(
                (e["rank"] for e in entries if e["user_id"] == user_id),
                None
            ),
        }

    def get_course_leaderboard(
        self,
        course_id: int,
        requesting_user_id: Optional[int] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get leaderboard for a specific course"""
        entries = []

        for user_id, profile in self.users.items():
            if course_id in profile.course_xp:
                if profile.visibility != LeaderboardVisibility.HIDDEN:
                    entries.append({
                        "user_id": user_id,
                        "display_name": profile.display_name if profile.visibility != LeaderboardVisibility.ANONYMOUS else "Anonymous",
                        "score": profile.course_xp[course_id],
                    })

        # Sort and rank
        entries.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
            entry["percentile"] = round((1 - i / max(1, len(entries))) * 100, 1)
            entry["is_you"] = entry["user_id"] == requesting_user_id

        return {
            "leaderboard_type": "course",
            "course_id": course_id,
            "total_participants": len(entries),
            "entries": entries[:limit],
            "your_position": next(
                ({"rank": e["rank"], "percentile": e["percentile"], "score": e["score"]}
                 for e in entries if e["user_id"] == requesting_user_id),
                None
            ),
        }

    def get_user_rankings(self, user_id: int) -> Dict[str, Any]:
        """Get user's rankings across all leaderboards"""
        if user_id not in self.users:
            return {"error": "User not found"}

        rankings = {}

        for lb_id, entries in self.leaderboard_data.items():
            for entry in entries:
                if entry.user_id == user_id:
                    rankings[lb_id] = {
                        "rank": entry.rank,
                        "percentile": round(entry.percentile, 1),
                        "score": entry.score,
                        "total_participants": len(entries),
                    }
                    break

        return {
            "user_id": user_id,
            "rankings": rankings,
        }

    def get_nearby_competitors(
        self,
        user_id: int,
        leaderboard_id: str,
        range_above: int = 3,
        range_below: int = 3,
    ) -> Dict[str, Any]:
        """Get competitors near user's rank (reduces demotivation)"""
        if leaderboard_id not in self.leaderboard_data:
            return {"error": "Leaderboard not found"}

        entries = self.leaderboard_data[leaderboard_id]

        # Find user's position
        user_idx = None
        for i, entry in enumerate(entries):
            if entry.user_id == user_id:
                user_idx = i
                break

        if user_idx is None:
            return {"error": "User not on leaderboard"}

        # Get nearby entries
        start = max(0, user_idx - range_above)
        end = min(len(entries), user_idx + range_below + 1)

        nearby = entries[start:end]

        return {
            "leaderboard_id": leaderboard_id,
            "your_rank": entries[user_idx].rank,
            "your_score": entries[user_idx].score,
            "nearby_competitors": [
                {
                    "rank": e.rank,
                    "display_name": e.display_name,
                    "score": e.score,
                    "gap_to_you": e.score - entries[user_idx].score,
                    "is_you": e.user_id == user_id,
                }
                for e in nearby
            ],
            "points_to_next_rank": (
                entries[user_idx - 1].score - entries[user_idx].score
                if user_idx > 0 else 0
            ),
        }

    def reset_periodic_leaderboards(self):
        """Reset time-based leaderboards (call via scheduled task)"""
        now = datetime.now()

        for lb_id, config in self.leaderboards.items():
            if config.reset_period_days:
                # Reset user stats for this period
                if config.leaderboard_type == LeaderboardType.WEEKLY:
                    for user in self.users.values():
                        user.weekly_xp = 0
                elif config.leaderboard_type == LeaderboardType.MONTHLY:
                    for user in self.users.values():
                        user.monthly_xp = 0

        self._rebuild_leaderboards()

    def _rebuild_leaderboards(self):
        """Rebuild all leaderboard rankings"""
        # Store previous ranks for change calculation
        previous_ranks: Dict[str, Dict[int, int]] = {}
        for lb_id, entries in self.leaderboard_data.items():
            previous_ranks[lb_id] = {e.user_id: e.rank for e in entries}

        # Rebuild each leaderboard
        for lb_id, config in self.leaderboards.items():
            entries = self._build_leaderboard_entries(config, previous_ranks.get(lb_id, {}))
            self.leaderboard_data[lb_id] = entries

    def _build_leaderboard_entries(
        self,
        config: LeaderboardConfig,
        previous_ranks: Dict[int, int],
    ) -> List[LeaderboardEntry]:
        """Build entries for a specific leaderboard"""
        entries = []

        for user_id, profile in self.users.items():
            # Skip hidden users
            if profile.visibility == LeaderboardVisibility.HIDDEN:
                continue

            # Get score based on leaderboard type
            if config.leaderboard_type == LeaderboardType.WEEKLY:
                score = profile.weekly_xp
            elif config.leaderboard_type == LeaderboardType.MONTHLY:
                score = profile.monthly_xp
            elif config.leaderboard_type == LeaderboardType.STREAK:
                score = profile.current_streak
            else:  # ALL_TIME, GLOBAL
                score = profile.total_xp

            # Check minimum threshold
            if score < config.min_activity_threshold:
                continue

            entries.append({
                "user_id": user_id,
                "display_name": profile.display_name,
                "score": score,
            })

        # Sort by score descending
        entries.sort(key=lambda x: x["score"], reverse=True)

        # Limit entries
        entries = entries[:config.max_entries]

        # Create LeaderboardEntry objects with ranks and percentiles
        total = len(entries)
        result = []

        for i, entry in enumerate(entries):
            rank = i + 1
            percentile = (1 - i / max(1, total)) * 100

            prev_rank = previous_ranks.get(entry["user_id"], rank)
            change = prev_rank - rank  # Positive = moved up

            result.append(LeaderboardEntry(
                user_id=entry["user_id"],
                display_name=entry["display_name"],
                score=entry["score"],
                rank=rank,
                percentile=percentile,
                change_since_last=change,
            ))

        return result


class SocialFeaturesManager:
    """
    High-level manager for all social features

    Combines leaderboards with other social elements
    """

    def __init__(self):
        self.leaderboard_manager = LeaderboardManager()

    def get_social_dashboard(
        self,
        user_id: int,
    ) -> Dict[str, Any]:
        """Get comprehensive social dashboard for user"""

        # Get all rankings
        rankings = self.leaderboard_manager.get_user_rankings(user_id)

        # Get friends leaderboard
        friends_lb = self.leaderboard_manager.get_friends_leaderboard(user_id)

        # Get nearby competitors for main leaderboard
        nearby = self.leaderboard_manager.get_nearby_competitors(
            user_id, "weekly_global"
        )

        # Get user profile
        profile = self.leaderboard_manager.users.get(user_id)

        return {
            "user_id": user_id,
            "profile": {
                "display_name": profile.display_name if profile else "Unknown",
                "total_xp": profile.total_xp if profile else 0,
                "current_streak": profile.current_streak if profile else 0,
                "friend_count": len(profile.friends) if profile else 0,
            },
            "rankings": rankings.get("rankings", {}),
            "friends_leaderboard": friends_lb,
            "nearby_competitors": nearby,
            "social_stats": {
                "percentile_weekly": rankings.get("rankings", {}).get("weekly_global", {}).get("percentile", 0),
                "rank_change_weekly": nearby.get("nearby_competitors", [{}])[0].get("change", 0) if nearby.get("nearby_competitors") else 0,
            },
        }

    def record_activity(
        self,
        user_id: int,
        xp_earned: int,
        course_id: Optional[int] = None,
        streak_days: Optional[int] = None,
    ):
        """Record user activity for leaderboards"""
        self.leaderboard_manager.update_user_stats(
            user_id=user_id,
            xp_earned=xp_earned,
            course_id=course_id,
            streak_days=streak_days,
        )
