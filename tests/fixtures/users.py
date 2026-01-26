"""
User fixtures and factories for testing.
"""

from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class UserFactory:
    """Factory for creating test user data."""

    email_domain: str = "example.com"
    password: str = "Test123!@#"

    def create(self, prefix: str = "testuser") -> Dict:
        """Create a unique test user data dict."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        return {
            "email": f"{prefix}_{timestamp}@{self.email_domain}",
            "username": f"{prefix}_{timestamp}",
            "password": self.password
        }


def create_test_user_data(
    prefix: str = "testuser",
    email_domain: str = "example.com",
    password: str = "Test123!@#"
) -> Dict:
    """
    Generate unique test user data.

    Args:
        prefix: Username prefix
        email_domain: Email domain to use
        password: User password

    Returns:
        Dict with email, username, and password
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    return {
        "email": f"{prefix}_{timestamp}@{email_domain}",
        "username": f"{prefix}_{timestamp}",
        "password": password
    }


def create_mock_learner_profile(
    fsrs_stability: float = 2.5,
    fsrs_difficulty: float = 5.0,
    zpd_lower: float = 0.35,
    zpd_upper: float = 0.70,
    total_xp: int = 0,
    level: int = 1,
    streak_days: int = 0
) -> Dict:
    """
    Generate mock learner profile data.

    Args:
        fsrs_stability: FSRS stability parameter
        fsrs_difficulty: FSRS difficulty parameter
        zpd_lower: ZPD lower bound
        zpd_upper: ZPD upper bound
        total_xp: Total XP earned
        level: Current level
        streak_days: Current streak

    Returns:
        Dict with learner profile data
    """
    return {
        "fsrs_stability": fsrs_stability,
        "fsrs_difficulty": fsrs_difficulty,
        "current_zpd_lower": zpd_lower,
        "current_zpd_upper": zpd_upper,
        "total_xp": total_xp,
        "level": level,
        "streak_days": streak_days
    }
