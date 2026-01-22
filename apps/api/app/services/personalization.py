"""
Personalization Settings Service

Manages user preferences and personalization including:
- Learning preferences (pace, style, difficulty)
- UI preferences (themes, layouts, accessibility)
- Notification preferences
- Privacy settings
- Content filtering

Features:
- Default settings with inheritance
- Setting validation
- Change history tracking
- Preference sync across devices
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from copy import deepcopy

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class Theme(str, Enum):
    """UI themes"""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    HIGH_CONTRAST = "high_contrast"


class LearningPace(str, Enum):
    """Learning pace preferences"""
    RELAXED = "relaxed"       # Longer intervals, less pressure
    STANDARD = "standard"     # Default pacing
    INTENSIVE = "intensive"   # Shorter intervals, more practice


class ContentDensity(str, Enum):
    """Content density preference"""
    COMPACT = "compact"
    COMFORTABLE = "comfortable"
    SPACIOUS = "spacious"


class DifficultyPreference(str, Enum):
    """Difficulty preference"""
    EASIER = "easier"        # Prefer easier content
    ADAPTIVE = "adaptive"    # System decides
    CHALLENGING = "challenging"  # Prefer harder content


class NotificationFrequency(str, Enum):
    """Notification frequency"""
    OFF = "off"
    MINIMAL = "minimal"
    STANDARD = "standard"
    FREQUENT = "frequent"


class AccessibilityFeature(str, Enum):
    """Accessibility features"""
    SCREEN_READER = "screen_reader"
    REDUCED_MOTION = "reduced_motion"
    LARGE_TEXT = "large_text"
    HIGH_CONTRAST = "high_contrast"
    DYSLEXIA_FONT = "dyslexia_font"
    KEYBOARD_NAVIGATION = "keyboard_navigation"
    CAPTIONS = "captions"


# ==================== Data Classes ====================

@dataclass
class LearningPreferences:
    """Learning-related preferences"""
    pace: LearningPace = LearningPace.STANDARD
    difficulty_preference: DifficultyPreference = DifficultyPreference.ADAPTIVE
    daily_goal_minutes: int = 30
    preferred_session_length: int = 20  # minutes
    review_reminder_enabled: bool = True
    practice_mode: str = "mixed"  # "flashcards", "quizzes", "mixed"
    show_hints: bool = True
    auto_advance: bool = False
    focus_mode: bool = False  # Hide distractions
    preferred_content_types: List[str] = field(default_factory=lambda: ["video", "text", "interactive"])
    excluded_topics: List[str] = field(default_factory=list)
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pace": self.pace.value,
            "difficulty_preference": self.difficulty_preference.value,
            "daily_goal_minutes": self.daily_goal_minutes,
            "preferred_session_length": self.preferred_session_length,
            "review_reminder_enabled": self.review_reminder_enabled,
            "practice_mode": self.practice_mode,
            "show_hints": self.show_hints,
            "auto_advance": self.auto_advance,
            "focus_mode": self.focus_mode,
            "preferred_content_types": self.preferred_content_types,
            "excluded_topics": self.excluded_topics,
            "language": self.language
        }


@dataclass
class UIPreferences:
    """User interface preferences"""
    theme: Theme = Theme.SYSTEM
    content_density: ContentDensity = ContentDensity.COMFORTABLE
    font_size: str = "medium"  # "small", "medium", "large", "xlarge"
    sidebar_collapsed: bool = False
    show_progress_bar: bool = True
    show_streaks: bool = True
    show_leaderboard: bool = True
    animation_enabled: bool = True
    sound_enabled: bool = True
    haptic_feedback: bool = True
    primary_color: str = "#3b82f6"  # Brand blue
    accessibility_features: List[AccessibilityFeature] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme.value,
            "content_density": self.content_density.value,
            "font_size": self.font_size,
            "sidebar_collapsed": self.sidebar_collapsed,
            "show_progress_bar": self.show_progress_bar,
            "show_streaks": self.show_streaks,
            "show_leaderboard": self.show_leaderboard,
            "animation_enabled": self.animation_enabled,
            "sound_enabled": self.sound_enabled,
            "haptic_feedback": self.haptic_feedback,
            "primary_color": self.primary_color,
            "accessibility_features": [f.value for f in self.accessibility_features]
        }


@dataclass
class NotificationPreferences:
    """Notification preferences"""
    email_frequency: NotificationFrequency = NotificationFrequency.STANDARD
    push_enabled: bool = True
    review_reminders: bool = True
    streak_reminders: bool = True
    achievement_notifications: bool = True
    social_notifications: bool = True
    marketing_emails: bool = False
    weekly_summary: bool = True
    quiet_hours_start: Optional[int] = 22  # 10 PM
    quiet_hours_end: Optional[int] = 8     # 8 AM
    preferred_reminder_time: str = "19:00"  # 7 PM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_frequency": self.email_frequency.value,
            "push_enabled": self.push_enabled,
            "review_reminders": self.review_reminders,
            "streak_reminders": self.streak_reminders,
            "achievement_notifications": self.achievement_notifications,
            "social_notifications": self.social_notifications,
            "marketing_emails": self.marketing_emails,
            "weekly_summary": self.weekly_summary,
            "quiet_hours_start": self.quiet_hours_start,
            "quiet_hours_end": self.quiet_hours_end,
            "preferred_reminder_time": self.preferred_reminder_time
        }


@dataclass
class PrivacyPreferences:
    """Privacy and data preferences"""
    profile_visibility: str = "friends"  # "public", "friends", "private"
    show_activity_status: bool = True
    show_in_leaderboards: bool = True
    allow_friend_requests: bool = True
    share_progress_with_instructors: bool = True
    analytics_enabled: bool = True
    personalization_enabled: bool = True
    data_retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_visibility": self.profile_visibility,
            "show_activity_status": self.show_activity_status,
            "show_in_leaderboards": self.show_in_leaderboards,
            "allow_friend_requests": self.allow_friend_requests,
            "share_progress_with_instructors": self.share_progress_with_instructors,
            "analytics_enabled": self.analytics_enabled,
            "personalization_enabled": self.personalization_enabled,
            "data_retention_days": self.data_retention_days
        }


@dataclass
class UserPreferences:
    """Complete user preferences"""
    user_id: str
    learning: LearningPreferences = field(default_factory=LearningPreferences)
    ui: UIPreferences = field(default_factory=UIPreferences)
    notifications: NotificationPreferences = field(default_factory=NotificationPreferences)
    privacy: PrivacyPreferences = field(default_factory=PrivacyPreferences)
    custom: Dict[str, Any] = field(default_factory=dict)  # App-specific custom settings
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "learning": self.learning.to_dict(),
            "ui": self.ui.to_dict(),
            "notifications": self.notifications.to_dict(),
            "privacy": self.privacy.to_dict(),
            "custom": self.custom,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class PreferenceChange:
    """Record of a preference change"""
    user_id: str
    category: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "user"  # "user", "system", "default"


# ==================== Personalization Service ====================

class PersonalizationService:
    """
    Manages user preferences and personalization settings.
    """

    # Validation rules for preferences
    VALIDATION_RULES = {
        "learning.daily_goal_minutes": {"min": 5, "max": 480},
        "learning.preferred_session_length": {"min": 5, "max": 120},
        "notifications.quiet_hours_start": {"min": 0, "max": 23},
        "notifications.quiet_hours_end": {"min": 0, "max": 23},
        "privacy.data_retention_days": {"min": 30, "max": 3650},
    }

    def __init__(self):
        self._preferences: Dict[str, UserPreferences] = {}
        self._change_history: Dict[str, List[PreferenceChange]] = {}
        self._defaults = UserPreferences(user_id="__default__")

    def get_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences (creates defaults if not exists).

        Args:
            user_id: User identifier

        Returns:
            User preferences
        """
        if user_id not in self._preferences:
            self._preferences[user_id] = UserPreferences(user_id=user_id)

        return self._preferences[user_id]

    def update_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any],
        source: str = "user"
    ) -> Dict[str, Any]:
        """
        Update user preferences.

        Args:
            user_id: User identifier
            updates: Dict of updates (can be nested: {"learning.pace": "intensive"})
            source: Source of update

        Returns:
            Updated preferences and any validation errors
        """
        prefs = self.get_preferences(user_id)
        errors = []
        changes = []

        for key, value in updates.items():
            # Validate
            error = self._validate_preference(key, value)
            if error:
                errors.append({"key": key, "error": error})
                continue

            # Get old value
            old_value = self._get_nested_value(prefs, key)

            # Set new value
            success = self._set_nested_value(prefs, key, value)

            if success and old_value != value:
                category = key.split(".")[0] if "." in key else "custom"
                changes.append(PreferenceChange(
                    user_id=user_id,
                    category=category,
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    source=source
                ))

        # Update timestamp
        prefs.updated_at = datetime.utcnow()

        # Record changes
        if changes:
            if user_id not in self._change_history:
                self._change_history[user_id] = []
            self._change_history[user_id].extend(changes)

        return {
            "preferences": prefs.to_dict(),
            "errors": errors,
            "changes_applied": len(changes)
        }

    def reset_preferences(
        self,
        user_id: str,
        category: Optional[str] = None
    ) -> UserPreferences:
        """
        Reset preferences to defaults.

        Args:
            user_id: User identifier
            category: Optional category to reset (None = all)

        Returns:
            Reset preferences
        """
        prefs = self.get_preferences(user_id)

        if category is None:
            # Reset all
            self._preferences[user_id] = UserPreferences(user_id=user_id)
        elif category == "learning":
            prefs.learning = LearningPreferences()
        elif category == "ui":
            prefs.ui = UIPreferences()
        elif category == "notifications":
            prefs.notifications = NotificationPreferences()
        elif category == "privacy":
            prefs.privacy = PrivacyPreferences()
        elif category == "custom":
            prefs.custom = {}

        prefs.updated_at = datetime.utcnow()
        return self._preferences[user_id]

    def get_change_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get preference change history"""
        history = self._change_history.get(user_id, [])
        return [
            {
                "category": c.category,
                "key": c.key,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "timestamp": c.timestamp.isoformat(),
                "source": c.source
            }
            for c in reversed(history[-limit:])
        ]

    def export_preferences(self, user_id: str) -> Dict[str, Any]:
        """Export preferences for backup/sync"""
        prefs = self.get_preferences(user_id)
        return {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "preferences": prefs.to_dict()
        }

    def import_preferences(
        self,
        user_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Import preferences from backup"""
        if "preferences" not in data:
            return {"error": "Invalid import data"}

        prefs_data = data["preferences"]
        updates = {}

        # Flatten nested structure
        for category in ["learning", "ui", "notifications", "privacy"]:
            if category in prefs_data:
                for key, value in prefs_data[category].items():
                    updates[f"{category}.{key}"] = value

        if "custom" in prefs_data:
            for key, value in prefs_data["custom"].items():
                updates[f"custom.{key}"] = value

        return self.update_preferences(user_id, updates, source="import")

    def _validate_preference(self, key: str, value: Any) -> Optional[str]:
        """Validate a preference value"""
        if key in self.VALIDATION_RULES:
            rules = self.VALIDATION_RULES[key]

            if "min" in rules and isinstance(value, (int, float)):
                if value < rules["min"]:
                    return f"Value must be at least {rules['min']}"

            if "max" in rules and isinstance(value, (int, float)):
                if value > rules["max"]:
                    return f"Value must be at most {rules['max']}"

        # Validate enums
        if key == "learning.pace" and value not in [p.value for p in LearningPace]:
            return f"Invalid pace. Must be one of: {[p.value for p in LearningPace]}"

        if key == "learning.difficulty_preference" and value not in [d.value for d in DifficultyPreference]:
            return f"Invalid difficulty preference"

        if key == "ui.theme" and value not in [t.value for t in Theme]:
            return f"Invalid theme. Must be one of: {[t.value for t in Theme]}"

        return None

    def _get_nested_value(self, obj: Any, key: str) -> Any:
        """Get value from nested object using dot notation"""
        parts = key.split(".")

        current = obj
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _set_nested_value(self, obj: Any, key: str, value: Any) -> bool:
        """Set value in nested object using dot notation"""
        parts = key.split(".")

        current = obj
        for i, part in enumerate(parts[:-1]):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            else:
                return False

        final_key = parts[-1]

        # Handle enum conversions
        if hasattr(current, final_key):
            attr = getattr(current, final_key)
            if isinstance(attr, Enum):
                # Convert string to enum
                enum_class = type(attr)
                try:
                    value = enum_class(value)
                except ValueError:
                    return False

            setattr(current, final_key, value)
            return True
        elif isinstance(current, dict):
            current[final_key] = value
            return True

        return False

    def get_effective_settings(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get effective settings considering context and A/B tests.

        Args:
            user_id: User identifier
            context: Optional context (device, time, etc.)

        Returns:
            Effective settings to use
        """
        prefs = self.get_preferences(user_id)
        effective = prefs.to_dict()

        context = context or {}

        # Apply time-based adjustments
        hour = datetime.utcnow().hour
        if prefs.notifications.quiet_hours_start and prefs.notifications.quiet_hours_end:
            if prefs.notifications.quiet_hours_start <= hour or hour < prefs.notifications.quiet_hours_end:
                effective["notifications"]["push_enabled"] = False

        # Apply device-based adjustments
        device = context.get("device", "desktop")
        if device == "mobile":
            effective["ui"]["sidebar_collapsed"] = True
            effective["ui"]["content_density"] = ContentDensity.COMPACT.value

        # Apply accessibility overrides
        if AccessibilityFeature.REDUCED_MOTION in prefs.ui.accessibility_features:
            effective["ui"]["animation_enabled"] = False

        if AccessibilityFeature.HIGH_CONTRAST in prefs.ui.accessibility_features:
            effective["ui"]["theme"] = Theme.HIGH_CONTRAST.value

        return effective


# Singleton instance
personalization_service = PersonalizationService()
