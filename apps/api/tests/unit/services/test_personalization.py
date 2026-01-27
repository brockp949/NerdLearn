"""
Tests for PersonalizationService
"""
import pytest
from datetime import datetime


class TestPersonalizationService:
    """Tests for the PersonalizationService class"""

    @pytest.fixture
    def service(self):
        """Create PersonalizationService instance"""
        from app.services.personalization import PersonalizationService
        return PersonalizationService()

    def test_get_preferences_new_user(self, service):
        """Test getting preferences for new user creates defaults"""
        prefs = service.get_preferences("new_user_123")

        assert prefs.user_id == "new_user_123"
        # Check defaults
        assert prefs.learning.pace.value == "standard"
        assert prefs.learning.daily_goal_minutes == 30
        assert prefs.ui.theme.value == "system"
        assert prefs.notifications.push_enabled is True

    def test_get_preferences_existing_user(self, service):
        """Test getting preferences for existing user returns same object"""
        prefs1 = service.get_preferences("user_1")
        prefs2 = service.get_preferences("user_1")

        assert prefs1 is prefs2

    def test_update_preferences_simple(self, service):
        """Test updating a simple preference"""
        result = service.update_preferences(
            "user_1",
            {"learning.daily_goal_minutes": 45}
        )

        assert result["changes_applied"] == 1
        assert result["errors"] == []

        prefs = service.get_preferences("user_1")
        assert prefs.learning.daily_goal_minutes == 45

    def test_update_preferences_enum(self, service):
        """Test updating an enum preference"""
        result = service.update_preferences(
            "user_1",
            {"learning.pace": "intensive"}
        )

        assert result["changes_applied"] == 1
        prefs = service.get_preferences("user_1")
        assert prefs.learning.pace.value == "intensive"

    def test_update_preferences_invalid_value(self, service):
        """Test updating with invalid value"""
        result = service.update_preferences(
            "user_1",
            {"learning.pace": "invalid_pace"}
        )

        assert len(result["errors"]) == 1
        assert "pace" in result["errors"][0]["key"]

    def test_update_preferences_validation_min(self, service):
        """Test validation for minimum value"""
        result = service.update_preferences(
            "user_1",
            {"learning.daily_goal_minutes": 1}  # Min is 5
        )

        assert len(result["errors"]) == 1
        assert "at least" in result["errors"][0]["error"]

    def test_update_preferences_validation_max(self, service):
        """Test validation for maximum value"""
        result = service.update_preferences(
            "user_1",
            {"learning.daily_goal_minutes": 1000}  # Max is 480
        )

        assert len(result["errors"]) == 1
        assert "at most" in result["errors"][0]["error"]

    def test_update_multiple_preferences(self, service):
        """Test updating multiple preferences at once"""
        result = service.update_preferences(
            "user_1",
            {
                "learning.pace": "relaxed",
                "ui.theme": "dark",
                "notifications.push_enabled": False
            }
        )

        assert result["changes_applied"] == 3
        assert result["errors"] == []

        prefs = service.get_preferences("user_1")
        assert prefs.learning.pace.value == "relaxed"
        assert prefs.ui.theme.value == "dark"
        assert prefs.notifications.push_enabled is False

    def test_reset_preferences_all(self, service):
        """Test resetting all preferences"""
        # First modify some preferences
        service.update_preferences(
            "user_1",
            {"learning.daily_goal_minutes": 60}
        )

        # Reset all
        prefs = service.reset_preferences("user_1")

        assert prefs.learning.daily_goal_minutes == 30  # Default

    def test_reset_preferences_category(self, service):
        """Test resetting a specific category"""
        # Modify learning and UI preferences
        service.update_preferences(
            "user_1",
            {
                "learning.daily_goal_minutes": 60,
                "ui.theme": "dark"
            }
        )

        # Reset only learning
        prefs = service.reset_preferences("user_1", category="learning")

        assert prefs.learning.daily_goal_minutes == 30  # Reset
        assert prefs.ui.theme.value == "dark"  # Unchanged

    def test_change_history(self, service):
        """Test that changes are recorded in history"""
        service.update_preferences(
            "user_1",
            {"learning.daily_goal_minutes": 45}
        )

        history = service.get_change_history("user_1")

        assert len(history) >= 1
        latest = history[0]
        assert latest["key"] == "learning.daily_goal_minutes"
        assert latest["old_value"] == 30
        assert latest["new_value"] == 45

    def test_export_preferences(self, service):
        """Test exporting preferences"""
        service.update_preferences("user_1", {"ui.theme": "dark"})

        exported = service.export_preferences("user_1")

        assert "version" in exported
        assert "exported_at" in exported
        assert "preferences" in exported
        assert exported["preferences"]["ui"]["theme"] == "dark"

    def test_import_preferences(self, service):
        """Test importing preferences"""
        import_data = {
            "version": "1.0",
            "preferences": {
                "learning": {
                    "pace": "intensive",
                    "daily_goal_minutes": 60
                },
                "ui": {
                    "theme": "dark"
                }
            }
        }

        result = service.import_preferences("user_2", import_data)

        assert result["changes_applied"] >= 2
        prefs = service.get_preferences("user_2")
        assert prefs.learning.pace.value == "intensive"
        assert prefs.ui.theme.value == "dark"

    def test_import_invalid_data(self, service):
        """Test importing invalid data"""
        result = service.import_preferences("user_1", {"invalid": "data"})

        assert "error" in result

    def test_effective_settings_basic(self, service):
        """Test getting effective settings"""
        settings = service.get_effective_settings("user_1")

        assert "learning" in settings
        assert "ui" in settings
        assert "notifications" in settings
        assert "privacy" in settings

    def test_effective_settings_mobile_context(self, service):
        """Test effective settings with mobile context"""
        settings = service.get_effective_settings(
            "user_1",
            context={"device": "mobile"}
        )

        # Mobile should force compact density and collapsed sidebar
        assert settings["ui"]["sidebar_collapsed"] is True
        assert settings["ui"]["content_density"] == "compact"

    def test_effective_settings_accessibility(self, service):
        """Test effective settings with accessibility features"""
        from app.services.personalization import AccessibilityFeature

        # Enable reduced motion
        prefs = service.get_preferences("user_1")
        prefs.ui.accessibility_features.append(AccessibilityFeature.REDUCED_MOTION)

        settings = service.get_effective_settings("user_1")

        assert settings["ui"]["animation_enabled"] is False


class TestPreferenceModels:
    """Tests for preference data models"""

    def test_learning_preferences_to_dict(self):
        """Test LearningPreferences serialization"""
        from app.services.personalization import LearningPreferences

        prefs = LearningPreferences()
        data = prefs.to_dict()

        assert "pace" in data
        assert "daily_goal_minutes" in data
        assert data["pace"] == "standard"

    def test_ui_preferences_to_dict(self):
        """Test UIPreferences serialization"""
        from app.services.personalization import UIPreferences

        prefs = UIPreferences()
        data = prefs.to_dict()

        assert "theme" in data
        assert "font_size" in data
        assert data["theme"] == "system"

    def test_notification_preferences_to_dict(self):
        """Test NotificationPreferences serialization"""
        from app.services.personalization import NotificationPreferences

        prefs = NotificationPreferences()
        data = prefs.to_dict()

        assert "push_enabled" in data
        assert "quiet_hours_start" in data

    def test_privacy_preferences_to_dict(self):
        """Test PrivacyPreferences serialization"""
        from app.services.personalization import PrivacyPreferences

        prefs = PrivacyPreferences()
        data = prefs.to_dict()

        assert "profile_visibility" in data
        assert "analytics_enabled" in data


class TestEnums:
    """Tests for preference enums"""

    def test_theme_values(self):
        """Test Theme enum values"""
        from app.services.personalization import Theme

        assert Theme.LIGHT.value == "light"
        assert Theme.DARK.value == "dark"
        assert Theme.SYSTEM.value == "system"
        assert Theme.HIGH_CONTRAST.value == "high_contrast"

    def test_learning_pace_values(self):
        """Test LearningPace enum values"""
        from app.services.personalization import LearningPace

        assert LearningPace.RELAXED.value == "relaxed"
        assert LearningPace.STANDARD.value == "standard"
        assert LearningPace.INTENSIVE.value == "intensive"

    def test_notification_frequency_values(self):
        """Test NotificationFrequency enum values"""
        from app.services.personalization import NotificationFrequency

        assert NotificationFrequency.OFF.value == "off"
        assert NotificationFrequency.MINIMAL.value == "minimal"
        assert NotificationFrequency.STANDARD.value == "standard"
        assert NotificationFrequency.FREQUENT.value == "frequent"
