"""
Tests for application configuration
"""
import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Tests for Settings configuration class"""

    def test_default_values(self):
        """Test default configuration values"""
        from app.core.config import Settings

        # Create settings with defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.APP_NAME == "NerdLearn API"
            assert settings.APP_VERSION == "1.0.0"
            assert settings.DEBUG is False
            assert settings.ENVIRONMENT == "development"

    def test_vector_settings(self):
        """Test vector embedding settings"""
        from app.core.config import Settings

        settings = Settings()

        assert settings.VECTOR_SIZE == 1536
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"

    def test_jwt_settings(self):
        """Test JWT configuration"""
        from app.core.config import Settings

        settings = Settings()

        assert settings.ALGORITHM == "HS256"
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert "secret" in settings.SECRET_KEY.lower() or len(settings.SECRET_KEY) > 0

    def test_cors_origins(self):
        """Test CORS allowed origins"""
        from app.core.config import Settings

        settings = Settings()

        assert isinstance(settings.ALLOWED_ORIGINS, list)
        assert "http://localhost:3000" in settings.ALLOWED_ORIGINS

    def test_rate_limiting_defaults(self):
        """Test rate limiting defaults"""
        from app.core.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            assert settings.RATE_LIMIT_ENABLED is False
            assert settings.RATE_LIMIT_PER_MINUTE == 60

    def test_environment_override(self):
        """Test environment variable overrides"""
        from app.core.config import Settings

        with patch.dict(os.environ, {"DEBUG": "true", "ENVIRONMENT": "production"}):
            settings = Settings()

            assert settings.DEBUG is True
            assert settings.ENVIRONMENT == "production"

    def test_database_url_default(self):
        """Test database URL default"""
        from app.core.config import Settings

        settings = Settings()

        assert "postgresql" in settings.DATABASE_URL
        assert "asyncpg" in settings.DATABASE_URL

    def test_redis_url_default(self):
        """Test Redis URL default"""
        from app.core.config import Settings

        settings = Settings()

        assert "redis://" in settings.REDIS_URL

    def test_minio_defaults(self):
        """Test MinIO/S3 defaults"""
        from app.core.config import Settings

        settings = Settings()

        assert settings.MINIO_ENDPOINT == "localhost:9000"
        assert settings.MINIO_BUCKET == "nerdlearn"
        assert settings.MINIO_SECURE is False

    def test_log_level_default(self):
        """Test log level default"""
        from app.core.config import Settings

        settings = Settings()

        assert settings.LOG_LEVEL == "INFO"


class TestEnvironmentDetection:
    """Tests for environment detection"""

    def test_development_environment(self):
        """Test development environment detection"""
        from app.core.config import Settings

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.ENVIRONMENT == "development"

    def test_staging_environment(self):
        """Test staging environment detection"""
        from app.core.config import Settings

        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            settings = Settings()
            assert settings.ENVIRONMENT == "staging"

    def test_production_environment(self):
        """Test production environment detection"""
        from app.core.config import Settings

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.ENVIRONMENT == "production"


class TestSecuritySettings:
    """Tests for security-related settings"""

    def test_secret_key_not_default_warning(self):
        """Test that default secret key should be changed"""
        from app.core.config import Settings

        settings = Settings()

        # Default key should contain warning text
        if "change-this" in settings.SECRET_KEY.lower():
            # This is expected in dev but not production
            assert settings.ENVIRONMENT != "production" or os.getenv("SECRET_KEY")

    def test_api_keys_can_be_empty(self):
        """Test that API keys can be empty for local development"""
        from app.core.config import Settings

        settings = Settings()

        # API keys can be empty strings
        assert isinstance(settings.OPENAI_API_KEY, str)
        assert isinstance(settings.ELEVENLABS_API_KEY, str)

    def test_sentry_dsn_optional(self):
        """Test that Sentry DSN is optional"""
        from app.core.config import Settings

        settings = Settings()

        assert isinstance(settings.SENTRY_DSN, str)


class TestConfigurationValidation:
    """Tests for configuration validation"""

    def test_vector_size_positive(self):
        """Test vector size is positive"""
        from app.core.config import Settings

        settings = Settings()
        assert settings.VECTOR_SIZE > 0

    def test_token_expiry_positive(self):
        """Test token expiry is positive"""
        from app.core.config import Settings

        settings = Settings()
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0

    def test_rate_limit_positive(self):
        """Test rate limit is positive"""
        from app.core.config import Settings

        settings = Settings()
        assert settings.RATE_LIMIT_PER_MINUTE > 0

    def test_allowed_origins_not_empty(self):
        """Test allowed origins list is not empty"""
        from app.core.config import Settings

        settings = Settings()
        assert len(settings.ALLOWED_ORIGINS) > 0


class TestSingletonSettings:
    """Tests for settings singleton behavior"""

    def test_settings_import(self):
        """Test settings can be imported"""
        from app.core.config import settings

        assert settings is not None
        assert settings.APP_NAME == "NerdLearn API"

    def test_settings_consistency(self):
        """Test settings are consistent across imports"""
        from app.core.config import settings as settings1
        from app.core.config import settings as settings2

        assert settings1.APP_NAME == settings2.APP_NAME
        assert settings1.VECTOR_SIZE == settings2.VECTOR_SIZE
