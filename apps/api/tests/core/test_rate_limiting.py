import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.rate_limit import RateLimiter
from app.core.telemetry import TelemetryService, BudgetExceededError
from fastapi import HTTPException

class TestRateLimiting:
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test that requests under limit are allowed."""
        limiter = RateLimiter(requests_per_minute=5)
        
        # Mock Redis client and pipeline
        mock_pipe = AsyncMock()
        mock_pipe.incr = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[1, True])
        
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)
        
        limiter._client = mock_redis
        
        is_allowed, count = await limiter.check_rate_limit("test_key")
        
        assert is_allowed is True
        assert count == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter(requests_per_minute=5)
        
        # Mock Redis returning count > limit
        mock_pipe = AsyncMock()
        mock_pipe.incr = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[6, True])
        
        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=None)
        
        limiter._client = mock_redis
        
        is_allowed, count = await limiter.check_rate_limit("test_key")
        
        assert is_allowed is False
        assert count == 6

class TestCostCaps:
    
    def setup_method(self):
        TelemetryService().reset_metrics()
    
    def test_budget_enforcement(self):
        """Test that BudgetExceededError is raised when cost exceeds limit."""
        telemetry = TelemetryService()
        
        # Track calls up to just under the limit (default is $10)
        for _ in range(9):
            telemetry.track_llm_call("gpt-4o", 1000, 1.0)
        
        # This should succeed (total = $9)
        telemetry.track_llm_call("gpt-4o", 100, 0.5)
        
        # This should fail (would bring total to $10.6)
        with pytest.raises(BudgetExceededError):
            telemetry.track_llm_call("gpt-4o", 1000, 1.1)
    
    def test_budget_tracks_correctly(self):
        """Test that budget tracking is accurate."""
        telemetry = TelemetryService()
        
        telemetry.track_llm_call("gpt-4o", 100, 0.5)
        telemetry.track_llm_call("gpt-4o", 100, 0.3)
        
        metrics = telemetry.get_metrics()
        assert metrics["llm_cost_est_usd"] == 0.8
