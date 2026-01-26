"""
Integration tests for API Gateway <-> Scheduler service communication.

These tests verify that the API Gateway correctly communicates with the
Scheduler service for FSRS scheduling operations.
"""

import pytest
import httpx


# ============================================================================
# API <-> Scheduler Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Placeholder - implement when services are ready")
async def test_api_forwards_review_to_scheduler(
    api_client,
    scheduler_client,
    ensure_api_healthy,
    ensure_scheduler_healthy
):
    """Test that API Gateway correctly forwards review requests to Scheduler."""
    # TODO: Implement test
    pass


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Placeholder - implement when services are ready")
async def test_scheduler_returns_next_review_date(
    scheduler_client,
    ensure_scheduler_healthy
):
    """Test that Scheduler returns correct next review date."""
    # TODO: Implement test
    pass


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Placeholder - implement when services are ready")
async def test_api_handles_scheduler_timeout():
    """Test that API Gateway handles Scheduler timeout gracefully."""
    # TODO: Implement test
    pass


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skip(reason="Placeholder - implement when services are ready")
async def test_api_handles_scheduler_error():
    """Test that API Gateway handles Scheduler errors gracefully."""
    # TODO: Implement test
    pass
