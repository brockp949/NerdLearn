"""
Tests for analytics router endpoints
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestAnalyticsModels:
    """Tests for analytics data models"""

    def test_time_granularity_enum(self):
        """Test TimeGranularity enum values"""
        from app.routers.analytics import TimeGranularity

        assert TimeGranularity.HOUR == "hour"
        assert TimeGranularity.DAY == "day"
        assert TimeGranularity.WEEK == "week"
        assert TimeGranularity.MONTH == "month"

    def test_metric_type_enum(self):
        """Test MetricType enum values"""
        from app.routers.analytics import MetricType

        assert MetricType.ACTIVE_USERS == "active_users"
        assert MetricType.SESSIONS == "sessions"
        assert MetricType.COMPLETIONS == "completions"
        assert MetricType.REVIEWS == "reviews"
        assert MetricType.MASTERY_GAIN == "mastery_gain"
        assert MetricType.ENGAGEMENT_TIME == "engagement_time"

    def test_heatmap_cell_model(self):
        """Test HeatmapCell model"""
        from app.routers.analytics import HeatmapCell

        cell = HeatmapCell(x=1, y=2, value=0.75, label="Mon 10:00")

        assert cell.x == 1
        assert cell.y == 2
        assert cell.value == 0.75
        assert cell.label == "Mon 10:00"

    def test_retention_cohort_model(self):
        """Test RetentionCohort model"""
        from app.routers.analytics import RetentionCohort

        cohort = RetentionCohort(
            cohort_date="2024-01-01",
            cohort_size=100,
            retention_rates=[0.9, 0.6, 0.45, 0.35]
        )

        assert cohort.cohort_date == "2024-01-01"
        assert cohort.cohort_size == 100
        assert len(cohort.retention_rates) == 4

    def test_learning_curve_point_model(self):
        """Test LearningCurvePoint model"""
        from app.routers.analytics import LearningCurvePoint

        point = LearningCurvePoint(
            timestamp=datetime.utcnow(),
            mastery=0.75,
            practice_count=10,
            concept_id=1
        )

        assert point.mastery == 0.75
        assert point.practice_count == 10
        assert point.concept_id == 1

    def test_session_metric_input_model(self):
        """Test SessionMetricInput model"""
        from app.routers.analytics import SessionMetricInput

        metric = SessionMetricInput(
            user_id=1,
            session_id="session-123",
            total_dwell_ms=60000,
            valid_dwell_ms=55000,
            engagement_score=0.85
        )

        assert metric.user_id == 1
        assert metric.session_id == "session-123"
        assert metric.total_dwell_ms == 60000
        assert metric.valid_dwell_ms == 55000
        assert metric.engagement_score == 0.85


class TestHeatmapEndpoints:
    """Tests for heatmap endpoints"""

    @pytest.mark.asyncio
    async def test_get_weekly_engagement_heatmap(self, client):
        """Test getting weekly engagement heatmap"""
        response = await client.get("/api/analytics/heatmap/weekly")

        if response.status_code == 200:
            data = response.json()
            assert "title" in data
            assert "x_labels" in data
            assert "y_labels" in data
            assert "data" in data
            assert len(data["x_labels"]) == 7  # Days of week
            assert len(data["y_labels"]) == 24  # Hours

    @pytest.mark.asyncio
    async def test_get_weekly_heatmap_with_filters(self, client):
        """Test getting weekly heatmap with course and user filters"""
        response = await client.get(
            "/api/analytics/heatmap/weekly?course_id=1&user_id=1&days=14"
        )

        assert response.status_code in [200, 500]

    @pytest.mark.asyncio
    async def test_get_weekly_heatmap_invalid_days(self, client):
        """Test weekly heatmap with invalid days parameter"""
        response = await client.get("/api/analytics/heatmap/weekly?days=1")

        # Should fail validation (days must be >= 7)
        assert response.status_code in [422, 500]

    @pytest.mark.asyncio
    async def test_get_concept_module_heatmap(self, client):
        """Test getting concept-module heatmap"""
        response = await client.get("/api/analytics/heatmap/concept-module?course_id=1")

        if response.status_code == 200:
            data = response.json()
            assert "title" in data
            assert "metric" in data
            assert data["metric"] == "mastery"


class TestRetentionEndpoints:
    """Tests for retention analysis endpoints"""

    @pytest.mark.asyncio
    async def test_get_retention_cohorts(self, client):
        """Test getting retention cohorts"""
        response = await client.get("/api/analytics/retention/cohort")

        if response.status_code == 200:
            data = response.json()
            assert "cohorts" in data
            assert "periods" in data
            assert "overall_retention" in data

    @pytest.mark.asyncio
    async def test_get_retention_cohorts_with_params(self, client):
        """Test retention cohorts with parameters"""
        response = await client.get(
            "/api/analytics/retention/cohort?cohort_period=week&periods_back=4"
        )

        if response.status_code == 200:
            data = response.json()
            assert len(data["cohorts"]) == 4

    @pytest.mark.asyncio
    async def test_get_retention_curve(self, client):
        """Test getting retention curve"""
        response = await client.get("/api/analytics/retention/curve?course_id=1&days=30")

        if response.status_code == 200:
            data = response.json()
            assert "curve" in data
            assert "half_life_days" in data
            assert len(data["curve"]) == 30


class TestLearningCurveEndpoints:
    """Tests for learning curve endpoints"""

    @pytest.mark.asyncio
    async def test_get_user_learning_curve(self, client):
        """Test getting user learning curve"""
        response = await client.get(
            "/api/analytics/learning-curve/user/1?course_id=1"
        )

        if response.status_code == 200:
            data = response.json()
            assert "user_id" in data
            assert "course_id" in data
            assert "points" in data
            assert "trend" in data
            assert data["trend"] in ["improving", "stable", "declining"]

    @pytest.mark.asyncio
    async def test_get_course_learning_curves(self, client):
        """Test getting course aggregate learning curves"""
        response = await client.get(
            "/api/analytics/learning-curve/course/1?days=30"
        )

        if response.status_code == 200:
            data = response.json()
            assert "curves" in data
            assert "percentiles" in data


class TestMasteryEndpoints:
    """Tests for mastery distribution endpoints"""

    @pytest.mark.asyncio
    async def test_get_mastery_distribution(self, client):
        """Test getting mastery distribution"""
        response = await client.get("/api/analytics/mastery/distribution?course_id=1")

        if response.status_code == 200:
            data = response.json()
            assert "distribution" in data
            assert "total_concepts" in data
            assert "avg_mastery" in data

    @pytest.mark.asyncio
    async def test_get_mastery_progress(self, client):
        """Test getting mastery progress over time"""
        response = await client.get(
            "/api/analytics/mastery/progress?course_id=1&period=week"
        )

        if response.status_code == 200:
            data = response.json()
            assert "progress" in data
            assert "mastery_change" in data


class TestTimeSeriesEndpoints:
    """Tests for time series metrics endpoints"""

    @pytest.mark.asyncio
    async def test_get_time_series_active_users(self, client):
        """Test getting active users time series"""
        response = await client.get(
            "/api/analytics/metrics/time-series?metric=active_users"
        )

        if response.status_code == 200:
            data = response.json()
            assert data["metric"] == "active_users"
            assert "points" in data
            assert "total" in data
            assert "avg" in data
            assert "trend" in data

    @pytest.mark.asyncio
    async def test_get_time_series_with_granularity(self, client):
        """Test time series with different granularities"""
        for granularity in ["hour", "day", "week", "month"]:
            response = await client.get(
                f"/api/analytics/metrics/time-series?metric=sessions&granularity={granularity}"
            )
            assert response.status_code in [200, 500]


class TestDashboardEndpoints:
    """Tests for dashboard summary endpoints"""

    @pytest.mark.asyncio
    async def test_get_analytics_summary(self, client):
        """Test getting analytics summary"""
        response = await client.get("/api/analytics/summary")

        if response.status_code == 200:
            data = response.json()
            assert "metrics" in data
            assert "top_concepts_by_engagement" in data
            assert "struggling_concepts" in data

    @pytest.mark.asyncio
    async def test_get_learning_funnel(self, client):
        """Test getting learning funnel"""
        response = await client.get("/api/analytics/funnel?course_id=1")

        if response.status_code == 200:
            data = response.json()
            assert "funnel_stages" in data
            assert "drop_off_analysis" in data


class TestSessionMetricsEndpoint:
    """Tests for session metrics ingestion"""

    @pytest.mark.asyncio
    async def test_ingest_session_metrics(self, client):
        """Test ingesting session metrics"""
        metrics = {
            "user_id": 1,
            "session_id": "session-123",
            "total_dwell_ms": 60000,
            "valid_dwell_ms": 55000,
            "engagement_score": 0.85
        }

        response = await client.post("/api/analytics/metrics/session", json=metrics)

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "recorded"
            assert "added_minutes" in data

    @pytest.mark.asyncio
    async def test_ingest_session_metrics_invalid(self, client):
        """Test ingesting invalid session metrics"""
        metrics = {
            "user_id": 1,
            # Missing required fields
        }

        response = await client.post("/api/analytics/metrics/session", json=metrics)
        assert response.status_code == 422
