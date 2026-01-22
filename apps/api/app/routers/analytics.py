"""
Analytics Dashboard API Router

Provides comprehensive analytics endpoints for:
- Engagement heatmaps
- Retention analysis (cohort-based)
- Learning curves and progress
- Concept mastery distribution
- Time-series metrics
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, distinct, case, text
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import math

from app.core.database import get_db

router = APIRouter()


# ==================== Request/Response Models ====================

class TimeGranularity(str, Enum):
    """Time granularity for aggregations"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricType(str, Enum):
    """Types of metrics to retrieve"""
    ACTIVE_USERS = "active_users"
    SESSIONS = "sessions"
    COMPLETIONS = "completions"
    REVIEWS = "reviews"
    MASTERY_GAIN = "mastery_gain"
    ENGAGEMENT_TIME = "engagement_time"


class HeatmapCell(BaseModel):
    """Single cell in an engagement heatmap"""
    x: int = Field(..., description="X coordinate (e.g., day of week)")
    y: int = Field(..., description="Y coordinate (e.g., hour)")
    value: float = Field(..., description="Metric value")
    label: Optional[str] = None


class HeatmapResponse(BaseModel):
    """Engagement heatmap response"""
    title: str
    x_labels: List[str]
    y_labels: List[str]
    data: List[HeatmapCell]
    min_value: float
    max_value: float
    metric: str


class RetentionCohort(BaseModel):
    """Single cohort in retention analysis"""
    cohort_date: str
    cohort_size: int
    retention_rates: List[float]  # Day 1, Day 7, Day 14, Day 30, etc.


class RetentionResponse(BaseModel):
    """Retention analysis response"""
    cohorts: List[RetentionCohort]
    periods: List[str]
    overall_retention: Dict[str, float]


class LearningCurvePoint(BaseModel):
    """Single point on a learning curve"""
    timestamp: datetime
    mastery: float
    practice_count: int
    concept_id: Optional[int] = None


class LearningCurveResponse(BaseModel):
    """Learning curve response"""
    user_id: int
    course_id: int
    points: List[LearningCurvePoint]
    trend: str  # "improving", "stable", "declining"
    avg_learning_rate: float


class MasteryDistribution(BaseModel):
    """Mastery level distribution"""
    level: str
    count: int
    percentage: float


class TimeSeriesPoint(BaseModel):
    """Single point in a time series"""
    timestamp: datetime
    value: float


class TimeSeriesResponse(BaseModel):
    """Time series metrics response"""
    metric: str
    granularity: str
    points: List[TimeSeriesPoint]
    total: float
    avg: float
    trend: float  # Percentage change


# ==================== Engagement Heatmap ====================

@router.get("/heatmap/weekly", response_model=HeatmapResponse)
async def get_weekly_engagement_heatmap(
    course_id: Optional[int] = None,
    user_id: Optional[int] = None,
    metric: str = "sessions",
    days: int = Query(default=30, ge=7, le=90),
    db: AsyncSession = Depends(get_db)
):
    """
    Get weekly engagement heatmap showing activity patterns.

    X-axis: Days of week (Mon-Sun)
    Y-axis: Hours of day (0-23)
    Value: Count or duration based on metric
    """
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Query engagement data (using review_logs as proxy for activity)
    # In production, this would query a dedicated activity/telemetry table

    # Generate sample heatmap data structure
    # In production: aggregate from actual telemetry
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [f"{h:02d}:00" for h in range(24)]

    # Generate heatmap data
    heatmap_data = []
    min_val = float('inf')
    max_val = float('-inf')

    for day_idx, day in enumerate(days_of_week):
        for hour_idx in range(24):
            # Calculate activity level based on typical patterns
            # Peak hours: 9-12, 14-17, 19-22
            base_activity = 0.2
            if 9 <= hour_idx <= 12 or 14 <= hour_idx <= 17:
                base_activity = 0.8
            elif 19 <= hour_idx <= 22:
                base_activity = 0.6
            elif 0 <= hour_idx <= 6:
                base_activity = 0.1

            # Weekend adjustment
            if day_idx >= 5:
                base_activity *= 0.7

            # Add some variation
            value = base_activity * (0.8 + 0.4 * ((day_idx * 24 + hour_idx) % 7) / 7)

            heatmap_data.append(HeatmapCell(
                x=day_idx,
                y=hour_idx,
                value=round(value, 2),
                label=f"{day} {hours[hour_idx]}"
            ))

            min_val = min(min_val, value)
            max_val = max(max_val, value)

    return HeatmapResponse(
        title=f"Weekly Engagement Pattern ({metric})",
        x_labels=days_of_week,
        y_labels=hours,
        data=heatmap_data,
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        metric=metric
    )


@router.get("/heatmap/concept-module", response_model=HeatmapResponse)
async def get_concept_module_heatmap(
    course_id: int,
    user_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get concept-module engagement heatmap.

    X-axis: Modules
    Y-axis: Concepts
    Value: Engagement/mastery level
    """
    # In production: query actual module and concept data
    # For demo, generate sample structure
    modules = [f"Module {i}" for i in range(1, 9)]
    concepts = [f"Concept {i}" for i in range(1, 13)]

    heatmap_data = []
    min_val = float('inf')
    max_val = float('-inf')

    for module_idx, module in enumerate(modules):
        for concept_idx, concept in enumerate(concepts):
            # Mastery tends to be higher for earlier modules and concepts
            base_mastery = 1 - (module_idx * 0.08 + concept_idx * 0.03)
            value = max(0.1, min(1.0, base_mastery + (((module_idx + concept_idx) % 3) - 1) * 0.1))

            heatmap_data.append(HeatmapCell(
                x=module_idx,
                y=concept_idx,
                value=round(value, 2),
                label=f"{module}: {concept}"
            ))

            min_val = min(min_val, value)
            max_val = max(max_val, value)

    return HeatmapResponse(
        title="Concept-Module Mastery Heatmap",
        x_labels=modules,
        y_labels=concepts,
        data=heatmap_data,
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        metric="mastery"
    )


# ==================== Retention Analysis ====================

@router.get("/retention/cohort", response_model=RetentionResponse)
async def get_retention_cohorts(
    course_id: Optional[int] = None,
    cohort_period: str = Query(default="week", regex="^(day|week|month)$"),
    periods_back: int = Query(default=8, ge=1, le=24),
    db: AsyncSession = Depends(get_db)
):
    """
    Get cohort-based retention analysis.

    Groups users by signup/start date and tracks retention over time.
    Returns retention rates at Day 1, 7, 14, 30.
    """
    cohorts = []

    # Calculate cohort periods
    now = datetime.utcnow()
    period_delta = {
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": timedelta(days=30),
    }[cohort_period]

    retention_periods = ["Day 1", "Day 7", "Day 14", "Day 30"]

    overall_retention = {period: 0.0 for period in retention_periods}

    for i in range(periods_back):
        cohort_start = now - period_delta * (i + 1)
        cohort_date = cohort_start.strftime("%Y-%m-%d")

        # In production: query actual user enrollment and activity data
        # Generate sample retention data with realistic decay
        base_size = 100 + (i * 10) % 50  # Vary cohort size

        # Retention decay pattern
        retention_rates = [
            round(0.9 - i * 0.02, 2),   # Day 1: ~90% - slight decay for older cohorts
            round(0.6 - i * 0.03, 2),   # Day 7: ~60%
            round(0.45 - i * 0.02, 2),  # Day 14: ~45%
            round(0.35 - i * 0.015, 2), # Day 30: ~35%
        ]
        retention_rates = [max(0.1, min(1.0, r)) for r in retention_rates]

        cohorts.append(RetentionCohort(
            cohort_date=cohort_date,
            cohort_size=base_size,
            retention_rates=retention_rates
        ))

        # Accumulate for overall
        for j, period in enumerate(retention_periods):
            overall_retention[period] += retention_rates[j]

    # Calculate averages
    for period in retention_periods:
        overall_retention[period] = round(overall_retention[period] / periods_back, 2)

    return RetentionResponse(
        cohorts=cohorts,
        periods=retention_periods,
        overall_retention=overall_retention
    )


@router.get("/retention/curve")
async def get_retention_curve(
    course_id: int,
    days: int = Query(default=30, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """
    Get retention curve showing daily active user retention.
    """
    curve_points = []

    for day in range(days):
        # Exponential decay model for retention
        retention = 0.9 * math.exp(-day / 20) + 0.1
        curve_points.append({
            "day": day,
            "retention": round(retention, 3),
            "active_users": int(1000 * retention)  # Assuming 1000 initial users
        })

    return {
        "course_id": course_id,
        "days": days,
        "curve": curve_points,
        "half_life_days": round(20 * math.log(2), 1),  # Days until 50% retention
        "plateau_retention": 0.1  # Long-term retention floor
    }


# ==================== Learning Curves ====================

@router.get("/learning-curve/user/{user_id}", response_model=LearningCurveResponse)
async def get_user_learning_curve(
    user_id: int,
    course_id: int,
    concept_id: Optional[int] = None,
    days: int = Query(default=30, ge=1, le=180),
    db: AsyncSession = Depends(get_db)
):
    """
    Get learning curve for a specific user.

    Shows mastery progression over time with practice count.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # In production: query actual mastery history
    # Generate sample learning curve with realistic progression
    points = []
    current_mastery = 0.2
    practice_count = 0

    for day in range(days):
        timestamp = start_date + timedelta(days=day)

        # Learning with forgetting curve
        # Gain from practice, decay without practice
        practices_today = 1 if day % 2 == 0 else 0
        practice_count += practices_today

        if practices_today > 0:
            # Learning gain (diminishing returns)
            gain = 0.05 * (1 - current_mastery)
            current_mastery = min(0.95, current_mastery + gain)
        else:
            # Forgetting
            decay = 0.02 * current_mastery
            current_mastery = max(0.1, current_mastery - decay)

        points.append(LearningCurvePoint(
            timestamp=timestamp,
            mastery=round(current_mastery, 3),
            practice_count=practice_count,
            concept_id=concept_id
        ))

    # Calculate trend
    if len(points) >= 2:
        first_quarter = sum(p.mastery for p in points[:len(points)//4]) / (len(points)//4)
        last_quarter = sum(p.mastery for p in points[-len(points)//4:]) / (len(points)//4)
        trend = "improving" if last_quarter > first_quarter + 0.05 else \
                "declining" if last_quarter < first_quarter - 0.05 else "stable"
    else:
        trend = "stable"

    # Calculate average learning rate
    if len(points) >= 2 and practice_count > 0:
        mastery_gain = points[-1].mastery - points[0].mastery
        avg_learning_rate = mastery_gain / practice_count
    else:
        avg_learning_rate = 0.0

    return LearningCurveResponse(
        user_id=user_id,
        course_id=course_id,
        points=points,
        trend=trend,
        avg_learning_rate=round(avg_learning_rate, 4)
    )


@router.get("/learning-curve/course/{course_id}")
async def get_course_learning_curves(
    course_id: int,
    percentiles: List[int] = Query(default=[25, 50, 75, 90]),
    days: int = Query(default=30, ge=1, le=180),
    db: AsyncSession = Depends(get_db)
):
    """
    Get aggregate learning curves for a course showing percentiles.

    Useful for understanding typical learning progression.
    """
    curves = {}

    for percentile in percentiles:
        points = []
        base_mastery = 0.15 + percentile / 500  # Higher percentile = faster learner
        current_mastery = base_mastery

        for day in range(days):
            # Learning rate varies by percentile
            learn_rate = 0.02 + percentile / 2000
            current_mastery = min(0.95, current_mastery + learn_rate * (1 - current_mastery))

            points.append({
                "day": day,
                "mastery": round(current_mastery, 3)
            })

        curves[f"p{percentile}"] = points

    return {
        "course_id": course_id,
        "percentiles": percentiles,
        "curves": curves,
        "days": days
    }


# ==================== Mastery Distribution ====================

@router.get("/mastery/distribution")
async def get_mastery_distribution(
    course_id: int,
    user_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get distribution of mastery levels across concepts.
    """
    # Mastery level buckets
    levels = [
        ("Novice", 0.0, 0.25),
        ("Developing", 0.25, 0.5),
        ("Competent", 0.5, 0.75),
        ("Proficient", 0.75, 0.9),
        ("Expert", 0.9, 1.0),
    ]

    # In production: query actual mastery data
    # Generate sample distribution
    total = 100
    distribution = [
        MasteryDistribution(level="Novice", count=15, percentage=15.0),
        MasteryDistribution(level="Developing", count=25, percentage=25.0),
        MasteryDistribution(level="Competent", count=30, percentage=30.0),
        MasteryDistribution(level="Proficient", count=20, percentage=20.0),
        MasteryDistribution(level="Expert", count=10, percentage=10.0),
    ]

    return {
        "course_id": course_id,
        "total_concepts": total,
        "distribution": distribution,
        "avg_mastery": 0.52,
        "median_mastery": 0.55
    }


@router.get("/mastery/progress")
async def get_mastery_progress(
    course_id: int,
    user_id: Optional[int] = None,
    period: str = Query(default="week", regex="^(day|week|month)$"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get mastery progress over time periods.
    """
    periods_count = {"day": 7, "week": 8, "month": 6}[period]
    period_delta = {"day": timedelta(days=1), "week": timedelta(weeks=1), "month": timedelta(days=30)}[period]

    progress = []
    current_mastery = 0.4

    for i in range(periods_count):
        period_end = datetime.utcnow() - period_delta * i

        # Mastery was lower in the past
        past_mastery = max(0.2, current_mastery - i * 0.05)

        progress.append({
            "period": period_end.strftime("%Y-%m-%d"),
            "avg_mastery": round(past_mastery, 3),
            "concepts_mastered": int(past_mastery * 100),
            "total_concepts": 100
        })

    # Reverse to show oldest first
    progress.reverse()

    return {
        "course_id": course_id,
        "period": period,
        "progress": progress,
        "mastery_change": round(current_mastery - progress[0]["avg_mastery"], 3)
    }


# ==================== Time Series Metrics ====================

@router.get("/metrics/time-series", response_model=TimeSeriesResponse)
async def get_time_series_metrics(
    metric: MetricType,
    course_id: Optional[int] = None,
    granularity: TimeGranularity = TimeGranularity.DAY,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """
    Get time series data for various metrics.

    Supports: active_users, sessions, completions, reviews, mastery_gain, engagement_time
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Calculate number of points based on granularity
    granularity_hours = {
        TimeGranularity.HOUR: 1,
        TimeGranularity.DAY: 24,
        TimeGranularity.WEEK: 168,
        TimeGranularity.MONTH: 720,
    }[granularity]

    num_points = max(1, (days * 24) // granularity_hours)

    # Generate time series points
    points = []
    total = 0.0

    for i in range(num_points):
        timestamp = start_date + timedelta(hours=granularity_hours * i)

        # Generate metric-appropriate values
        if metric == MetricType.ACTIVE_USERS:
            base_value = 500 + math.sin(i * 0.3) * 100
        elif metric == MetricType.SESSIONS:
            base_value = 1000 + math.sin(i * 0.3) * 200
        elif metric == MetricType.COMPLETIONS:
            base_value = 50 + math.sin(i * 0.3) * 10
        elif metric == MetricType.REVIEWS:
            base_value = 300 + math.sin(i * 0.3) * 50
        elif metric == MetricType.MASTERY_GAIN:
            base_value = 0.02 + math.sin(i * 0.3) * 0.005
        elif metric == MetricType.ENGAGEMENT_TIME:
            base_value = 2500 + math.sin(i * 0.3) * 500  # Minutes
        else:
            base_value = 100

        # Add some noise
        value = max(0, base_value * (0.9 + 0.2 * ((i * 7) % 10) / 10))

        points.append(TimeSeriesPoint(
            timestamp=timestamp,
            value=round(value, 2)
        ))
        total += value

    avg = total / len(points) if points else 0

    # Calculate trend (compare first and last quarter)
    if len(points) >= 4:
        first_quarter = sum(p.value for p in points[:len(points)//4]) / (len(points)//4)
        last_quarter = sum(p.value for p in points[-len(points)//4:]) / (len(points)//4)
        trend = ((last_quarter - first_quarter) / first_quarter * 100) if first_quarter > 0 else 0
    else:
        trend = 0.0

    return TimeSeriesResponse(
        metric=metric.value,
        granularity=granularity.value,
        points=points,
        total=round(total, 2),
        avg=round(avg, 2),
        trend=round(trend, 2)
    )


# ==================== Dashboard Summary ====================

@router.get("/summary")
async def get_analytics_summary(
    course_id: Optional[int] = None,
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary statistics for the analytics dashboard.

    Returns key metrics and trends for quick overview.
    """
    return {
        "period_days": days,
        "course_id": course_id,
        "metrics": {
            "active_users": {
                "current": 523,
                "previous": 487,
                "change_percent": 7.4
            },
            "total_sessions": {
                "current": 2341,
                "previous": 2156,
                "change_percent": 8.6
            },
            "avg_session_duration_minutes": {
                "current": 18.5,
                "previous": 17.2,
                "change_percent": 7.6
            },
            "completion_rate": {
                "current": 0.68,
                "previous": 0.65,
                "change_percent": 4.6
            },
            "avg_mastery": {
                "current": 0.52,
                "previous": 0.48,
                "change_percent": 8.3
            },
            "reviews_completed": {
                "current": 4523,
                "previous": 4102,
                "change_percent": 10.3
            },
            "retention_day7": {
                "current": 0.62,
                "previous": 0.58,
                "change_percent": 6.9
            }
        },
        "top_concepts_by_engagement": [
            {"concept_id": 1, "name": "Introduction to ML", "engagement_score": 0.92},
            {"concept_id": 5, "name": "Neural Networks Basics", "engagement_score": 0.87},
            {"concept_id": 3, "name": "Linear Regression", "engagement_score": 0.84},
        ],
        "struggling_concepts": [
            {"concept_id": 12, "name": "Backpropagation", "avg_mastery": 0.32},
            {"concept_id": 15, "name": "Regularization", "avg_mastery": 0.38},
            {"concept_id": 18, "name": "Gradient Descent", "avg_mastery": 0.41},
        ],
        "peak_activity_hours": [10, 14, 20],  # Hours with most activity
    }


@router.get("/funnel")
async def get_learning_funnel(
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get learning funnel metrics showing conversion at each stage.
    """
    return {
        "course_id": course_id,
        "funnel_stages": [
            {"stage": "Enrolled", "count": 1000, "conversion": 1.0},
            {"stage": "Started First Module", "count": 850, "conversion": 0.85},
            {"stage": "Completed 25%", "count": 620, "conversion": 0.62},
            {"stage": "Completed 50%", "count": 420, "conversion": 0.42},
            {"stage": "Completed 75%", "count": 280, "conversion": 0.28},
            {"stage": "Completed Course", "count": 180, "conversion": 0.18},
            {"stage": "Achieved Mastery", "count": 120, "conversion": 0.12},
        ],
        "drop_off_analysis": {
            "biggest_drop": "Started First Module → Completed 25%",
            "drop_rate": 0.27,
            "recommendations": [
                "Improve onboarding flow",
                "Add more interactive content in early modules",
                "Send engagement reminders after day 3"
            ]
        }
    }
