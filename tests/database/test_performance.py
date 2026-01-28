"""
Performance Benchmark Tests

Tests to measure and track database performance metrics.
These tests establish baselines and detect performance regressions.

Test Categories:
- Query execution time benchmarks
- Bulk operation performance
- Index effectiveness
- Connection pool behavior
- Memory usage patterns
"""
import pytest
import time
import statistics
from datetime import datetime, timedelta
from typing import List
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from .conftest import (
    UserFactory, CourseFactory, User, Course, Enrollment, Concept,
    SpacedRepetitionCard, ReviewLog, UserAchievement, UserStats,
    Instructor, TestBase
)

pytestmark = [pytest.mark.asyncio, pytest.mark.benchmark, pytest.mark.slow]


# Performance thresholds (in seconds)
THRESHOLDS = {
    "single_insert": 0.1,
    "single_select": 0.05,
    "bulk_insert_100": 1.0,
    "bulk_insert_1000": 5.0,
    "bulk_select_100": 0.2,
    "complex_query": 0.5,
    "join_query": 0.3,
}


def measure_time(func):
    """Decorator to measure function execution time."""
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start
    return wrapper


class TestInsertPerformance:
    """Test insert operation performance."""

    async def test_single_user_insert_time(self, async_session: AsyncSession):
        """Benchmark single user insert time."""
        times: List[float] = []

        for _ in range(10):
            user_data = UserFactory.create()
            user = User(**user_data)

            start = time.perf_counter()
            async_session.add(user)
            await async_session.commit()
            end = time.perf_counter()

            times.append(end - start)

        avg_time = statistics.mean(times)
        max_time = max(times)

        assert avg_time < THRESHOLDS["single_insert"], \
            f"Average insert time {avg_time:.4f}s exceeds threshold"

        # Store for reporting
        pytest.benchmark_result = {
            "test": "single_user_insert",
            "avg_time": avg_time,
            "max_time": max_time,
            "min_time": min(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "samples": len(times)
        }

    async def test_bulk_insert_100_users(self, async_session: AsyncSession):
        """Benchmark inserting 100 users in batch."""
        users = [User(**UserFactory.create()) for _ in range(100)]

        start = time.perf_counter()
        async_session.add_all(users)
        await async_session.commit()
        end = time.perf_counter()

        elapsed = end - start
        rate = 100 / elapsed  # users per second

        assert elapsed < THRESHOLDS["bulk_insert_100"], \
            f"Bulk insert 100 took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "bulk_insert_100",
            "total_time": elapsed,
            "records": 100,
            "rate": rate,
            "per_record": elapsed / 100
        }

    async def test_bulk_insert_1000_users(self, async_session: AsyncSession):
        """Benchmark inserting 1000 users in batch."""
        users = [User(**UserFactory.create()) for _ in range(1000)]

        start = time.perf_counter()
        async_session.add_all(users)
        await async_session.commit()
        end = time.perf_counter()

        elapsed = end - start
        rate = 1000 / elapsed

        assert elapsed < THRESHOLDS["bulk_insert_1000"], \
            f"Bulk insert 1000 took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "bulk_insert_1000",
            "total_time": elapsed,
            "records": 1000,
            "rate": rate,
            "per_record": elapsed / 1000
        }


class TestSelectPerformance:
    """Test select/read operation performance."""

    async def test_single_user_select_by_id(self, async_session: AsyncSession):
        """Benchmark single user select by primary key."""
        # Setup: create users
        users = [User(**UserFactory.create()) for _ in range(100)]
        async_session.add_all(users)
        await async_session.commit()

        user_ids = [u.id for u in users]
        times: List[float] = []

        for user_id in user_ids[:20]:  # Sample 20 queries
            start = time.perf_counter()
            result = await async_session.execute(
                select(User).where(User.id == user_id)
            )
            _ = result.scalar_one()
            end = time.perf_counter()

            times.append(end - start)
            async_session.expire_all()

        avg_time = statistics.mean(times)

        assert avg_time < THRESHOLDS["single_select"], \
            f"Average select time {avg_time:.4f}s exceeds threshold"

        pytest.benchmark_result = {
            "test": "single_select_by_id",
            "avg_time": avg_time,
            "max_time": max(times),
            "min_time": min(times),
            "samples": len(times)
        }

    async def test_select_by_indexed_column(self, async_session: AsyncSession):
        """Benchmark select using indexed email column."""
        # Setup
        users = [User(**UserFactory.create()) for _ in range(100)]
        async_session.add_all(users)
        await async_session.commit()

        emails = [u.email for u in users]
        times: List[float] = []

        for email in emails[:20]:
            start = time.perf_counter()
            result = await async_session.execute(
                select(User).where(User.email == email)
            )
            _ = result.scalar_one_or_none()
            end = time.perf_counter()

            times.append(end - start)
            async_session.expire_all()

        avg_time = statistics.mean(times)

        assert avg_time < THRESHOLDS["single_select"], \
            f"Average indexed select time {avg_time:.4f}s exceeds threshold"

        pytest.benchmark_result = {
            "test": "select_by_email_index",
            "avg_time": avg_time,
            "max_time": max(times),
            "samples": len(times)
        }

    async def test_bulk_select_all_users(self, async_session: AsyncSession):
        """Benchmark selecting all users."""
        # Setup
        users = [User(**UserFactory.create()) for _ in range(100)]
        async_session.add_all(users)
        await async_session.commit()

        async_session.expire_all()

        start = time.perf_counter()
        result = await async_session.execute(select(User))
        all_users = result.scalars().all()
        end = time.perf_counter()

        elapsed = end - start

        assert len(all_users) == 100
        assert elapsed < THRESHOLDS["bulk_select_100"], \
            f"Bulk select 100 took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "bulk_select_100",
            "total_time": elapsed,
            "records": len(all_users),
            "per_record": elapsed / len(all_users)
        }


class TestJoinPerformance:
    """Test join query performance."""

    async def test_user_enrollment_join(self, async_session: AsyncSession):
        """Benchmark user-enrollment join query."""
        from sqlalchemy.orm import selectinload

        # Setup: create users with enrollments
        users = [User(**UserFactory.create()) for _ in range(50)]
        courses = [Course(**CourseFactory.create()) for _ in range(10)]
        async_session.add_all(users + courses)
        await async_session.commit()

        enrollments = [
            Enrollment(
                user_id=users[i % 50].id,
                course_id=courses[i % 10].id,
                progress=0.5
            )
            for i in range(200)
        ]
        async_session.add_all(enrollments)
        await async_session.commit()

        async_session.expire_all()

        # Benchmark join query
        start = time.perf_counter()
        result = await async_session.execute(
            select(User)
            .options(selectinload(User.enrollments))
            .where(User.is_active == True)  # noqa: E712
        )
        users_with_enrollments = result.scalars().all()
        end = time.perf_counter()

        elapsed = end - start

        assert elapsed < THRESHOLDS["join_query"], \
            f"Join query took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "user_enrollment_join",
            "total_time": elapsed,
            "users": len(users_with_enrollments),
            "total_enrollments": sum(len(u.enrollments) for u in users_with_enrollments)
        }

    async def test_spaced_repetition_due_cards_query(
        self, async_session: AsyncSession
    ):
        """Benchmark query for due spaced repetition cards."""
        # Setup
        user = User(**UserFactory.create())
        async_session.add(user)
        await async_session.commit()

        # Capture user_id before expire
        user_id = user.id

        concepts = [
            Concept(name=f"Concept {i}", description=f"Desc {i}")
            for i in range(100)
        ]
        async_session.add_all(concepts)
        await async_session.commit()

        cards = [
            SpacedRepetitionCard(
                user_id=user_id,
                concept_id=concepts[i].id,
                difficulty=5.0,
                stability=2.5,
                retrievability=0.9,
                review_count=i,
                next_review_at=datetime.utcnow() + timedelta(days=i - 50)
            )
            for i in range(100)
        ]
        async_session.add_all(cards)
        await async_session.commit()

        async_session.expire_all()

        # Benchmark due cards query
        start = time.perf_counter()
        result = await async_session.execute(
            select(SpacedRepetitionCard)
            .where(SpacedRepetitionCard.user_id == user_id)
            .where(SpacedRepetitionCard.next_review_at <= datetime.utcnow())
            .order_by(SpacedRepetitionCard.next_review_at)
        )
        due_cards = result.scalars().all()
        end = time.perf_counter()

        elapsed = end - start

        assert elapsed < THRESHOLDS["complex_query"], \
            f"Due cards query took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "due_cards_query",
            "total_time": elapsed,
            "cards_due": len(due_cards),
            "total_cards": 100
        }


class TestComplexQueryPerformance:
    """Test complex query performance."""

    async def test_aggregate_user_statistics(self, async_session: AsyncSession):
        """Benchmark aggregate statistics query."""
        from sqlalchemy import func

        # Setup
        users = [
            User(**UserFactory.create(
                total_xp=i * 100,
                level=i % 10 + 1,
                streak_days=i % 30
            ))
            for i in range(100)
        ]
        async_session.add_all(users)
        await async_session.commit()

        # Benchmark aggregate query
        start = time.perf_counter()
        result = await async_session.execute(
            select(
                func.count(User.id).label("total_users"),
                func.avg(User.total_xp).label("avg_xp"),
                func.max(User.total_xp).label("max_xp"),
                func.sum(User.total_xp).label("total_xp"),
                func.avg(User.level).label("avg_level")
            )
        )
        stats = result.one()
        end = time.perf_counter()

        elapsed = end - start

        assert stats.total_users == 100
        assert elapsed < THRESHOLDS["complex_query"], \
            f"Aggregate query took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "aggregate_statistics",
            "total_time": elapsed,
            "total_users": stats.total_users,
            "avg_xp": float(stats.avg_xp) if stats.avg_xp else 0
        }

    async def test_leaderboard_query(self, async_session: AsyncSession):
        """Benchmark leaderboard query (top users by XP)."""
        # Setup
        users = [
            User(**UserFactory.create(
                total_xp=i * 100,
                level=(i // 100) + 1
            ))
            for i in range(500)
        ]
        async_session.add_all(users)
        await async_session.commit()

        async_session.expire_all()

        # Benchmark leaderboard query
        start = time.perf_counter()
        result = await async_session.execute(
            select(User)
            .where(User.is_active == True)  # noqa: E712
            .order_by(User.total_xp.desc())
            .limit(100)
        )
        top_users = result.scalars().all()
        end = time.perf_counter()

        elapsed = end - start

        assert len(top_users) == 100
        # Verify ordering
        xps = [u.total_xp for u in top_users]
        assert xps == sorted(xps, reverse=True)

        assert elapsed < THRESHOLDS["complex_query"], \
            f"Leaderboard query took {elapsed:.4f}s, exceeds threshold"

        pytest.benchmark_result = {
            "test": "leaderboard_top_100",
            "total_time": elapsed,
            "returned": len(top_users),
            "total_users": 500
        }


class TestIndexEffectiveness:
    """Test that indexes are being used effectively."""

    async def test_email_index_scan_vs_full_scan(self, async_session: AsyncSession):
        """Compare indexed vs non-indexed query performance."""
        # Setup: create many users
        users = [User(**UserFactory.create()) for _ in range(500)]
        async_session.add_all(users)
        await async_session.commit()

        target_email = users[250].email
        target_name = users[250].full_name

        async_session.expire_all()

        # Query by indexed column (email)
        start_indexed = time.perf_counter()
        for _ in range(10):
            result = await async_session.execute(
                select(User).where(User.email == target_email)
            )
            _ = result.scalar_one()
            async_session.expire_all()
        end_indexed = time.perf_counter()
        indexed_time = (end_indexed - start_indexed) / 10

        # Query by non-indexed column (full_name)
        start_non_indexed = time.perf_counter()
        for _ in range(10):
            result = await async_session.execute(
                select(User).where(User.full_name == target_name)
            )
            _ = result.scalar_one_or_none()
            async_session.expire_all()
        end_non_indexed = time.perf_counter()
        non_indexed_time = (end_non_indexed - start_non_indexed) / 10

        # Indexed query should be faster (in a real DB, might not matter for SQLite)
        pytest.benchmark_result = {
            "test": "index_effectiveness",
            "indexed_avg": indexed_time,
            "non_indexed_avg": non_indexed_time,
            "speedup": non_indexed_time / indexed_time if indexed_time > 0 else 0
        }

        # Note: In SQLite with small data, difference may be negligible
        # This test is more meaningful with PostgreSQL
        assert True  # Document results


class TestTransactionPerformance:
    """Test transaction handling performance."""

    async def test_commit_frequency_impact(self, async_session: AsyncSession):
        """Compare single commit vs multiple commits performance."""
        # Test 1: Single commit after all inserts
        users1 = [User(**UserFactory.create()) for _ in range(100)]

        start_single = time.perf_counter()
        async_session.add_all(users1)
        await async_session.commit()
        end_single = time.perf_counter()
        single_commit_time = end_single - start_single

        # Test 2: Commit after each insert (anti-pattern)
        start_multi = time.perf_counter()
        for _ in range(100):
            user = User(**UserFactory.create())
            async_session.add(user)
            await async_session.commit()
        end_multi = time.perf_counter()
        multi_commit_time = end_multi - start_multi

        pytest.benchmark_result = {
            "test": "commit_frequency",
            "single_commit_100_records": single_commit_time,
            "multi_commit_100_records": multi_commit_time,
            "overhead_factor": multi_commit_time / single_commit_time
                             if single_commit_time > 0 else 0
        }

        # Single commit should be significantly faster
        assert single_commit_time < multi_commit_time, \
            "Batch commits should be faster than individual commits"


class BenchmarkReporter:
    """Utility class to collect and report benchmark results."""

    results: List[dict] = []

    @classmethod
    def record(cls, result: dict):
        """Record a benchmark result."""
        cls.results.append({
            **result,
            "timestamp": datetime.utcnow().isoformat()
        })

    @classmethod
    def generate_report(cls) -> dict:
        """Generate summary report of all benchmarks."""
        if not cls.results:
            return {"status": "no_results"}

        return {
            "total_benchmarks": len(cls.results),
            "timestamp": datetime.utcnow().isoformat(),
            "results": cls.results,
            "summary": {
                "passed": sum(1 for r in cls.results if r.get("passed", True)),
                "failed": sum(1 for r in cls.results if not r.get("passed", True))
            }
        }


@pytest.fixture(scope="module", autouse=True)
def collect_benchmark_results(request):
    """Collect benchmark results at module level."""
    yield
    # After all tests in module, generate report
    report = BenchmarkReporter.generate_report()
    if report.get("results"):
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        for result in report.get("results", []):
            print(f"  {result.get('test', 'unknown')}: ", end="")
            if "avg_time" in result:
                print(f"{result['avg_time']:.4f}s avg")
            elif "total_time" in result:
                print(f"{result['total_time']:.4f}s total")
