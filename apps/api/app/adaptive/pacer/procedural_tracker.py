"""
Procedural Progress Tracker
Tracks user progress through procedural (practice) content

Per PACER protocol:
- Procedural (P) = step-by-step instructions for HOW to do something
- Must be learned through PRACTICE, not just reading
- Track step completion, timing, errors to measure skill acquisition
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.pacer import (
    ProceduralProgress,
    PACERContentItem,
    PACERType,
)


@dataclass
class StepResult:
    """Result of completing a procedural step"""

    step_number: int
    success: bool
    time_ms: int
    error_count: int
    feedback: Optional[str] = None


@dataclass
class ProcedureStatus:
    """Current status of a user's procedural learning"""

    item_id: int
    title: str
    current_step: int
    total_steps: int
    completed: bool
    attempts: int
    progress_percent: float
    average_step_time_ms: float
    total_errors: int


class ProceduralProgressTracker:
    """
    Tracks and manages procedural learning progress per PACER protocol.

    Key operations:
    1. Initialize progress for a procedural item
    2. Record step completions with timing and errors
    3. Track overall progress and mastery
    4. Generate practice recommendations
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    async def start_procedure(
        self,
        user_id: int,
        pacer_item_id: int,
        total_steps: int,
    ) -> ProceduralProgress:
        """
        Initialize or reset progress for a procedural item.

        Args:
            user_id: ID of the user
            pacer_item_id: ID of the procedural PACERContentItem
            total_steps: Total number of steps in the procedure

        Returns:
            ProceduralProgress record
        """
        if not self.db:
            raise ValueError("Database session required")

        # Check for existing progress
        existing = await self._get_progress(user_id, pacer_item_id)

        if existing:
            # Reset if restarting
            existing.current_step = 0
            existing.attempts += 1
            existing.completed = False
            existing.step_times_ms = []
            existing.errors_per_step = []
            existing.step_completions = [False] * total_steps
            existing.completed_at = None
            await self.db.flush()
            return existing

        # Create new progress
        progress = ProceduralProgress(
            user_id=user_id,
            pacer_item_id=pacer_item_id,
            total_steps=total_steps,
            current_step=0,
            attempts=1,
            completed=False,
            step_times_ms=[],
            errors_per_step=[],
            step_completions=[False] * total_steps,
        )

        self.db.add(progress)
        await self.db.flush()
        return progress

    async def complete_step(
        self,
        user_id: int,
        pacer_item_id: int,
        step_number: int,
        success: bool,
        time_ms: int,
        error_count: int = 0,
    ) -> StepResult:
        """
        Record completion of a procedural step.

        Args:
            user_id: ID of the user
            pacer_item_id: ID of the procedural item
            step_number: Which step was completed (0-indexed)
            success: Whether step was completed successfully
            time_ms: Time taken in milliseconds
            error_count: Number of errors made

        Returns:
            StepResult with feedback
        """
        if not self.db:
            raise ValueError("Database session required")

        progress = await self._get_progress(user_id, pacer_item_id)
        if not progress:
            raise ValueError("Progress not found. Call start_procedure first.")

        # Validate step number
        if step_number < 0 or step_number >= progress.total_steps:
            raise ValueError(f"Invalid step number: {step_number}")

        # Update step data
        step_times = progress.step_times_ms or []
        errors = progress.errors_per_step or []
        completions = progress.step_completions or [False] * progress.total_steps

        # Extend lists if needed
        while len(step_times) <= step_number:
            step_times.append(0)
        while len(errors) <= step_number:
            errors.append(0)
        while len(completions) <= step_number:
            completions.append(False)

        # Record this attempt
        step_times[step_number] = time_ms
        errors[step_number] = error_count
        completions[step_number] = success

        progress.step_times_ms = step_times
        progress.errors_per_step = errors
        progress.step_completions = completions

        # Update current step if successful
        if success and step_number == progress.current_step:
            progress.current_step = step_number + 1

        # Check if procedure is complete
        if progress.current_step >= progress.total_steps:
            progress.completed = True
            progress.completed_at = datetime.utcnow()

        await self.db.flush()

        # Generate feedback
        feedback = self._generate_step_feedback(
            step_number, success, time_ms, error_count, progress.total_steps
        )

        return StepResult(
            step_number=step_number,
            success=success,
            time_ms=time_ms,
            error_count=error_count,
            feedback=feedback,
        )

    async def get_progress_status(
        self,
        user_id: int,
        pacer_item_id: int,
    ) -> Optional[ProcedureStatus]:
        """
        Get current status of user's procedural progress.

        Args:
            user_id: ID of the user
            pacer_item_id: ID of the procedural item

        Returns:
            ProcedureStatus or None if not started
        """
        if not self.db:
            raise ValueError("Database session required")

        progress = await self._get_progress(user_id, pacer_item_id)
        if not progress:
            return None

        # Get item title
        item_result = await self.db.execute(
            select(PACERContentItem).where(PACERContentItem.id == pacer_item_id)
        )
        item = item_result.scalar_one_or_none()
        title = item.title if item else "Unknown"

        # Calculate metrics
        step_times = progress.step_times_ms or []
        errors = progress.errors_per_step or []

        avg_time = sum(step_times) / len(step_times) if step_times else 0
        total_errors = sum(errors)
        progress_percent = (progress.current_step / progress.total_steps) * 100

        return ProcedureStatus(
            item_id=pacer_item_id,
            title=title,
            current_step=progress.current_step,
            total_steps=progress.total_steps,
            completed=progress.completed,
            attempts=progress.attempts,
            progress_percent=progress_percent,
            average_step_time_ms=avg_time,
            total_errors=total_errors,
        )

    async def get_user_active_procedures(
        self,
        user_id: int,
        include_completed: bool = False,
    ) -> List[ProcedureStatus]:
        """
        Get all procedures a user is working on.

        Args:
            user_id: ID of the user
            include_completed: Whether to include completed procedures

        Returns:
            List of ProcedureStatus
        """
        if not self.db:
            raise ValueError("Database session required")

        query = select(ProceduralProgress).where(
            ProceduralProgress.user_id == user_id
        )

        if not include_completed:
            query = query.where(ProceduralProgress.completed == False)

        result = await self.db.execute(query)
        progress_records = result.scalars().all()

        statuses = []
        for progress in progress_records:
            status = await self.get_progress_status(user_id, progress.pacer_item_id)
            if status:
                statuses.append(status)

        return statuses

    async def get_procedure_analytics(
        self,
        user_id: int,
        pacer_item_id: int,
    ) -> Dict[str, Any]:
        """
        Get detailed analytics for a user's procedural learning.

        Args:
            user_id: ID of the user
            pacer_item_id: ID of the procedural item

        Returns:
            Analytics data including per-step performance
        """
        if not self.db:
            raise ValueError("Database session required")

        progress = await self._get_progress(user_id, pacer_item_id)
        if not progress:
            return {}

        step_times = progress.step_times_ms or []
        errors = progress.errors_per_step or []
        completions = progress.step_completions or []

        # Identify problem steps (high errors or time)
        avg_time = sum(step_times) / len(step_times) if step_times else 0
        problem_steps = []

        for i, (time, err, done) in enumerate(zip(step_times, errors, completions)):
            if err > 2 or time > avg_time * 2:
                problem_steps.append({
                    "step": i,
                    "time_ms": time,
                    "errors": err,
                    "completed": done,
                    "issue": "high_errors" if err > 2 else "slow",
                })

        return {
            "total_steps": progress.total_steps,
            "completed_steps": progress.current_step,
            "attempts": progress.attempts,
            "total_time_ms": sum(step_times),
            "average_step_time_ms": avg_time,
            "total_errors": sum(errors),
            "step_details": [
                {
                    "step": i,
                    "time_ms": step_times[i] if i < len(step_times) else 0,
                    "errors": errors[i] if i < len(errors) else 0,
                    "completed": completions[i] if i < len(completions) else False,
                }
                for i in range(progress.total_steps)
            ],
            "problem_steps": problem_steps,
            "mastery_estimate": self._estimate_mastery(progress),
        }

    async def reset_progress(
        self,
        user_id: int,
        pacer_item_id: int,
    ) -> bool:
        """Reset user's progress on a procedure"""
        if not self.db:
            raise ValueError("Database session required")

        progress = await self._get_progress(user_id, pacer_item_id)
        if not progress:
            return False

        progress.current_step = 0
        progress.completed = False
        progress.step_times_ms = []
        progress.errors_per_step = []
        progress.step_completions = [False] * progress.total_steps
        progress.completed_at = None
        progress.attempts += 1

        await self.db.flush()
        return True

    async def _get_progress(
        self, user_id: int, pacer_item_id: int
    ) -> Optional[ProceduralProgress]:
        """Get progress record for user and item"""
        result = await self.db.execute(
            select(ProceduralProgress).where(
                and_(
                    ProceduralProgress.user_id == user_id,
                    ProceduralProgress.pacer_item_id == pacer_item_id,
                )
            )
        )
        return result.scalar_one_or_none()

    def _generate_step_feedback(
        self,
        step_number: int,
        success: bool,
        time_ms: int,
        error_count: int,
        total_steps: int,
    ) -> str:
        """Generate feedback for a completed step"""
        if not success:
            return f"Step {step_number + 1} needs more practice. Try again with the instructions."

        if error_count == 0 and time_ms < 30000:  # < 30 seconds
            return f"Excellent! Step {step_number + 1} completed efficiently."
        elif error_count <= 1:
            return f"Good job on step {step_number + 1}. Minor refinements possible."
        else:
            return f"Step {step_number + 1} complete, but review for accuracy ({error_count} errors)."

    def _estimate_mastery(self, progress: ProceduralProgress) -> float:
        """
        Estimate mastery level based on performance metrics.
        Returns 0-1 score.
        """
        if not progress.completed:
            # Partial credit for progress
            return (progress.current_step / progress.total_steps) * 0.5

        errors = progress.errors_per_step or []
        total_errors = sum(errors)
        steps = progress.total_steps

        # Penalize errors
        error_penalty = min(0.4, total_errors * 0.05)

        # Penalize multiple attempts
        attempt_penalty = min(0.2, (progress.attempts - 1) * 0.1)

        mastery = 1.0 - error_penalty - attempt_penalty
        return max(0.3, mastery)  # Minimum 0.3 if completed
