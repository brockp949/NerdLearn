"""
FSRS Parameter Optimization Tasks

Scheduled tasks for optimizing user-specific FSRS parameters based on review history.
Runs periodically to improve spaced repetition scheduling accuracy.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import math
from celery import shared_task
from ..celery_app import app, DEFAULT_RETRY_KWARGS

logger = logging.getLogger(__name__)


# FSRS parameter bounds for optimization
PARAM_BOUNDS = {
    "w0": (0.1, 2.0),   # Initial stability for AGAIN
    "w1": (0.1, 3.0),   # Initial stability for HARD
    "w2": (1.0, 5.0),   # Initial stability for GOOD
    "w3": (2.0, 10.0),  # Initial stability for EASY
    "w4": (3.0, 8.0),   # Stability multiplier base
    "w5": (0.5, 1.5),   # Stability multiplier factor
    "w6": (0.5, 1.5),   # Difficulty weight
    "w7": (0.001, 0.1), # Difficulty decay
    "w8": (1.0, 3.0),   # Stability increase for successful recall
    "w9": (0.05, 0.5),  # Stability factor for difficulty
}

# Default FSRS parameters
DEFAULT_PARAMS = [
    0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01,
    1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61
]


class FSRSOptimizer:
    """
    Optimizes FSRS parameters based on user review history.

    Uses gradient descent to minimize prediction error for review outcomes.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        min_reviews: int = 50,
    ):
        """
        Initialize optimizer.

        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum optimization iterations
            min_reviews: Minimum reviews required for optimization
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_reviews = min_reviews

    def calculate_loss(
        self,
        params: List[float],
        reviews: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate loss (RMSE of retrievability prediction).

        Args:
            params: Current parameter values
            reviews: List of review records

        Returns:
            Root mean squared error
        """
        total_error = 0.0
        count = 0

        for review in reviews:
            if review.get("elapsed_days", 0) == 0 or review.get("stability", 0) == 0:
                continue

            # Predicted retrievability
            elapsed = review["elapsed_days"]
            stability = review["stability"]
            predicted_r = math.pow(0.9, elapsed / stability)

            # Actual outcome (1 if recalled, 0 if forgot)
            actual = 1.0 if review["rating"] >= 3 else 0.0

            # Binary cross-entropy loss
            epsilon = 1e-7
            predicted_r = max(epsilon, min(1 - epsilon, predicted_r))
            error = -(actual * math.log(predicted_r) + (1 - actual) * math.log(1 - predicted_r))
            total_error += error
            count += 1

        if count == 0:
            return float("inf")

        return total_error / count

    def compute_gradient(
        self,
        params: List[float],
        reviews: List[Dict[str, Any]],
        param_index: int,
    ) -> float:
        """
        Compute numerical gradient for a parameter.

        Args:
            params: Current parameters
            reviews: Review data
            param_index: Index of parameter to optimize

        Returns:
            Approximate gradient
        """
        h = 0.001
        params_plus = params.copy()
        params_minus = params.copy()

        params_plus[param_index] += h
        params_minus[param_index] -= h

        loss_plus = self.calculate_loss(params_plus, reviews)
        loss_minus = self.calculate_loss(params_minus, reviews)

        return (loss_plus - loss_minus) / (2 * h)

    def optimize(
        self,
        reviews: List[Dict[str, Any]],
        initial_params: Optional[List[float]] = None,
    ) -> Tuple[List[float], float]:
        """
        Optimize FSRS parameters using gradient descent.

        Args:
            reviews: User's review history
            initial_params: Starting parameters (default: standard FSRS params)

        Returns:
            (Optimized parameters, final loss)
        """
        if len(reviews) < self.min_reviews:
            logger.info(f"Insufficient reviews ({len(reviews)} < {self.min_reviews})")
            return initial_params or DEFAULT_PARAMS[:10], float("inf")

        params = list(initial_params or DEFAULT_PARAMS[:10])
        best_params = params.copy()
        best_loss = self.calculate_loss(params, reviews)

        for iteration in range(self.max_iterations):
            # Update each parameter
            for i in range(len(params)):
                if i >= len(PARAM_BOUNDS):
                    continue

                gradient = self.compute_gradient(params, reviews, i)

                # Gradient descent step
                params[i] -= self.learning_rate * gradient

                # Apply bounds
                bounds = list(PARAM_BOUNDS.values())[i]
                params[i] = max(bounds[0], min(bounds[1], params[i]))

            # Calculate new loss
            current_loss = self.calculate_loss(params, reviews)

            # Track best
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()

            # Early stopping if converged
            if iteration > 10 and abs(current_loss - best_loss) < 1e-6:
                logger.info(f"Converged at iteration {iteration}")
                break

        return best_params, best_loss


@app.task(name="optimize_user_fsrs_params", bind=True, **DEFAULT_RETRY_KWARGS)
def optimize_user_fsrs_params(self, user_id: int) -> Dict[str, Any]:
    """
    Optimize FSRS parameters for a single user.

    Args:
        user_id: User to optimize parameters for

    Returns:
        Optimization results
    """
    from sqlalchemy import create_engine, text
    from ..config import config

    logger.info(f"Optimizing FSRS parameters for user {user_id}")

    # Update task progress
    self.update_state(
        state="PROCESSING",
        meta={"step": "fetching_reviews", "user_id": user_id}
    )

    # Connect to database
    engine = create_engine(config.DATABASE_URL)

    try:
        with engine.connect() as conn:
            # Fetch user's review history
            result = conn.execute(
                text("""
                    SELECT
                        rl.card_id,
                        rl.rating,
                        rl.elapsed_days,
                        rl.stability,
                        rl.difficulty,
                        rl.review_time
                    FROM review_logs rl
                    JOIN spaced_repetition_cards src ON rl.card_id = src.id
                    WHERE src.user_id = :user_id
                    ORDER BY rl.review_time ASC
                """),
                {"user_id": user_id}
            )

            reviews = [dict(row._mapping) for row in result]

            if len(reviews) < 50:
                return {
                    "user_id": user_id,
                    "status": "skipped",
                    "reason": f"Insufficient reviews ({len(reviews)} < 50)",
                    "review_count": len(reviews),
                }

            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={"step": "optimizing", "user_id": user_id, "review_count": len(reviews)}
            )

            # Run optimization
            optimizer = FSRSOptimizer()
            optimized_params, loss = optimizer.optimize(reviews)

            # Store optimized parameters
            conn.execute(
                text("""
                    INSERT INTO user_fsrs_params (user_id, params, loss, optimized_at, review_count)
                    VALUES (:user_id, :params, :loss, :optimized_at, :review_count)
                    ON CONFLICT (user_id) DO UPDATE
                    SET params = :params, loss = :loss, optimized_at = :optimized_at,
                        review_count = :review_count
                """),
                {
                    "user_id": user_id,
                    "params": ",".join(str(p) for p in optimized_params),
                    "loss": loss,
                    "optimized_at": datetime.utcnow(),
                    "review_count": len(reviews),
                }
            )
            conn.commit()

            logger.info(f"Optimized FSRS params for user {user_id}, loss={loss:.4f}")

            return {
                "user_id": user_id,
                "status": "success",
                "review_count": len(reviews),
                "loss": loss,
                "params": optimized_params,
            }

    except Exception as e:
        logger.error(f"FSRS optimization failed for user {user_id}: {e}")
        raise


@app.task(name="optimize_all_users_fsrs", bind=True, **DEFAULT_RETRY_KWARGS)
def optimize_all_users_fsrs(self, min_reviews: int = 50) -> Dict[str, Any]:
    """
    Batch optimize FSRS parameters for all eligible users.

    Args:
        min_reviews: Minimum reviews required for optimization

    Returns:
        Summary of optimization results
    """
    from sqlalchemy import create_engine, text
    from ..config import config

    logger.info("Starting batch FSRS parameter optimization")

    # Update task progress
    self.update_state(
        state="PROCESSING",
        meta={"step": "finding_eligible_users"}
    )

    engine = create_engine(config.DATABASE_URL)

    try:
        with engine.connect() as conn:
            # Find users with enough reviews who haven't been optimized recently
            result = conn.execute(
                text("""
                    SELECT
                        src.user_id,
                        COUNT(rl.id) as review_count
                    FROM spaced_repetition_cards src
                    JOIN review_logs rl ON rl.card_id = src.id
                    LEFT JOIN user_fsrs_params ufp ON ufp.user_id = src.user_id
                    WHERE (ufp.optimized_at IS NULL
                           OR ufp.optimized_at < NOW() - INTERVAL '7 days')
                    GROUP BY src.user_id
                    HAVING COUNT(rl.id) >= :min_reviews
                """),
                {"min_reviews": min_reviews}
            )

            eligible_users = [dict(row._mapping) for row in result]

        logger.info(f"Found {len(eligible_users)} users eligible for optimization")

        # Optimize each user
        results = {
            "total_users": len(eligible_users),
            "optimized": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }

        for i, user_data in enumerate(eligible_users):
            user_id = user_data["user_id"]

            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "step": "optimizing_users",
                    "current_user": user_id,
                    "progress": i + 1,
                    "total": len(eligible_users),
                }
            )

            try:
                # Call individual optimization
                result = optimize_user_fsrs_params.delay(user_id)
                user_result = result.get(timeout=300)  # 5 minute timeout per user

                if user_result.get("status") == "success":
                    results["optimized"] += 1
                else:
                    results["skipped"] += 1

                results["details"].append(user_result)

            except Exception as e:
                logger.error(f"Failed to optimize user {user_id}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "user_id": user_id,
                    "status": "failed",
                    "error": str(e),
                })

        logger.info(
            f"Batch optimization complete: {results['optimized']} optimized, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )

        return results

    except Exception as e:
        logger.error(f"Batch FSRS optimization failed: {e}")
        raise


@app.task(name="scheduled_fsrs_optimization")
def scheduled_fsrs_optimization() -> Dict[str, Any]:
    """
    Scheduled task for periodic FSRS optimization.

    Called by Celery beat to optimize parameters for all eligible users.
    """
    logger.info("Running scheduled FSRS optimization")

    try:
        # Trigger batch optimization
        result = optimize_all_users_fsrs.delay(min_reviews=50)
        task_id = result.id

        return {
            "status": "triggered",
            "task_id": task_id,
            "triggered_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Scheduled FSRS optimization failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "triggered_at": datetime.utcnow().isoformat(),
        }
