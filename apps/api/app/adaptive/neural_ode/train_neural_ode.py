"""
CLI script for training Neural ODE memory model.

Usage:
    python -m app.adaptive.neural_ode.train_neural_ode --epochs 50 --batch_size 32

For synthetic data testing:
    python -m app.adaptive.neural_ode.train_neural_ode --synthetic --num_sequences 500
"""

import argparse
import logging
import sys
import torch
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from .model import NeuralMemoryODE
from .trainer import NeuralODETrainer, TrainingConfig
from .dataset import (
    ReviewSequenceDataset,
    ReviewSequence,
    ReviewEvent,
    FeatureExtractor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    num_sequences: int = 500,
    min_reviews: int = 10,
    max_reviews: int = 50,
    seed: int = 42,
) -> List[ReviewSequence]:
    """
    Generate synthetic review sequences for testing.

    Creates realistic-looking review patterns with varying:
    - Learning speeds (some users learn faster)
    - Forgetting rates (some concepts harder to remember)
    - Review intervals (spaced repetition pattern)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    feature_extractor = FeatureExtractor()
    sequences = []

    for seq_idx in range(num_sequences):
        user_id = seq_idx // 10 + 1
        concept_id = seq_idx % 100 + 1

        # User characteristics
        user_learning_speed = np.random.uniform(0.5, 1.5)
        user_retention_rate = np.random.uniform(0.6, 0.95)

        # Concept difficulty
        concept_difficulty = np.random.uniform(0.3, 0.9)

        # Generate card features
        card_data = {
            'difficulty': concept_difficulty * 10,
            'stability': np.random.uniform(0, 100),
            'reps': np.random.randint(0, 50),
            'lapses': np.random.randint(0, 10),
            'state': np.random.choice(['new', 'learning', 'review']),
            'elapsed_days': np.random.randint(0, 30),
        }

        # Simulate review history for card feature extraction
        num_reviews = np.random.randint(min_reviews, max_reviews + 1)
        review_history = []

        for _ in range(num_reviews):
            review_history.append({
                'rating': np.random.randint(1, 5),
                'elapsed_days': np.random.randint(1, 30),
                'review_duration_ms': np.random.randint(2000, 30000),
            })

        card_features = feature_extractor.extract_card_features(
            card_data, review_history
        )

        # Generate user features
        user_stats = {
            'avg_retention': user_retention_rate,
            'total_reviews': np.random.randint(50, 500),
            'avg_interval_days': np.random.uniform(1, 14),
            'retention_variance': np.random.uniform(0.05, 0.2),
            'forgetting_rate': 1 - user_retention_rate,
            'learning_speed': user_learning_speed,
            'avg_session_length': np.random.uniform(10, 40),
            'sessions_per_week': np.random.uniform(2, 7),
        }

        phenotype_data = {
            'decay_rate_factor': np.random.uniform(0.7, 1.3),
            'learning_rate_factor': user_learning_speed,
            'assignment_confidence': np.random.uniform(0.5, 0.95),
            'phenotype_id': np.random.randint(0, 7),
        }

        user_features = feature_extractor.extract_user_features(
            user_stats, phenotype_data
        )

        # Generate review events
        events = []
        current_time = 0.0
        current_strength = 0.3  # Initial memory strength

        for review_idx in range(num_reviews):
            # Time until next review (spaced repetition pattern)
            if review_idx == 0:
                interval_hours = 0.0
            else:
                # Interval grows with successful reviews
                base_interval = 24 * (2 ** min(review_idx / 3, 4))  # Hours
                interval_hours = base_interval * np.random.uniform(0.5, 1.5)

            current_time += interval_hours

            # Simulate forgetting (memory decays over time)
            decay_rate = 0.1 * (1 / user_learning_speed) * concept_difficulty
            time_factor = np.exp(-decay_rate * interval_hours / 24)
            current_strength *= time_factor

            # Add some noise
            current_strength = np.clip(
                current_strength + np.random.normal(0, 0.05),
                0.0, 1.0
            )

            # Determine grade based on memory strength
            recall_prob = current_strength * user_retention_rate
            if np.random.random() < recall_prob:
                # Successful recall
                if recall_prob > 0.9:
                    grade = 4  # Easy
                elif recall_prob > 0.7:
                    grade = 3  # Good
                else:
                    grade = 2  # Hard
                # Strengthen memory
                current_strength = min(1.0, current_strength + 0.2 * user_learning_speed)
            else:
                # Failed recall
                grade = 1  # Again
                # Reset memory
                current_strength = 0.3

            # Generate telemetry
            # Faster response time for stronger memory
            base_rt = 5000 / (current_strength + 0.1)
            response_time_ms = int(np.clip(
                base_rt * np.random.uniform(0.5, 1.5),
                1000, 60000
            ))
            normalized_rt = np.log(min(response_time_ms, 60000) + 1) / np.log(60001)

            # More hesitation for weaker memory
            hesitation_count = int(np.clip(
                (1 - current_strength) * 5 * np.random.uniform(0.5, 1.5),
                0, 10
            ))

            tortuosity = np.clip(1.0 + (1 - current_strength) * np.random.uniform(0, 0.5), 1.0, 2.0)
            fluency = current_strength * np.random.uniform(0.7, 1.0)

            event = ReviewEvent(
                time_hours=current_time,
                grade=grade,
                response_time_ms=response_time_ms,
                normalized_rt=normalized_rt,
                hesitation_count=hesitation_count,
                cursor_tortuosity=tortuosity,
                retrieval_fluency=fluency,
                elapsed_days=int(interval_hours / 24),
                stability_after=current_strength * 100,
                difficulty_after=concept_difficulty * 10,
            )
            events.append(event)

        sequence = ReviewSequence(
            user_id=user_id,
            concept_id=concept_id,
            card_features=card_features,
            user_features=user_features,
            events=events,
        )
        sequences.append(sequence)

    return sequences


async def load_data_from_database(
    min_reviews: int = 5,
    max_users: int = None,
) -> List[ReviewSequence]:
    """
    Load real review data from database.

    Requires database connection and existing review logs.
    """
    # Import database dependencies
    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import select, func
        from app.models.spaced_repetition import ReviewLog, SpacedRepetitionCard
        from app.models.neural_ode import ResponseTimeObservation
    except ImportError as e:
        logger.error(f"Failed to import database modules: {e}")
        logger.info("Use --synthetic flag to generate synthetic data instead")
        return []

    feature_extractor = FeatureExtractor()

    async with AsyncSessionLocal() as db:
        # Query review logs with related data
        query = (
            select(ReviewLog)
            .order_by(ReviewLog.user_id, ReviewLog.card_id, ReviewLog.review_time)
        )

        result = await db.execute(query)
        review_logs = result.scalars().all()

        if not review_logs:
            logger.warning("No review logs found in database")
            return []

        # Convert to dictionaries
        logs_dicts = [
            {
                'id': log.id,
                'user_id': log.user_id,
                'card_id': log.card_id,
                'rating': log.rating,
                'review_time': log.review_time,
                'elapsed_days': log.elapsed_days,
                'scheduled_days': log.scheduled_days,
                'stability': log.stability,
                'difficulty': log.difficulty,
                'state': log.state,
                'review_duration_ms': log.review_duration_ms,
            }
            for log in review_logs
        ]

        # Get card data
        card_ids = set(log['card_id'] for log in logs_dicts)
        card_query = select(SpacedRepetitionCard).where(
            SpacedRepetitionCard.id.in_(card_ids)
        )
        card_result = await db.execute(card_query)
        cards = {
            card.id: {
                'concept_id': card.concept_id,
                'difficulty': card.difficulty,
                'stability': card.stability,
                'reps': card.reps,
                'lapses': card.lapses,
                'state': card.state,
                'elapsed_days': card.elapsed_days,
            }
            for card in card_result.scalars().all()
        }

        # Get telemetry data
        log_ids = [log['id'] for log in logs_dicts]
        telem_query = select(ResponseTimeObservation).where(
            ResponseTimeObservation.review_log_id.in_(log_ids)
        )
        telem_result = await db.execute(telem_query)
        telemetry = {
            obs.review_log_id: {
                'response_time_ms': obs.response_time_ms,
                'normalized_rt': obs.normalized_rt,
                'hesitation_count': obs.hesitation_count,
                'cursor_tortuosity': obs.cursor_tortuosity,
                'retrieval_fluency': obs.retrieval_fluency,
            }
            for obs in telem_result.scalars().all()
        }

        # Compute user stats (simplified)
        user_stats = {}
        for log in logs_dicts:
            uid = log['user_id']
            if uid not in user_stats:
                user_stats[uid] = {
                    'ratings': [],
                    'intervals': [],
                }
            user_stats[uid]['ratings'].append(log['rating'])
            if log['elapsed_days']:
                user_stats[uid]['intervals'].append(log['elapsed_days'])

        for uid, stats in user_stats.items():
            ratings = stats['ratings']
            intervals = stats['intervals']
            user_stats[uid] = {
                'avg_retention': sum(1 for r in ratings if r >= 2) / len(ratings) if ratings else 0.8,
                'total_reviews': len(ratings),
                'avg_interval_days': np.mean(intervals) if intervals else 1.0,
                'retention_variance': np.std([1 if r >= 2 else 0 for r in ratings]) if ratings else 0.1,
                'forgetting_rate': sum(1 for r in ratings if r == 1) / len(ratings) if ratings else 0.1,
                'learning_speed': 1.0,
                'avg_session_length': 20,
                'sessions_per_week': 3,
            }

        # Create dataset
        dataset = ReviewSequenceDataset.from_database_records(
            review_logs=logs_dicts,
            cards=cards,
            user_stats=user_stats,
            telemetry=telemetry,
            feature_extractor=feature_extractor,
            min_reviews=min_reviews,
        )

        return dataset.sequences


def main():
    parser = argparse.ArgumentParser(
        description="Train Neural ODE memory model"
    )

    # Data arguments
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of database"
    )
    parser.add_argument(
        "--num_sequences", type=int, default=500,
        help="Number of synthetic sequences to generate"
    )
    parser.add_argument(
        "--min_reviews", type=int, default=5,
        help="Minimum reviews per sequence"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--early_stopping", type=int, default=5,
        help="Early stopping patience"
    )

    # Model arguments
    parser.add_argument(
        "--state_dim", type=int, default=32,
        help="Latent state dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="Hidden layer dimension"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="./models/neural_ode",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on"
    )

    # Seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("Neural ODE Memory Model Training")
    logger.info("=" * 60)

    # Load or generate data
    if args.synthetic:
        logger.info(f"Generating {args.num_sequences} synthetic sequences...")
        sequences = generate_synthetic_data(
            num_sequences=args.num_sequences,
            min_reviews=args.min_reviews,
            seed=args.seed,
        )
    else:
        logger.info("Loading data from database...")
        import asyncio
        sequences = asyncio.run(load_data_from_database(
            min_reviews=args.min_reviews,
        ))

        if not sequences:
            logger.warning("No data found. Falling back to synthetic data.")
            sequences = generate_synthetic_data(
                num_sequences=args.num_sequences,
                min_reviews=args.min_reviews,
                seed=args.seed,
            )

    logger.info(f"Loaded {len(sequences)} sequences")

    # Create dataset
    dataset = ReviewSequenceDataset(
        sequences=sequences,
        min_reviews=args.min_reviews,
    )
    logger.info(f"Dataset size: {len(dataset)} sequences")

    # Create model
    model = NeuralMemoryODE(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        card_feat_dim=64,
        user_feat_dim=16,
    )
    logger.info(f"Model created with state_dim={args.state_dim}, hidden_dim={args.hidden_dim}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params:,}")

    # Create config
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.output_dir,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
    )

    # Create trainer
    trainer = NeuralODETrainer(
        model=model,
        config=config,
        device=args.device,
    )

    # Train
    logger.info("Starting training...")
    metrics = trainer.train(dataset)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"  Final train loss: {metrics.train_loss:.4f}")
    logger.info(f"  Final val loss: {metrics.val_loss:.4f}")
    logger.info(f"  Best val loss: {metrics.best_val_loss:.4f}")
    logger.info(f"  Val AUC: {metrics.val_auc:.4f}")
    logger.info(f"  Val calibration error: {metrics.val_calibration_error:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
