"""
Dataset classes for Neural ODE training.

Loads review history from database, groups by (user_id, concept_id) pairs,
and constructs sequences for training the memory model.
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from datetime import datetime, timedelta
import math


@dataclass
class ReviewEvent:
    """Single review event with all associated data."""
    time_hours: float  # Hours since first review
    grade: int  # 1-4 (Again, Hard, Good, Easy)
    response_time_ms: int
    normalized_rt: float
    hesitation_count: int
    cursor_tortuosity: float
    retrieval_fluency: float
    elapsed_days: int
    stability_after: float
    difficulty_after: float


@dataclass
class ReviewSequence:
    """Complete review sequence for a user-concept pair."""
    user_id: int
    concept_id: int
    card_features: torch.Tensor  # [64] difficulty, complexity, embedding
    user_features: torch.Tensor  # [16] avg_retention, review_count, etc.
    events: List[ReviewEvent] = field(default_factory=list)

    # Targets for training
    recall_outcomes: List[bool] = field(default_factory=list)  # Did user recall?

    def __len__(self) -> int:
        return len(self.events)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert to tensors for model input."""
        if not self.events:
            raise ValueError("Cannot convert empty sequence to tensors")

        # Event times
        times = torch.tensor([e.time_hours for e in self.events], dtype=torch.float32)

        # Grades (1-4)
        grades = torch.tensor([e.grade for e in self.events], dtype=torch.long)

        # Telemetry [n_events, 4]: RT_norm, hesitation, tortuosity, fluency
        telemetry = torch.tensor([
            [e.normalized_rt, e.hesitation_count / 10.0,  # Normalize hesitation
             e.cursor_tortuosity, e.retrieval_fluency]
            for e in self.events
        ], dtype=torch.float32)

        # Recall outcomes (grade >= 2 means recalled, grade == 1 means forgot)
        recalls = torch.tensor(
            [1.0 if e.grade >= 2 else 0.0 for e in self.events],
            dtype=torch.float32
        )

        # Response times as targets
        latencies = torch.tensor(
            [e.normalized_rt for e in self.events],
            dtype=torch.float32
        )

        return {
            'card_features': self.card_features,
            'user_features': self.user_features,
            'times': times,
            'grades': grades,
            'telemetry': telemetry,
            'recalls': recalls,
            'latencies': latencies,
        }


class FeatureExtractor:
    """
    Extracts features from database records for Neural ODE training.

    Feature dimensions:
    - card_features: 64D (difficulty, complexity, stability stats, embedding placeholder)
    - user_features: 16D (avg_retention, review_count, intervals, phenotype factors)
    - telemetry: 4D (normalized_rt, hesitation, tortuosity, fluency)
    """

    CARD_FEAT_DIM = 64
    USER_FEAT_DIM = 16
    TELEMETRY_DIM = 4

    def __init__(self, embedding_dim: int = 48):
        """
        Initialize feature extractor.

        Args:
            embedding_dim: Dimension of semantic embeddings (from vector store)
        """
        self.embedding_dim = embedding_dim
        # Reserve: 8 for stats + 8 for complexity + 48 for embedding = 64
        self.stats_dim = 8
        self.complexity_dim = 8

    def extract_card_features(
        self,
        card_data: Dict[str, Any],
        review_history: List[Dict[str, Any]],
        embedding: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Extract 64D card feature vector.

        Args:
            card_data: SpacedRepetitionCard data dict
            review_history: List of ReviewLog records for this card
            embedding: Optional semantic embedding from vector store

        Returns:
            Tensor of shape [64]
        """
        features = []

        # 1. Basic card stats (8 dims)
        features.extend([
            card_data.get('difficulty', 5.0) / 10.0,  # Normalize 1-10 to 0-1
            card_data.get('stability', 0.0) / 365.0,  # Normalize days to ~0-1
            card_data.get('reps', 0) / 100.0,  # Normalize review count
            card_data.get('lapses', 0) / 20.0,  # Normalize lapse count
            1.0 if card_data.get('state') == 'review' else 0.0,
            1.0 if card_data.get('state') == 'learning' else 0.0,
            1.0 if card_data.get('state') == 'relearning' else 0.0,
            card_data.get('elapsed_days', 0) / 30.0,  # Days since last review
        ])

        # 2. Review history complexity features (8 dims)
        if review_history:
            ratings = [r.get('rating', 3) for r in review_history]
            intervals = [r.get('elapsed_days', 1) for r in review_history if r.get('elapsed_days')]
            durations = [r.get('review_duration_ms', 5000) for r in review_history if r.get('review_duration_ms')]

            features.extend([
                np.mean(ratings) / 4.0 if ratings else 0.5,  # Avg rating
                np.std(ratings) / 2.0 if len(ratings) > 1 else 0.0,  # Rating variance
                np.mean(intervals) / 30.0 if intervals else 0.0,  # Avg interval
                np.std(intervals) / 30.0 if len(intervals) > 1 else 0.0,  # Interval variance
                (ratings.count(1) / len(ratings)) if ratings else 0.0,  # Lapse rate
                np.mean(durations) / 30000.0 if durations else 0.5,  # Avg duration
                np.median(durations) / 30000.0 if durations else 0.5,  # Median duration
                len(ratings) / 100.0,  # Review count
            ])
        else:
            features.extend([0.5] * 8)  # Default for new cards

        # 3. Semantic embedding (48 dims)
        if embedding is not None and len(embedding) >= self.embedding_dim:
            features.extend(embedding[:self.embedding_dim].tolist())
        else:
            # Placeholder: zero embedding for cards without semantic vectors
            features.extend([0.0] * self.embedding_dim)

        # Ensure exactly 64 dims
        features = features[:self.CARD_FEAT_DIM]
        while len(features) < self.CARD_FEAT_DIM:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32)

    def extract_user_features(
        self,
        user_stats: Dict[str, Any],
        phenotype_data: Optional[Dict[str, Any]] = None,
        circadian_data: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Extract 16D user feature vector.

        Args:
            user_stats: Aggregated user statistics
            phenotype_data: LearnerPhenotype record if available
            circadian_data: UserCircadianPattern record if available

        Returns:
            Tensor of shape [16]
        """
        features = []

        # 1. Learning performance stats (6 dims)
        features.extend([
            user_stats.get('avg_retention', 0.8),  # Average retention rate
            user_stats.get('total_reviews', 0) / 1000.0,  # Normalized review count
            user_stats.get('avg_interval_days', 1.0) / 30.0,  # Avg interval
            user_stats.get('retention_variance', 0.1),  # Retention variance
            user_stats.get('forgetting_rate', 0.1),  # Rate of lapses
            user_stats.get('learning_speed', 1.0),  # Relative learning speed
        ])

        # 2. Phenotype features (4 dims)
        if phenotype_data:
            features.extend([
                phenotype_data.get('decay_rate_factor', 1.0),
                phenotype_data.get('learning_rate_factor', 1.0),
                phenotype_data.get('assignment_confidence', 0.5),
                phenotype_data.get('phenotype_id', 1) / 7.0,  # Normalized phenotype ID
            ])
        else:
            features.extend([1.0, 1.0, 0.5, 0.14])  # Defaults (steady learner)

        # 3. Circadian features (4 dims)
        if circadian_data:
            features.extend([
                circadian_data.get('amplitude', 0.2),
                circadian_data.get('phase_offset', 0.0) / 12.0,  # Normalize hours
                circadian_data.get('typical_sleep_start', 23) / 24.0,
                circadian_data.get('typical_sleep_end', 7) / 24.0,
            ])
        else:
            features.extend([0.2, 0.0, 23/24, 7/24])  # Defaults

        # 4. Session features (2 dims)
        features.extend([
            user_stats.get('avg_session_length', 20) / 60.0,  # Minutes normalized
            user_stats.get('sessions_per_week', 3) / 7.0,  # Frequency
        ])

        # Ensure exactly 16 dims
        features = features[:self.USER_FEAT_DIM]
        while len(features) < self.USER_FEAT_DIM:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32)

    def extract_telemetry(
        self,
        response_time_obs: Optional[Dict[str, Any]],
        review_duration_ms: int = 5000,
    ) -> torch.Tensor:
        """
        Extract 4D telemetry vector for a single review.

        Args:
            response_time_obs: ResponseTimeObservation record if available
            review_duration_ms: Fallback duration from ReviewLog

        Returns:
            Tensor of shape [4]: [normalized_rt, hesitation, tortuosity, fluency]
        """
        if response_time_obs:
            # Use detailed telemetry
            rt = response_time_obs.get('normalized_rt')
            if rt is None:
                # Compute if raw RT available
                raw_rt = response_time_obs.get('response_time_ms', review_duration_ms)
                rt = math.log(min(raw_rt, 60000) + 1) / math.log(60001)  # Normalize to ~0-1

            hesitation = response_time_obs.get('hesitation_count', 0) / 10.0
            tortuosity = response_time_obs.get('cursor_tortuosity', 1.0)
            fluency = response_time_obs.get('retrieval_fluency', 0.5)
        else:
            # Derive from review duration only
            rt = math.log(min(review_duration_ms, 60000) + 1) / math.log(60001)
            hesitation = 0.0  # Unknown
            tortuosity = 1.0  # Assume direct
            fluency = 1.0 - min(rt, 1.0)  # Faster = more fluent

        return torch.tensor([rt, hesitation, tortuosity, fluency], dtype=torch.float32)


class ReviewSequenceDataset(Dataset):
    """
    PyTorch Dataset for Neural ODE training.

    Loads review history from database, groups by (user_id, concept_id),
    and provides sequences for batch training.
    """

    def __init__(
        self,
        sequences: List[ReviewSequence],
        min_reviews: int = 5,
        max_sequence_length: int = 100,
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of ReviewSequence objects
            min_reviews: Minimum reviews required per sequence
            max_sequence_length: Maximum events to include
        """
        self.max_sequence_length = max_sequence_length

        # Filter sequences with enough reviews
        self.sequences = [
            s for s in sequences
            if len(s.events) >= min_reviews
        ]

        # Truncate long sequences
        for seq in self.sequences:
            if len(seq.events) > max_sequence_length:
                seq.events = seq.events[-max_sequence_length:]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence as tensors."""
        return self.sequences[idx].to_tensors()

    @classmethod
    def from_database_records(
        cls,
        review_logs: List[Dict[str, Any]],
        cards: Dict[int, Dict[str, Any]],
        user_stats: Dict[int, Dict[str, Any]],
        telemetry: Dict[int, Dict[str, Any]],
        feature_extractor: Optional[FeatureExtractor] = None,
        min_reviews: int = 5,
    ) -> 'ReviewSequenceDataset':
        """
        Create dataset from raw database records.

        Args:
            review_logs: List of ReviewLog records
            cards: Map from card_id to SpacedRepetitionCard data
            user_stats: Map from user_id to aggregated stats
            telemetry: Map from review_log_id to ResponseTimeObservation
            feature_extractor: Optional FeatureExtractor instance
            min_reviews: Minimum reviews per sequence

        Returns:
            ReviewSequenceDataset instance
        """
        if feature_extractor is None:
            feature_extractor = FeatureExtractor()

        # Group reviews by (user_id, card_id)
        grouped: Dict[Tuple[int, int], List[Dict]] = {}
        for log in review_logs:
            key = (log['user_id'], log['card_id'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(log)

        sequences = []

        for (user_id, card_id), logs in grouped.items():
            if len(logs) < min_reviews:
                continue

            # Sort by review time
            logs.sort(key=lambda x: x['review_time'])

            card_data = cards.get(card_id, {})
            concept_id = card_data.get('concept_id', card_id)
            user_stat = user_stats.get(user_id, {})

            # Extract features
            card_features = feature_extractor.extract_card_features(
                card_data, logs
            )
            user_features = feature_extractor.extract_user_features(user_stat)

            # Build events
            events = []
            first_time = logs[0]['review_time']

            for log in logs:
                # Time in hours since first review
                if isinstance(first_time, datetime) and isinstance(log['review_time'], datetime):
                    time_delta = log['review_time'] - first_time
                    time_hours = time_delta.total_seconds() / 3600.0
                else:
                    time_hours = 0.0

                # Get telemetry if available
                telem = telemetry.get(log['id'], {})
                telem_tensor = feature_extractor.extract_telemetry(
                    telem, log.get('review_duration_ms', 5000)
                )

                event = ReviewEvent(
                    time_hours=time_hours,
                    grade=log.get('rating', 3),
                    response_time_ms=log.get('review_duration_ms', 5000),
                    normalized_rt=telem_tensor[0].item(),
                    hesitation_count=int(telem.get('hesitation_count', 0)),
                    cursor_tortuosity=telem.get('cursor_tortuosity', 1.0),
                    retrieval_fluency=telem.get('retrieval_fluency', 0.5),
                    elapsed_days=log.get('elapsed_days', 0),
                    stability_after=log.get('stability', 0.0),
                    difficulty_after=log.get('difficulty', 5.0),
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

        return cls(sequences, min_reviews=min_reviews)


def collate_review_sequences(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads variable-length sequences and creates attention masks.

    Args:
        batch: List of sequence tensors from __getitem__

    Returns:
        Batched and padded tensors with masks
    """
    batch_size = len(batch)
    max_len = max(b['times'].size(0) for b in batch)

    # Initialize padded tensors
    card_features = torch.stack([b['card_features'] for b in batch])  # [B, 64]
    user_features = torch.stack([b['user_features'] for b in batch])  # [B, 16]

    times = torch.zeros(batch_size, max_len)
    grades = torch.zeros(batch_size, max_len, dtype=torch.long)
    telemetry = torch.zeros(batch_size, max_len, 4)
    recalls = torch.zeros(batch_size, max_len)
    latencies = torch.zeros(batch_size, max_len)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        seq_len = b['times'].size(0)
        times[i, :seq_len] = b['times']
        grades[i, :seq_len] = b['grades']
        telemetry[i, :seq_len] = b['telemetry']
        recalls[i, :seq_len] = b['recalls']
        latencies[i, :seq_len] = b['latencies']
        mask[i, :seq_len] = True

    return {
        'card_features': card_features,
        'user_features': user_features,
        'times': times,
        'grades': grades,
        'telemetry': telemetry,
        'recalls': recalls,
        'latencies': latencies,
        'mask': mask,
        'sequence_lengths': torch.tensor([b['times'].size(0) for b in batch]),
    }
