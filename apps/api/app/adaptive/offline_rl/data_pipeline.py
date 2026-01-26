"""
Data Pipeline for Offline RL Training

Extracts and processes student trajectories from ReviewLog for offline RL training.

Pipeline Steps:
1. Ingestion: Load raw interaction logs from database
2. State Annotation: Run TD-BKT to compute belief states
3. Reward Annotation: Use HLR to compute reward signals
4. Trajectory Formation: Group by user and order by timestamp
5. Tokenization: Prepare for model input (DT or CQL)

Output Format:
- Trajectory: [(s_t, a_t, r_t, s_{t+1}), ...]
- For Decision Transformer: (R̂_t, s_t, a_t) sequences
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """
    Single state-action-reward-next_state transition.

    This is the fundamental unit for Q-learning style algorithms.
    """
    state: np.ndarray              # Belief state vector
    action: int                    # Concept index that was practiced
    reward: float                  # HLR-based reward
    next_state: np.ndarray         # Next belief state
    done: bool = False             # Episode termination flag

    # Metadata for analysis
    user_id: Optional[str] = None
    concept_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    correct: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "user_id": self.user_id,
            "concept_id": self.concept_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Transition":
        return cls(
            state=np.array(d["state"]),
            action=d["action"],
            reward=d["reward"],
            next_state=np.array(d["next_state"]),
            done=d.get("done", False),
            user_id=d.get("user_id"),
            concept_id=d.get("concept_id"),
            timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None,
            correct=d.get("correct"),
        )


@dataclass
class Trajectory:
    """
    Complete trajectory for a user session or learning episode.

    Used for Decision Transformer (sequence modeling) and trajectory-level analysis.
    """
    user_id: str
    transitions: List[Transition] = field(default_factory=list)

    # Computed values
    total_reward: float = 0.0
    returns_to_go: Optional[List[float]] = None  # For Decision Transformer

    # Metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    course_id: Optional[int] = None

    def __len__(self) -> int:
        return len(self.transitions)

    def add_transition(self, transition: Transition):
        """Add a transition to the trajectory"""
        self.transitions.append(transition)
        self.total_reward += transition.reward

        # Update time bounds
        if transition.timestamp:
            if self.start_time is None or transition.timestamp < self.start_time:
                self.start_time = transition.timestamp
            if self.end_time is None or transition.timestamp > self.end_time:
                self.end_time = transition.timestamp

    def compute_returns_to_go(self, gamma: float = 0.99):
        """
        Compute return-to-go for each timestep.

        R̂_t = Σ_{t'=t}^T γ^{t'-t} * r_{t'}

        Args:
            gamma: Discount factor
        """
        T = len(self.transitions)
        rtg = [0.0] * T

        # Compute backwards
        running_return = 0.0
        for t in range(T - 1, -1, -1):
            running_return = self.transitions[t].reward + gamma * running_return
            rtg[t] = running_return

        self.returns_to_go = rtg

    def get_states(self) -> np.ndarray:
        """Get all states as array (T, state_dim)"""
        return np.array([t.state for t in self.transitions])

    def get_actions(self) -> np.ndarray:
        """Get all actions as array (T,)"""
        return np.array([t.action for t in self.transitions])

    def get_rewards(self) -> np.ndarray:
        """Get all rewards as array (T,)"""
        return np.array([t.reward for t in self.transitions])

    def get_returns_to_go_array(self) -> np.ndarray:
        """Get returns-to-go as array (T,)"""
        if self.returns_to_go is None:
            self.compute_returns_to_go()
        return np.array(self.returns_to_go)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "transitions": [t.to_dict() for t in self.transitions],
            "total_reward": self.total_reward,
            "returns_to_go": self.returns_to_go,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "course_id": self.course_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Trajectory":
        traj = cls(
            user_id=d["user_id"],
            total_reward=d.get("total_reward", 0.0),
            returns_to_go=d.get("returns_to_go"),
            course_id=d.get("course_id"),
        )
        if d.get("start_time"):
            traj.start_time = datetime.fromisoformat(d["start_time"])
        if d.get("end_time"):
            traj.end_time = datetime.fromisoformat(d["end_time"])

        for t_dict in d.get("transitions", []):
            traj.transitions.append(Transition.from_dict(t_dict))

        return traj


class TrajectoryDataset:
    """
    Dataset of trajectories for offline RL training.

    Provides:
    - Iteration over transitions (for CQL)
    - Iteration over trajectories (for Decision Transformer)
    - Batching and sampling
    - Statistics and analysis
    """

    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        """
        Initialize dataset.

        Args:
            trajectories: Optional initial list of trajectories
        """
        self.trajectories: List[Trajectory] = trajectories or []
        self._transition_cache: Optional[List[Transition]] = None

    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the dataset"""
        self.trajectories.append(trajectory)
        self._transition_cache = None  # Invalidate cache

    def get_all_transitions(self) -> List[Transition]:
        """Get flat list of all transitions"""
        if self._transition_cache is None:
            self._transition_cache = []
            for traj in self.trajectories:
                self._transition_cache.extend(traj.transitions)
        return self._transition_cache

    def __len__(self) -> int:
        """Number of trajectories"""
        return len(self.trajectories)

    def num_transitions(self) -> int:
        """Total number of transitions"""
        return len(self.get_all_transitions())

    def sample_transitions(self, batch_size: int) -> List[Transition]:
        """
        Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions
        """
        all_transitions = self.get_all_transitions()
        indices = np.random.choice(
            len(all_transitions),
            size=min(batch_size, len(all_transitions)),
            replace=False
        )
        return [all_transitions[i] for i in indices]

    def sample_trajectories(self, batch_size: int) -> List[Trajectory]:
        """
        Sample random batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            List of sampled trajectories
        """
        indices = np.random.choice(
            len(self.trajectories),
            size=min(batch_size, len(self.trajectories)),
            replace=False
        )
        return [self.trajectories[i] for i in indices]

    def get_transition_batch(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch of transitions as numpy arrays.

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        transitions = self.sample_transitions(batch_size)

        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def get_trajectory_batch(
        self,
        batch_size: int,
        max_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch of trajectories as padded arrays.

        Args:
            batch_size: Number of trajectories
            max_length: Maximum sequence length (pad/truncate)

        Returns:
            (returns_to_go, states, actions, timesteps, attention_mask)
        """
        trajectories = self.sample_trajectories(batch_size)

        # Get dimensions
        if trajectories:
            state_dim = trajectories[0].transitions[0].state.shape[0]
        else:
            state_dim = 1

        # Initialize arrays
        rtg = np.zeros((batch_size, max_length, 1), dtype=np.float32)
        states = np.zeros((batch_size, max_length, state_dim), dtype=np.float32)
        actions = np.zeros((batch_size, max_length), dtype=np.int64)
        timesteps = np.zeros((batch_size, max_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.float32)

        for i, traj in enumerate(trajectories):
            # Ensure returns-to-go computed
            if traj.returns_to_go is None:
                traj.compute_returns_to_go()

            length = min(len(traj), max_length)

            rtg[i, :length, 0] = traj.returns_to_go[:length]
            states[i, :length] = traj.get_states()[:length]
            actions[i, :length] = traj.get_actions()[:length]
            timesteps[i, :length] = np.arange(length)
            attention_mask[i, :length] = 1.0

        return rtg, states, actions, timesteps, attention_mask

    def compute_statistics(self) -> Dict:
        """Compute dataset statistics"""
        all_transitions = self.get_all_transitions()
        all_rewards = [t.reward for t in all_transitions]

        traj_lengths = [len(t) for t in self.trajectories]
        traj_returns = [t.total_reward for t in self.trajectories]

        return {
            "num_trajectories": len(self.trajectories),
            "num_transitions": len(all_transitions),
            "mean_trajectory_length": np.mean(traj_lengths) if traj_lengths else 0,
            "std_trajectory_length": np.std(traj_lengths) if traj_lengths else 0,
            "mean_reward": np.mean(all_rewards) if all_rewards else 0,
            "std_reward": np.std(all_rewards) if all_rewards else 0,
            "min_reward": np.min(all_rewards) if all_rewards else 0,
            "max_reward": np.max(all_rewards) if all_rewards else 0,
            "mean_return": np.mean(traj_returns) if traj_returns else 0,
            "std_return": np.std(traj_returns) if traj_returns else 0,
        }

    def save(self, path: str):
        """Save dataset to JSON file"""
        data = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "statistics": self.compute_statistics(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "TrajectoryDataset":
        """Load dataset from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        trajectories = [
            Trajectory.from_dict(t_dict)
            for t_dict in data.get("trajectories", [])
        ]
        return cls(trajectories)


class DataPipeline:
    """
    Pipeline for extracting training data from historical interactions.

    Integrates:
    - TD-BKT for state computation
    - HLR for reward computation
    - Database queries for raw data extraction
    """

    def __init__(
        self,
        td_bkt: Any = None,
        reward_calculator: Any = None,
        concept_order: Optional[List[str]] = None
    ):
        """
        Initialize data pipeline.

        Args:
            td_bkt: TemporalDifferenceBKT instance
            reward_calculator: RewardCalculator instance
            concept_order: Ordered list of concept IDs
        """
        self.td_bkt = td_bkt
        self.reward_calculator = reward_calculator
        self.concept_order = concept_order or []

        # Concept ID to index mapping
        self._concept_to_idx: Dict[str, int] = {}
        if concept_order:
            self._concept_to_idx = {c: i for i, c in enumerate(concept_order)}

    def set_concept_order(self, concept_order: List[str]):
        """Set the concept ordering"""
        self.concept_order = concept_order
        self._concept_to_idx = {c: i for i, c in enumerate(concept_order)}

    def concept_to_action(self, concept_id: str) -> int:
        """Convert concept ID to action index"""
        return self._concept_to_idx.get(concept_id, -1)

    def action_to_concept(self, action: int) -> str:
        """Convert action index to concept ID"""
        if 0 <= action < len(self.concept_order):
            return self.concept_order[action]
        return ""

    def process_interaction_sequence(
        self,
        user_id: str,
        interactions: List[Dict],
        course_id: Optional[int] = None
    ) -> Trajectory:
        """
        Process a sequence of interactions into a trajectory.

        Args:
            user_id: User identifier
            interactions: List of interaction dicts with keys:
                - concept_id: str
                - correct: bool
                - timestamp: datetime or str
                - Optional: response_time_ms, rating, etc.
            course_id: Optional course identifier

        Returns:
            Trajectory with computed states and rewards
        """
        # Import here to avoid circular imports
        from ..td_bkt import TemporalDifferenceBKT, TDBKTConfig, BeliefState
        from ..hlr import RewardCalculator, HLRConfig

        # Initialize TD-BKT if not provided
        if self.td_bkt is None:
            self.td_bkt = TemporalDifferenceBKT(TDBKTConfig())

        # Initialize reward calculator if not provided
        if self.reward_calculator is None:
            self.reward_calculator = RewardCalculator(HLRConfig())

        # Create belief state for user
        belief_state = self.td_bkt.create_belief_state(
            user_id=user_id,
            concept_ids=self.concept_order
        )

        # Initialize reward calculator state
        self.reward_calculator.initialize_state(
            concept_ids=self.concept_order
        )

        # Create trajectory
        trajectory = Trajectory(user_id=user_id, course_id=course_id)

        # Sort interactions by timestamp
        sorted_interactions = sorted(
            interactions,
            key=lambda x: x["timestamp"] if isinstance(x["timestamp"], datetime)
            else datetime.fromisoformat(x["timestamp"])
        )

        prev_state = belief_state.to_vector()

        for interaction in sorted_interactions:
            concept_id = interaction["concept_id"]
            correct = interaction["correct"]
            timestamp = interaction["timestamp"]

            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # Get action index
            action = self.concept_to_action(concept_id)
            if action < 0:
                logger.warning(f"Unknown concept {concept_id}, skipping")
                continue

            # Update belief state
            belief_state, _ = self.td_bkt.update(
                belief_state=belief_state,
                concept_id=concept_id,
                correct=correct,
                timestamp=timestamp
            )

            # Calculate reward
            reward, _ = self.reward_calculator.calculate_reward(
                concept_id=concept_id,
                correct=correct,
                current_time=timestamp,
                all_concept_ids=self.concept_order
            )

            # Get new state
            new_state = belief_state.to_vector()

            # Create transition
            transition = Transition(
                state=prev_state.copy(),
                action=action,
                reward=reward,
                next_state=new_state.copy(),
                done=False,
                user_id=user_id,
                concept_id=concept_id,
                timestamp=timestamp,
                correct=correct
            )

            trajectory.add_transition(transition)
            prev_state = new_state

        # Mark last transition as done
        if trajectory.transitions:
            trajectory.transitions[-1].done = True

        # Compute returns-to-go
        trajectory.compute_returns_to_go()

        return trajectory

    async def extract_from_database(
        self,
        db_session: Any,
        course_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_interactions: int = 5
    ) -> TrajectoryDataset:
        """
        Extract training data from database.

        Args:
            db_session: Database session
            course_id: Course to extract data for
            start_date: Optional start date filter
            end_date: Optional end date filter
            min_interactions: Minimum interactions per user

        Returns:
            TrajectoryDataset with extracted trajectories
        """
        # This would be implemented with actual database queries
        # Here's the structure:

        # 1. Query review logs
        # SELECT user_id, card_id, rating, review_time, ...
        # FROM review_logs
        # WHERE course_id = ? AND review_time BETWEEN ? AND ?
        # ORDER BY user_id, review_time

        # 2. Query card -> concept mapping
        # SELECT id, concept_id FROM spaced_repetition_cards WHERE course_id = ?

        # 3. Group by user and process

        raise NotImplementedError(
            "Database extraction requires actual database session. "
            "Use process_interaction_sequence for in-memory processing."
        )

    def create_synthetic_dataset(
        self,
        num_users: int = 100,
        interactions_per_user: int = 50,
        num_concepts: int = 20,
        seed: Optional[int] = None
    ) -> TrajectoryDataset:
        """
        Create synthetic dataset for pre-training.

        Generates realistic student interaction patterns:
        - Varied learning rates
        - Forgetting dynamics
        - Prerequisite relationships

        Args:
            num_users: Number of simulated users
            interactions_per_user: Interactions per user
            num_concepts: Number of concepts
            seed: Random seed

        Returns:
            Synthetic TrajectoryDataset
        """
        if seed is not None:
            np.random.seed(seed)

        # Setup concept order if not set
        if not self.concept_order:
            self.concept_order = [f"concept_{i}" for i in range(num_concepts)]
            self._concept_to_idx = {c: i for i, c in enumerate(self.concept_order)}

        dataset = TrajectoryDataset()

        for user_idx in range(num_users):
            user_id = f"synthetic_user_{user_idx}"

            # Random user characteristics
            learning_rate = np.random.uniform(0.1, 0.3)
            forgetting_rate = np.random.uniform(0.02, 0.1)
            initial_knowledge = np.random.uniform(0.0, 0.2, num_concepts)

            # Simulate interactions
            interactions = []
            current_time = datetime.now() - timedelta(days=30)
            mastery = initial_knowledge.copy()

            for step in range(interactions_per_user):
                # Choose concept (biased toward lower mastery)
                probs = 1 - mastery + 0.1
                probs = probs / probs.sum()
                concept_idx = np.random.choice(num_concepts, p=probs)
                concept_id = self.concept_order[concept_idx]

                # Simulate response
                p_correct = mastery[concept_idx] * 0.9 + 0.1  # Min 10% guess rate
                correct = np.random.random() < p_correct

                # Add interaction
                interactions.append({
                    "concept_id": concept_id,
                    "correct": correct,
                    "timestamp": current_time,
                })

                # Update mastery
                if correct:
                    mastery[concept_idx] = min(1.0, mastery[concept_idx] + learning_rate)
                else:
                    mastery[concept_idx] = max(0.0, mastery[concept_idx] - 0.05)

                # Apply forgetting to all concepts
                mastery = mastery * (1 - forgetting_rate)
                mastery[concept_idx] = min(1.0, mastery[concept_idx] + 0.05)  # Recency boost

                # Advance time
                current_time += timedelta(hours=np.random.exponential(4))

            # Process into trajectory
            trajectory = self.process_interaction_sequence(
                user_id=user_id,
                interactions=interactions
            )
            dataset.add_trajectory(trajectory)

        logger.info(
            f"Created synthetic dataset: {num_users} users, "
            f"{dataset.num_transitions()} transitions"
        )

        return dataset
