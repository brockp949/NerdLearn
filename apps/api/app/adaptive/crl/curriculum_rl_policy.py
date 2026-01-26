"""
Curriculum Reinforcement Learning Policy Service

Unified policy interface that integrates:
- TD-BKT for state estimation
- HLR for reward tracking
- Knowledge Graph Action Masking for constraints
- Decision Transformer or CQL for action selection

This is the main entry point for using the CRL system in production.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# Import components
from ..td_bkt import TemporalDifferenceBKT, TDBKTConfig, BeliefState
from ..hlr import HLRModel, HLRConfig, RewardCalculator
from ..kg_mask import ActionMasker, ActionMaskerConfig, PrerequisiteGraph
from ..offline_rl import DTLite, DTLiteConfig

# Conditional PyTorch imports
try:
    from ..offline_rl import DecisionTransformer, DecisionTransformerConfig
    from ..offline_rl import CQLAgent, CQLConfig
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class PolicyType(str, Enum):
    """Available policy types"""
    DECISION_TRANSFORMER = "decision_transformer"  # PyTorch DT
    DT_LITE = "dt_lite"                           # Lightweight DT
    CQL = "cql"                                    # Conservative Q-Learning
    ZPD_BASELINE = "zpd_baseline"                  # ZPD recommender (baseline)
    RANDOM = "random"                              # Random (for testing)


@dataclass
class PolicyConfig:
    """Configuration for CRL Policy"""

    # Policy selection
    policy_type: PolicyType = PolicyType.DT_LITE

    # Model paths
    dt_weights_path: Optional[str] = None
    dt_config_path: Optional[str] = None
    cql_checkpoint_path: Optional[str] = None

    # Component configs (optional - uses defaults if not provided)
    td_bkt_config: Optional[Dict] = None
    hlr_config: Optional[Dict] = None
    action_masker_config: Optional[Dict] = None

    # Inference parameters
    temperature: float = 0.1
    deterministic: bool = False
    target_return: float = 10.0

    # Fallback behavior
    fallback_to_zpd: bool = True  # Fall back to ZPD if RL fails
    confidence_threshold: float = 0.3  # Min confidence to use RL action

    def to_dict(self) -> Dict:
        return {
            "policy_type": self.policy_type.value,
            "dt_weights_path": self.dt_weights_path,
            "dt_config_path": self.dt_config_path,
            "cql_checkpoint_path": self.cql_checkpoint_path,
            "td_bkt_config": self.td_bkt_config,
            "hlr_config": self.hlr_config,
            "action_masker_config": self.action_masker_config,
            "temperature": self.temperature,
            "deterministic": self.deterministic,
            "target_return": self.target_return,
            "fallback_to_zpd": self.fallback_to_zpd,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PolicyConfig":
        d = d.copy()
        if "policy_type" in d:
            d["policy_type"] = PolicyType(d["policy_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolicyConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class CurriculumRLPolicy:
    """
    Main policy class for Curriculum Reinforcement Learning.

    Integrates all components:
    - TD-BKT: Tracks student knowledge state with temporal dynamics
    - HLR: Provides memory strength estimates for reward calculation
    - Action Masker: Ensures prerequisite constraints are respected
    - RL Model: Decision Transformer or CQL for optimal concept selection

    Usage:
        # Initialize
        policy = CurriculumRLPolicy(config)
        policy.initialize(user_id, concept_ids, prerequisite_graph)

        # During learning session
        for interaction in session:
            # Get next concept recommendation
            concept_id, info = policy.select_next_concept()

            # After student responds
            policy.update(concept_id, correct, timestamp)
    """

    def __init__(self, config: PolicyConfig):
        """
        Initialize CRL Policy.

        Args:
            config: Policy configuration
        """
        print(f"DEBUG: Initializing CurriculumRLPolicy with config: {config}")
        self.config = config

        # Initialize components
        td_bkt_cfg = TDBKTConfig.from_dict(config.td_bkt_config or {})
        self.td_bkt = TemporalDifferenceBKT(td_bkt_cfg)

        hlr_cfg = HLRConfig.from_dict(config.hlr_config or {})
        self.hlr_model = HLRModel(hlr_cfg)
        self.reward_calculator = RewardCalculator(hlr_cfg, self.hlr_model)

        masker_cfg = ActionMaskerConfig.from_dict(config.action_masker_config or {})
        self.action_masker = ActionMasker(masker_cfg)

        # RL model (loaded lazily)
        self._rl_model: Optional[Union[DTLite, Any]] = None

        # State
        self.belief_state: Optional[BeliefState] = None
        self.concept_order: List[str] = []
        self.concept_to_idx: Dict[str, int] = {}
        self.current_user_id: Optional[str] = None

        # Session tracking
        self.session_history: List[Dict] = []
        self.total_reward: float = 0.0

        # Load RL model eagerly
        self._load_rl_model()

    def _load_rl_model(self):
        """Lazily load the RL model"""
        print(f"DEBUG: Loading RL model. Type: {self.config.policy_type}")
        print(f"DEBUG: Paths: {self.config.dt_weights_path}, {self.config.dt_config_path}")
        
        if self._rl_model is not None:
            return

        if self.config.policy_type == PolicyType.DT_LITE:
            if self.config.dt_weights_path and self.config.dt_config_path:
                try:
                    dt_config = DTLiteConfig.load(self.config.dt_config_path)
                    self._rl_model = DTLite.load(self.config.dt_weights_path, dt_config)
                    logger.info("Loaded DT-Lite model")
                    print("DEBUG: Successfully loaded DT-Lite model")
                except Exception as e:
                    logger.error(f"Failed to load DT-Lite model: {e}")
                    print(f"DEBUG: Failed to load DT-Lite model: {e}")
            else:
                logger.warning("DT-Lite paths not configured, using random fallback")
                print("DEBUG: Paths not configured")

        elif self.config.policy_type == PolicyType.DECISION_TRANSFORMER:
            if not PYTORCH_AVAILABLE:
                logger.warning("PyTorch not available, falling back to DT-Lite")
                self.config.policy_type = PolicyType.DT_LITE
                return self._load_rl_model()

            if self.config.dt_weights_path:
                self._rl_model = DecisionTransformer.load(self.config.dt_weights_path)
                logger.info("Loaded Decision Transformer model")

        elif self.config.policy_type == PolicyType.CQL:
            if not PYTORCH_AVAILABLE:
                logger.warning("PyTorch not available, falling back to DT-Lite")
                self.config.policy_type = PolicyType.DT_LITE
                return self._load_rl_model()

            if self.config.cql_checkpoint_path:
                self._rl_model = CQLAgent.load(self.config.cql_checkpoint_path)
                logger.info("Loaded CQL model")

    def initialize(
        self,
        user_id: str,
        concept_ids: List[str],
        prerequisite_graph: Optional[PrerequisiteGraph] = None,
        initial_masteries: Optional[Dict[str, float]] = None
    ):
        """
        Initialize policy for a user and curriculum.

        Args:
            user_id: User identifier
            concept_ids: Ordered list of concept IDs
            prerequisite_graph: Optional prerequisite graph
            initial_masteries: Optional initial mastery values
        """
        self.current_user_id = user_id
        self.concept_order = concept_ids
        self.concept_to_idx = {c: i for i, c in enumerate(concept_ids)}

        # Initialize belief state
        self.belief_state = self.td_bkt.create_belief_state(
            user_id=user_id,
            concept_ids=concept_ids,
            initial_masteries=initial_masteries
        )

        # Initialize reward calculator
        self.reward_calculator.initialize_state(concept_ids)

        # Set up action masker
        if prerequisite_graph:
            self.action_masker.set_prerequisite_graph(prerequisite_graph)
        else:
            # Create empty graph if not provided
            graph = PrerequisiteGraph()
            for cid in concept_ids:
                graph.add_concept(cid)
            self.action_masker.set_prerequisite_graph(graph)

        # Load RL model
        self._load_rl_model()

        # Reset DT-Lite context if using it
        if isinstance(self._rl_model, DTLite):
            self._rl_model.reset_context()

        # Reset session
        self.session_history = []
        self.total_reward = 0.0

        logger.info(f"Initialized CRL policy for user {user_id} with {len(concept_ids)} concepts")

    def select_next_concept(
        self,
        current_time: Optional[datetime] = None,
        target_return: Optional[float] = None
    ) -> Tuple[str, Dict]:
        """
        Select the next concept to practice.

        Args:
            current_time: Current timestamp
            target_return: Desired return (for DT models)

        Returns:
            (concept_id, selection_info)
        """
        if self.belief_state is None:
            raise RuntimeError("Policy not initialized. Call initialize() first.")

        current_time = current_time or datetime.now()
        target_return = target_return or self.config.target_return

        # Get belief state vector
        state_vector = self.belief_state.to_vector(current_time)

        # Get mastery dict for action masking
        masteries = {
            cid: self.belief_state.get_concept_mastery(cid)
            for cid in self.concept_order
        }

        # Compute action mask
        action_mask = self.action_masker.compute_mask(masteries, self.concept_order)

        # Select action based on policy type
        if self.config.policy_type == PolicyType.RANDOM:
            action, probs = self._select_random(action_mask)

        elif self.config.policy_type == PolicyType.ZPD_BASELINE:
            action, probs = self._select_zpd_baseline(masteries, action_mask)

        elif self.config.policy_type in [PolicyType.DT_LITE, PolicyType.DECISION_TRANSFORMER]:
            action, probs = self._select_dt(state_vector, action_mask, target_return)

        elif self.config.policy_type == PolicyType.CQL:
            action, probs = self._select_cql(state_vector, action_mask)

        else:
            action, probs = self._select_random(action_mask)

        # Map action to concept
        concept_id = self.concept_order[action]

        # Compile selection info
        info = {
            "action": action,
            "concept_id": concept_id,
            "policy_type": self.config.policy_type.value,
            "action_probabilities": probs.tolist() if probs is not None else None,
            "action_mask": action_mask.tolist(),
            "valid_actions": int(action_mask.sum()),
            "mastery": masteries.get(concept_id, 0.0),
            "urgency": self.reward_calculator.get_urgency_scores(
                [concept_id], current_time
            ).get(concept_id, 0.0),
        }

        return concept_id, info

    def _select_random(self, action_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """Random action selection (respecting mask)"""
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            valid_actions = np.arange(len(action_mask))

        action = int(np.random.choice(valid_actions))
        probs = np.zeros(len(action_mask))
        probs[valid_actions] = 1.0 / len(valid_actions)

        return action, probs

    def _select_zpd_baseline(
        self,
        masteries: Dict[str, float],
        action_mask: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        ZPD-based selection (baseline).

        Selects concepts in the Zone of Proximal Development:
        - Not too easy (mastery > 0.3)
        - Not too hard (mastery < 0.9)
        - Prerequisites met (from mask)
        """
        scores = np.zeros(len(self.concept_order))

        for i, concept_id in enumerate(self.concept_order):
            if action_mask[i] < 0.5:
                scores[i] = -np.inf
                continue

            mastery = masteries.get(concept_id, 0.1)

            # ZPD score: peak at 0.5 mastery
            zpd_score = 1 - abs(mastery - 0.5) * 2

            # Urgency bonus
            urgency = self.reward_calculator.get_urgency_scores(
                [concept_id]
            ).get(concept_id, 0.0)

            scores[i] = zpd_score + 0.3 * urgency

        # Softmax over valid actions
        valid_mask = scores > -np.inf
        if not valid_mask.any():
            return self._select_random(action_mask)

        probs = np.zeros_like(scores)
        valid_scores = scores[valid_mask]
        valid_probs = np.exp(valid_scores - valid_scores.max())
        valid_probs /= valid_probs.sum()
        probs[valid_mask] = valid_probs

        if self.config.deterministic:
            action = int(np.argmax(scores))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        return action, probs

    def _select_dt(
        self,
        state_vector: np.ndarray,
        action_mask: np.ndarray,
        target_return: float
    ) -> Tuple[int, np.ndarray]:
        """Decision Transformer action selection"""
        if self._rl_model is None:
            logger.warning("DT model not loaded, falling back to ZPD")
            masteries = {
                cid: self.belief_state.get_concept_mastery(cid)
                for cid in self.concept_order
            }
            return self._select_zpd_baseline(masteries, action_mask)

        if isinstance(self._rl_model, DTLite):
            action, probs = self._rl_model.select_action(
                state=state_vector,
                target_return=target_return,
                action_mask=action_mask,
                deterministic=self.config.deterministic
            )
        else:
            # PyTorch Decision Transformer
            import torch

            # Get context from DT-Lite style buffer or build fresh
            # For simplicity, use single-step inference
            rtg = torch.tensor([[[target_return]]], dtype=torch.float32)
            states = torch.tensor([[state_vector]], dtype=torch.float32)
            actions = torch.tensor([[0]], dtype=torch.long)  # Placeholder
            timesteps = torch.tensor([[0]], dtype=torch.long)

            action, probs = self._rl_model.get_action(
                rtg, states, actions, timesteps,
                action_mask=action_mask,
                temperature=self.config.temperature,
                deterministic=self.config.deterministic
            )
            probs = probs.cpu().numpy()

        return action, probs

    def _select_cql(
        self,
        state_vector: np.ndarray,
        action_mask: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """CQL action selection"""
        if self._rl_model is None:
            logger.warning("CQL model not loaded, falling back to ZPD")
            masteries = {
                cid: self.belief_state.get_concept_mastery(cid)
                for cid in self.concept_order
            }
            return self._select_zpd_baseline(masteries, action_mask)

        action, q_values = self._rl_model.select_action(
            state=state_vector,
            action_mask=action_mask,
            deterministic=self.config.deterministic,
            temperature=self.config.temperature
        )

        # Convert Q-values to probabilities for logging
        masked_q = q_values + (1 - action_mask) * (-1e9)
        probs = np.exp(masked_q - masked_q.max())
        probs = probs / probs.sum()

        return action, probs

    def update(
        self,
        concept_id: str,
        correct: bool,
        timestamp: Optional[datetime] = None,
        response_time_ms: Optional[int] = None
    ) -> Dict:
        """
        Update policy state after an observation.

        Args:
            concept_id: Concept that was practiced
            correct: Whether response was correct
            timestamp: Time of observation
            response_time_ms: Optional response time

        Returns:
            Update details
        """
        if self.belief_state is None:
            raise RuntimeError("Policy not initialized. Call initialize() first.")

        timestamp = timestamp or datetime.now()

        # Update TD-BKT belief state
        self.belief_state, bkt_details = self.td_bkt.update(
            belief_state=self.belief_state,
            concept_id=concept_id,
            correct=correct,
            timestamp=timestamp,
            response_time_ms=response_time_ms
        )

        # Calculate reward
        reward, reward_details = self.reward_calculator.calculate_reward(
            concept_id=concept_id,
            correct=correct,
            current_time=timestamp,
            all_concept_ids=self.concept_order
        )

        self.total_reward += reward

        # Update DT-Lite context if using it
        if isinstance(self._rl_model, DTLite):
            action = self.concept_to_idx.get(concept_id, 0)
            self._rl_model.update_action(action)

        # Record in session history
        history_entry = {
            "concept_id": concept_id,
            "correct": correct,
            "timestamp": timestamp.isoformat(),
            "reward": reward,
            "mastery_after": self.belief_state.get_concept_mastery(concept_id),
        }
        self.session_history.append(history_entry)

        # Compile update details
        update_details = {
            "concept_id": concept_id,
            "correct": correct,
            "reward": reward,
            "total_reward": self.total_reward,
            "bkt_update": bkt_details,
            "reward_details": reward_details,
            "session_length": len(self.session_history),
        }

        return update_details

    def get_session_summary(self) -> Dict:
        """Get summary of current session"""
        if not self.session_history:
            return {"message": "No session data"}

        correct_count = sum(1 for h in self.session_history if h["correct"])

        return {
            "user_id": self.current_user_id,
            "num_interactions": len(self.session_history),
            "correct_count": correct_count,
            "accuracy": correct_count / len(self.session_history),
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / len(self.session_history),
            "concepts_practiced": list(set(h["concept_id"] for h in self.session_history)),
            "interleaving_score": self._compute_interleaving_score(),
        }

    def _compute_interleaving_score(self) -> float:
        """
        Compute interleaving score (entropy of concept sequence).

        Higher score = more interleaved practice
        Lower score = more blocked practice
        """
        if len(self.session_history) < 2:
            return 0.0

        # Count consecutive same-concept pairs
        concept_sequence = [h["concept_id"] for h in self.session_history]
        switches = sum(
            1 for i in range(1, len(concept_sequence))
            if concept_sequence[i] != concept_sequence[i-1]
        )

        # Normalize by maximum possible switches
        max_switches = len(concept_sequence) - 1
        if max_switches == 0:
            return 0.0

        return switches / max_switches

    def get_belief_state(self) -> Optional[BeliefState]:
        """Get current belief state"""
        return self.belief_state

    def get_concept_recommendations(
        self,
        n: int = 5,
        current_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get top N concept recommendations with details.

        Args:
            n: Number of recommendations
            current_time: Current time

        Returns:
            List of recommendation dicts
        """
        if self.belief_state is None:
            return []

        current_time = current_time or datetime.now()

        # Get masteries and action mask
        masteries = {
            cid: self.belief_state.get_concept_mastery(cid)
            for cid in self.concept_order
        }
        action_mask = self.action_masker.compute_mask(masteries, self.concept_order)

        # Get urgencies
        urgencies = self.reward_calculator.get_urgency_scores(
            self.concept_order, current_time
        )

        # Score all concepts
        recommendations = []
        for i, concept_id in enumerate(self.concept_order):
            if action_mask[i] < 0.5:
                continue

            mastery = masteries.get(concept_id, 0.1)
            urgency = urgencies.get(concept_id, 0.0)

            # Combined score
            zpd_score = 1 - abs(mastery - 0.5) * 2
            score = zpd_score + 0.5 * urgency

            recommendations.append({
                "concept_id": concept_id,
                "score": score,
                "mastery": mastery,
                "urgency": urgency,
                "is_valid": True,
            })

        # Sort by score and return top N
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:n]

    @classmethod
    def load(cls, config_path: str) -> "CurriculumRLPolicy":
        """
        Load policy from configuration file.

        Args:
            config_path: Path to config JSON

        Returns:
            Loaded policy instance
        """
        config = PolicyConfig.load(config_path)
        return cls(config)
