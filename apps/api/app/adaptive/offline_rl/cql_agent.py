"""
Conservative Q-Learning (CQL) for Curriculum RL

A value-based offline RL algorithm that addresses overestimation bias
by adding a conservative regularization term to the Q-learning objective.

Key Innovation:
CQL penalizes Q-values for out-of-distribution actions, ensuring
the agent stays close to the behavior policy (historical data).
This makes it safe for educational deployment.

Loss Function:
L_CQL = α * (logsumexp(Q(s,·)) - Q(s, a_data)) + (1/2) * ||Q - (r + γ·max_a' Q')||²

Where:
- First term: Conservative regularization (push down OOD actions)
- Second term: Standard Bellman error
- α: Conservatism coefficient (higher = safer)

Why CQL for Education:
- Safe: Doesn't try untested action sequences
- Data-efficient: Learns from historical student data
- Generalizable: Can improve over behavioral policy
- Robust: Handles noisy educational data

References:
- Kumar et al. (2020): Conservative Q-Learning for Offline RL
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. CQLAgent will not work.")


@dataclass
class CQLConfig:
    """Configuration for CQL Agent"""

    # Network architecture
    state_dim: int = 60              # Belief state dimension
    action_dim: int = 20             # Number of concepts
    hidden_dims: List[int] = None    # Hidden layer dimensions

    # CQL parameters
    alpha: float = 2.0               # Conservatism coefficient
    gamma: float = 0.99              # Discount factor
    tau: float = 0.005               # Target network soft update rate

    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000

    # Inference parameters
    temperature: float = 0.1         # Boltzmann exploration temperature

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

    def to_dict(self) -> Dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "tau": self.tau,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CQLConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CQLConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """
        Q-Network for estimating state-action values.

        Architecture: MLP with ReLU activations
        Input: State (belief vector)
        Output: Q-values for all actions
        """

        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()

            layers = []
            prev_dim = state_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, action_dim))

            self.network = nn.Sequential(*layers)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.zeros_(module.bias)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Compute Q-values for all actions.

            Args:
                state: State tensor (batch, state_dim)

            Returns:
                Q-values (batch, action_dim)
            """
            return self.network(state)

    class CQLAgent:
        """
        Conservative Q-Learning Agent for curriculum optimization.

        Learns a Q-function from offline data with conservative regularization
        to avoid overestimating values of out-of-distribution actions.
        """

        def __init__(
            self,
            config: CQLConfig,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        ):
            """
            Initialize CQL Agent.

            Args:
                config: Agent configuration
                device: Compute device
            """
            self.config = config
            self.device = device

            # Q-networks
            self.q_network = QNetwork(
                config.state_dim,
                config.action_dim,
                config.hidden_dims
            ).to(device)

            self.target_q_network = QNetwork(
                config.state_dim,
                config.action_dim,
                config.hidden_dims
            ).to(device)

            # Copy weights to target
            self.target_q_network.load_state_dict(self.q_network.state_dict())

            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=config.learning_rate
            )

            # Training state
            self.training_step = 0

        def get_q_values(
            self,
            states: torch.Tensor,
            use_target: bool = False
        ) -> torch.Tensor:
            """
            Get Q-values for states.

            Args:
                states: State tensor (batch, state_dim)
                use_target: If True, use target network

            Returns:
                Q-values (batch, action_dim)
            """
            network = self.target_q_network if use_target else self.q_network
            return network(states)

        def select_action(
            self,
            state: np.ndarray,
            action_mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
            temperature: Optional[float] = None
        ) -> Tuple[int, np.ndarray]:
            """
            Select action using Q-values.

            Args:
                state: State vector (state_dim,)
                action_mask: Valid action mask (action_dim,)
                deterministic: If True, return greedy action
                temperature: Boltzmann temperature

            Returns:
                (selected_action, q_values)
            """
            temperature = temperature or self.config.temperature

            # Convert to tensor
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Get Q-values
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)

            q_values_np = q_values.cpu().numpy()

            # Apply action mask
            if action_mask is not None:
                q_values_np = q_values_np + (1 - action_mask) * (-1e9)

            if deterministic:
                action = int(np.argmax(q_values_np))
            else:
                # Boltzmann exploration
                scaled_q = q_values_np / temperature
                # Stable softmax
                max_q = np.max(scaled_q)
                exp_q = np.exp(scaled_q - max_q)
                probs = exp_q / np.sum(exp_q)

                # Handle numerical issues
                probs = np.clip(probs, 1e-10, 1.0)
                probs = probs / probs.sum()

                action = int(np.random.choice(len(probs), p=probs))

            return action, q_values_np

        def compute_cql_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Compute CQL loss.

            L_CQL = α * (logsumexp(Q(s,·)) - Q(s,a)) + (1/2) * Bellman_error²

            Args:
                states: Current states (batch, state_dim)
                actions: Actions taken (batch,)
                rewards: Rewards received (batch,)
                next_states: Next states (batch, state_dim)
                dones: Done flags (batch,)
                action_mask: Optional action mask (batch, action_dim)

            Returns:
                (total_loss, loss_components)
            """
            batch_size = states.shape[0]

            # Get current Q-values
            current_q_values = self.q_network(states)  # (batch, action_dim)

            # Get Q-value for taken action
            current_q = current_q_values.gather(
                1, actions.unsqueeze(1).long()
            ).squeeze(1)  # (batch,)

            # Compute target Q-value
            with torch.no_grad():
                next_q_values = self.target_q_network(next_states)

                # Apply action mask if provided
                if action_mask is not None:
                    next_q_values = next_q_values + (1 - action_mask) * (-1e9)

                max_next_q = next_q_values.max(dim=1)[0]
                target_q = rewards + (1 - dones) * self.config.gamma * max_next_q

            # Bellman error
            bellman_error = F.mse_loss(current_q, target_q)

            # CQL conservative regularization
            # Term 1: Push down Q-values for all actions
            logsumexp_q = torch.logsumexp(current_q_values, dim=1).mean()

            # Term 2: Push up Q-value for actions in dataset
            dataset_q = current_q.mean()

            # CQL regularization: logsumexp - dataset
            cql_regularization = logsumexp_q - dataset_q

            # Total loss
            total_loss = (
                self.config.alpha * cql_regularization +
                0.5 * bellman_error
            )

            loss_components = {
                "total_loss": total_loss.item(),
                "bellman_error": bellman_error.item(),
                "cql_regularization": cql_regularization.item(),
                "logsumexp_q": logsumexp_q.item(),
                "dataset_q": dataset_q.item(),
                "mean_q": current_q.mean().item(),
            }

            return total_loss, loss_components

        def train_step(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            dones: np.ndarray,
            action_mask: Optional[np.ndarray] = None
        ) -> Dict:
            """
            Single training step.

            Args:
                states: States (batch, state_dim)
                actions: Actions (batch,)
                rewards: Rewards (batch,)
                next_states: Next states (batch, state_dim)
                dones: Done flags (batch,)
                action_mask: Optional action mask

            Returns:
                Loss components dictionary
            """
            # Convert to tensors
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

            mask_t = None
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)

            # Compute loss
            loss, loss_info = self.compute_cql_loss(
                states_t, actions_t, rewards_t, next_states_t, dones_t, mask_t
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            # Soft update target network
            self._soft_update_target()

            self.training_step += 1
            return loss_info

        def _soft_update_target(self):
            """Soft update target network parameters"""
            for target_param, param in zip(
                self.target_q_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * param.data +
                    (1 - self.config.tau) * target_param.data
                )

        def train(
            self,
            dataset,  # TrajectoryDataset
            num_steps: int,
            log_interval: int = 100,
            save_interval: int = 5000,
            save_path: str = "cql_checkpoint.pt"
        ) -> Dict[str, List[float]]:
            """
            Full training loop.

            Args:
                dataset: TrajectoryDataset with training data
                num_steps: Number of training steps
                log_interval: Steps between logging
                save_interval: Steps between checkpoints
                save_path: Path for saving checkpoints

            Returns:
                Dictionary with training history
            """
            history = {
                "loss": [],
                "bellman_error": [],
                "cql_reg": [],
                "mean_q": [],
            }

            for step in range(num_steps):
                # Sample batch
                states, actions, rewards, next_states, dones = \
                    dataset.get_transition_batch(self.config.batch_size)

                # Training step
                loss_info = self.train_step(
                    states, actions, rewards, next_states, dones
                )

                # Record history
                history["loss"].append(loss_info["total_loss"])
                history["bellman_error"].append(loss_info["bellman_error"])
                history["cql_reg"].append(loss_info["cql_regularization"])
                history["mean_q"].append(loss_info["mean_q"])

                # Logging
                if step % log_interval == 0:
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Loss: {loss_info['total_loss']:.4f} | "
                        f"Bellman: {loss_info['bellman_error']:.4f} | "
                        f"CQL: {loss_info['cql_regularization']:.4f} | "
                        f"Q: {loss_info['mean_q']:.2f}"
                    )

                # Save checkpoint
                if step > 0 and step % save_interval == 0:
                    self.save(save_path.replace('.pt', f'_step{step}.pt'))

            # Save final model
            self.save(save_path)

            return history

        def save(self, path: str):
            """Save model checkpoint"""
            checkpoint = {
                'config': self.config.to_dict(),
                'q_network': self.q_network.state_dict(),
                'target_q_network': self.target_q_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step,
            }
            torch.save(checkpoint, path)
            logger.info(f"Saved CQL checkpoint to {path}")

        @classmethod
        def load(cls, path: str, device: str = 'cpu') -> "CQLAgent":
            """Load model from checkpoint"""
            checkpoint = torch.load(path, map_location=device)
            config = CQLConfig.from_dict(checkpoint['config'])
            agent = cls(config, device)
            agent.q_network.load_state_dict(checkpoint['q_network'])
            agent.target_q_network.load_state_dict(checkpoint['target_q_network'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            agent.training_step = checkpoint['training_step']
            logger.info(f"Loaded CQL checkpoint from {path}")
            return agent

        def export_weights(self, path: str):
            """Export Q-network weights for lightweight inference"""
            weights = {}
            for name, param in self.q_network.named_parameters():
                weights[name] = param.detach().cpu().numpy()
            np.savez(path, **weights)
            logger.info(f"Exported CQL weights to {path}")

else:
    # Stub classes when PyTorch not available
    class QNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for QNetwork")

    class CQLAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for CQLAgent")
