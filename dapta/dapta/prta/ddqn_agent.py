"""
Double Deep Q-Network (DDQN) with:
  - GRU-based patient history encoder (addresses partial observability)
  - Dueling network head (V + A streams)
  - Prioritised Experience Replay (PER)

Architecture references:
  - DDQN: van Hasselt et al. (2016). AAAI-30.
  - Dueling DQN: Wang et al. (2016). ICML.
  - PER: Schaul et al. (2016). ICLR.
  - GRU for POMDP: Yu et al. (2021). ACM Computing Surveys.
"""

from __future__ import annotations
import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from dapta.dae.state_builder import STATE_DIM
from dapta.prta.action_space import N_ACTIONS
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


<<<<<<< Updated upstream
# Neural network: GRU encoder + Dueling Q head

class GRUDuelingQNetwork(nn.Module):
    """
    Q-network with GRU history encoder and Dueling streams.

    Input
    -----
    state      : (B, STATE_DIM)       — current patient state vector
    history    : (B, T, STATE_DIM+1)  — past (state, action) pairs, T steps

    Output
    ------
    q_values   : (B, N_ACTIONS)
    """
=======
class GRUDuelingQNetwork(nn.Module):
    """GRU-based Dueling DDQN — original DAPTA architecture."""
>>>>>>> Stashed changes

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.gru_hidden = gru_hidden

<<<<<<< Updated upstream
        # GRU encodes history of (state, action_index) pairs
        # Input per step: state_dim + 1 (action index as scalar)
=======
>>>>>>> Stashed changes
        self.gru = nn.GRU(
            input_size=state_dim + 1,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )

        # Shared feature layer: current state + GRU context
        shared_input = state_dim + gru_hidden
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_input, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
        )

        # Dueling streams
        fc_out = fc_hidden // 2
        self.value_stream = nn.Sequential(
            nn.Linear(fc_out, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc_out, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(
        self,
        state: "torch.Tensor",
        history: "torch.Tensor",
    ) -> "torch.Tensor":
<<<<<<< Updated upstream
        """
        Parameters
        ----------
        state   : (B, state_dim)
        history : (B, T, state_dim + 1)  — T can be 0 (first step)

        Returns
        -------
        q_values : (B, n_actions)
        """
        B = state.shape[0]

        if history.shape[1] > 0:
            _, h_n = self.gru(history)          # h_n: (gru_layers, B, gru_hidden)
            context = h_n[-1]                   # (B, gru_hidden) — last layer
        else:
            context = torch.zeros(B, self.gru_hidden, device=state.device)

        combined = torch.cat([state, context], dim=-1)  # (B, state_dim + gru_hidden)
        features = self.shared_fc(combined)              # (B, fc_hidden//2)

        # Dueling combination: Q = V + (A - mean(A))
        value = self.value_stream(features)             # (B, 1)
        advantage = self.advantage_stream(features)     # (B, n_actions)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
=======
        B = state.shape[0]

        if history.shape[1] > 0:
            _, h_n  = self.gru(history)
            context = h_n[-1]
        else:
            context = torch.zeros(B, self.gru_hidden, device=state.device)

        combined = torch.cat([state, context], dim=-1)
        features = self.shared_fc(combined)

        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DuelingQNetwork(nn.Module):
    """Standard Dueling DDQN without GRU — ablation comparison.

    Replaces the GRU context with a simple feedforward path so that
    the only difference from GRUDuelingQNetwork is the absence of
    sequential memory.  History tensors are accepted but ignored so
    that training loops can be shared between the two architectures.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.shared_fc = nn.Sequential(
            nn.Linear(state_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
        )

        fc_out = fc_hidden // 2
        self.value_stream = nn.Sequential(
            nn.Linear(fc_out, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc_out, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(
        self,
        state: "torch.Tensor",
        history: "torch.Tensor" = None,
    ) -> "torch.Tensor":
        features  = self.shared_fc(state)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
>>>>>>> Stashed changes



<<<<<<< Updated upstream
# Prioritised Replay Buffer

class PrioritisedReplayBuffer:
    """
    Prioritised Experience Replay (PER) buffer.
    Stores (state, history, action, reward, next_state, next_history, done).

    References
    ----------
    Schaul et al. (2016). Prioritized Experience Replay. ICLR.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self._buffer: deque = deque(maxlen=capacity)
=======
    def __init__(
        self,
        capacity: int  = 10_000,
        alpha: float   = 0.6,
        beta_start: float = 0.4,
        beta_end: float   = 1.0,
    ) -> None:
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta_start
        self.beta_end  = beta_end
        self._buffer:     deque = deque(maxlen=capacity)
>>>>>>> Stashed changes
        self._priorities: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        history: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_history: np.ndarray,
        done: bool,
    ) -> None:
        max_priority = max(self._priorities, default=1.0)
        self._buffer.append(
            (state, history, action, reward, next_state, next_history, done)
        )
        self._priorities.append(max_priority)

    def sample(
        self, batch_size: int
    ) -> Tuple[List, np.ndarray, np.ndarray]:
<<<<<<< Updated upstream
        """
        Returns
        -------
        (batch, indices, importance_sampling_weights)
        """
        probs = np.array(self._priorities, dtype=np.float64) ** self.alpha
=======
        probs  = np.array(self._priorities, dtype=np.float64) ** self.alpha
>>>>>>> Stashed changes
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), size=batch_size, p=probs)
        batch   = [self._buffer[i] for i in indices]

<<<<<<< Updated upstream
        # Importance sampling weights
        n = len(self._buffer)
=======
        n       = len(self._buffer)
>>>>>>> Stashed changes
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return batch, indices, weights.astype(np.float32)

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        for idx, td in zip(indices, td_errors):
            self._priorities[idx] = float(abs(td)) + 1e-5

    def anneal_beta(self, step: int, total_steps: int) -> None:
<<<<<<< Updated upstream
        """Linearly anneal beta from beta_start to 1.0."""
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta) * step / total_steps)
=======
        self.beta = min(
            self.beta_end,
            self.beta + (self.beta_end - self.beta) * step / total_steps,
        )
>>>>>>> Stashed changes

    def __len__(self) -> int:
        return len(self._buffer)


<<<<<<< Updated upstream
# DDQN Agent

class DDQNAgent:
    """
    Double DQN agent with GRU history encoder, Dueling head, and PER.

    Parameters
    ----------
    state_dim          : State vector dimension (default STATE_DIM=14)
    n_actions          : Number of actions (default N_ACTIONS=12)
    gru_hidden         : GRU hidden size
    gru_layers         : GRU layers
    learning_rate      : Adam learning rate
    gamma              : Discount factor
    tau                : Soft target network update coefficient
    target_update_freq : Hard target update interval (used if tau is None)
    buffer_size        : Replay buffer capacity
    batch_size         : Training batch size
    eps_start          : Initial exploration rate
    eps_end            : Final exploration rate
    eps_fraction       : Fraction of training over which eps decays
    history_len        : Number of past steps to include in history (T)
    device             : "cuda" | "cpu"
    checkpoint_path    : Where to save model
=======
class DDQNAgent:
    """Double DQN agent supporting both GRU and no-GRU architectures.

    Parameters
    ----------
    use_gru : bool
        If True (default) uses GRUDuelingQNetwork — the original DAPTA
        architecture.  If False uses DuelingQNetwork (no sequential
        memory) for the ablation study.
>>>>>>> Stashed changes
    """

    def __init__(
        self,
<<<<<<< Updated upstream
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        target_update_freq: int = 100,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_fraction: float = 0.3,
        history_len: int = 10,
        device: Optional[str] = None,
        checkpoint_path: Optional[str | Path] = None,
=======
        state_dim: int        = STATE_DIM,
        n_actions: int        = N_ACTIONS,
        gru_hidden: int       = 128,
        gru_layers: int       = 2,
        learning_rate: float  = 1e-4,
        gamma: float          = 0.95,
        tau: float            = 0.005,
        target_update_freq: int = 100,
        buffer_size: int      = 10_000,
        batch_size: int       = 64,
        eps_start: float      = 1.0,
        eps_end: float        = 0.05,
        eps_fraction: float   = 0.3,
        history_len: int      = 10,
        device: Optional[str] = None,
        checkpoint_path: Optional[str | Path] = None,
        use_gru: bool         = True,
>>>>>>> Stashed changes
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for DDQNAgent.")

        self.state_dim          = state_dim
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.tau                = tau
        self.target_update_freq = target_update_freq
        self.batch_size         = batch_size
        self.eps                = eps_start
        self.eps_end            = eps_end
        self.eps_fraction       = eps_fraction
        self.history_len        = history_len
        self.use_gru            = use_gru
        self.device             = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )

        if use_gru:
            self.online_net = GRUDuelingQNetwork(
                state_dim, n_actions, gru_hidden, gru_layers
            ).to(self.device)
            self.target_net = GRUDuelingQNetwork(
                state_dim, n_actions, gru_hidden, gru_layers
            ).to(self.device)
        else:
            self.online_net = DuelingQNetwork(
                state_dim, n_actions
            ).to(self.device)
            self.target_net = DuelingQNetwork(
                state_dim, n_actions
            ).to(self.device)

<<<<<<< Updated upstream
        # Networks
        self.online_net = GRUDuelingQNetwork(
            state_dim, n_actions, gru_hidden, gru_layers
        ).to(self.device)
        self.target_net = GRUDuelingQNetwork(
            state_dim, n_actions, gru_hidden, gru_layers
        ).to(self.device)
=======
>>>>>>> Stashed changes
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.replay_buffer = PrioritisedReplayBuffer(capacity=buffer_size)
        self._step = 0

<<<<<<< Updated upstream
    # Action selection

=======
>>>>>>> Stashed changes
    def select_action(
        self,
        state: np.ndarray,
        history: np.ndarray,
        greedy: bool = False,
    ) -> int:
<<<<<<< Updated upstream
        """
        ε-greedy action selection.

        Parameters
        ----------
        state   : (STATE_DIM,)
        history : (T, STATE_DIM+1) — past (state, action) sequence
        greedy  : If True, always select greedy action (evaluation mode)
        """
=======
>>>>>>> Stashed changes
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        self.online_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            if self.use_gru:
                h_t = torch.tensor(
                    history, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                q_values = self.online_net(s_t, h_t)
            else:
                q_values = self.online_net(s_t)

        return int(q_values.argmax(dim=1).item())

    
    # Training step
    

    def update(self, total_steps: int) -> Optional[float]:
<<<<<<< Updated upstream
        """
        Sample a minibatch and perform one gradient update.

        Returns
        -------
        float | None : TD loss (None if buffer too small)
        """
=======
>>>>>>> Stashed changes
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, indices, is_weights = self.replay_buffer.sample(
            self.batch_size
        )
        self.replay_buffer.anneal_beta(self._step, total_steps)

        (states, histories, actions, rewards,
         next_states, next_histories, dones) = zip(*batch)

        states      = torch.tensor(
            np.stack(states), dtype=torch.float32
        ).to(self.device)
        histories   = torch.tensor(
            np.stack(histories), dtype=torch.float32
        ).to(self.device)
        actions     = torch.tensor(
            actions, dtype=torch.long
        ).to(self.device)
        rewards     = torch.tensor(
            rewards, dtype=torch.float32
        ).to(self.device)
        next_states = torch.tensor(
            np.stack(next_states), dtype=torch.float32
        ).to(self.device)
        next_hist   = torch.tensor(
            np.stack(next_histories), dtype=torch.float32
        ).to(self.device)
        dones       = torch.tensor(
            dones, dtype=torch.float32
        ).to(self.device)
        is_weights  = torch.tensor(
            is_weights, dtype=torch.float32
        ).to(self.device)

        self.online_net.train()

        # Current Q values
<<<<<<< Updated upstream
        current_q = self.online_net(states, histories).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN target: online net selects action, target net evaluates it
        with torch.no_grad():
            online_next_q = self.online_net(next_states, next_hist)
            best_actions = online_next_q.argmax(dim=1)
            target_next_q = self.target_net(next_states, next_hist)
            target_q = rewards + self.gamma * (1 - dones) * target_next_q.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

        # Weighted Huber loss (PER importance sampling)
=======
        if self.use_gru:
            current_q = self.online_net(states, histories).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
        else:
            current_q = self.online_net(states).gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

        # Target Q values — Double DQN
        with torch.no_grad():
            if self.use_gru:
                online_next_q = self.online_net(next_states, next_hist)
                target_next_q = self.target_net(next_states, next_hist)
            else:
                online_next_q = self.online_net(next_states)
                target_next_q = self.target_net(next_states)

            best_actions = online_next_q.argmax(dim=1)
            target_q = rewards + self.gamma * (1 - dones) * (
                target_next_q.gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(1)
            )

>>>>>>> Stashed changes
        td_errors = (target_q - current_q).detach().cpu().numpy()
        loss = (
            is_weights * F.huber_loss(
                current_q, target_q, reduction="none"
            )
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=10.0
        )
        self.optimizer.step()

<<<<<<< Updated upstream
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Soft target update
=======
        self.replay_buffer.update_priorities(indices, td_errors)
>>>>>>> Stashed changes
        self._soft_update_target()

        self._step += 1
        return float(loss.item())

    
    # Epsilon decay
    

    def decay_epsilon(self, step: int, total_steps: int) -> None:
        """Linear epsilon decay."""
        decay_steps = int(total_steps * self.eps_fraction)
        if step < decay_steps:
            self.eps = 1.0 - (1.0 - self.eps_end) * step / decay_steps
        else:
            self.eps = self.eps_end

    
    # Helpers
    

    def _soft_update_target(self) -> None:
        for param, target_param in zip(
            self.online_net.parameters(),
            self.target_net.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def build_history_tensor(
        self,
        state_history: List[np.ndarray],
        action_history: List[int],
    ) -> np.ndarray:
<<<<<<< Updated upstream
        """
        Build history tensor from lists of past states and actions.
        Pads to history_len if necessary.

        Returns
        -------
        np.ndarray, shape (history_len, STATE_DIM + 1)
        """
        T = min(len(state_history), self.history_len)
        history = np.zeros((self.history_len, self.state_dim + 1), dtype=np.float32)
=======
        T       = min(len(state_history), self.history_len)
        history = np.zeros(
            (self.history_len, self.state_dim + 1), dtype=np.float32
        )
>>>>>>> Stashed changes
        for i in range(T):
            idx = len(state_history) - T + i
            history[i, :self.state_dim]  = state_history[idx]
            history[i,  self.state_dim]  = (
                float(action_history[idx]) / self.n_actions
            )
        return history
<<<<<<< Updated upstream

    
    # Persistence
    
=======
>>>>>>> Stashed changes

    def save(self, path: Optional[str | Path] = None) -> None:
        path = Path(path or self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "step":       self._step,
                "eps":        self.eps,
                "use_gru":    self.use_gru,
            },
            str(path),
        )
        arch = "GRU-DDQN" if self.use_gru else "DDQN (no GRU)"
        logger.info(f"{arch} saved to {path}")

    def load(self, path: Optional[str | Path] = None) -> "DDQNAgent":
        path = Path(path or self.checkpoint_path)
        ckpt = torch.load(str(path), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step   = ckpt.get("step",    0)
        self.eps     = ckpt.get("eps",     self.eps_end)
        self.use_gru = ckpt.get("use_gru", self.use_gru)
        arch = "GRU-DDQN" if self.use_gru else "DDQN (no GRU)"
        logger.info(f"{arch} loaded from {path}")
        return self