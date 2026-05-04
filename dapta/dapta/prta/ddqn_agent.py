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


class GRUDuelingQNetwork(nn.Module):
    """GRU-based Dueling DDQN — original DAPTA architecture."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,   # number of features describing the patient (47)
        n_actions: int = N_ACTIONS,   # number of therapy options (12)
        gru_hidden: int = 128,        # size of the GRU memory
        gru_layers: int = 2,          # number of GRU layers (depth of memory)
        fc_hidden: int = 256,         # size of fully connected layers
    ) -> None:
        super().__init__()

        # Store basic settings
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.gru_hidden = gru_hidden

       
        # GRU: learns from past sessions
        # Input = (patient state + action taken)
        # Output = a "memory" of what has happened over time
        self.gru = nn.GRU(
            input_size=state_dim + 1,     # 47 (state) + 1 (action)
            hidden_size=gru_hidden,       # memory size (128)
            num_layers=gru_layers,        # stacked GRU layers
            batch_first=True,             # input format: (batch, time, features)
            dropout=0.1 if gru_layers > 1 else 0.0,  # prevent overfitting
        )

       
        # Combine current state + memory
        # We take:
        # - current patient state (47)
        # - GRU memory (128)
        # → combine into one vector
        shared_input = state_dim + gru_hidden

        # Fully connected layers to process this combined information
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_input, fc_hidden),   # 175 → 256
            nn.ReLU(),                            # activation
            nn.Linear(fc_hidden, fc_hidden // 2), # 256 → 128
            nn.ReLU(),
        )

       
        # Dueling Network Split
       
        # After shared processing, we split into TWO parts

        fc_out = fc_hidden // 2  # = 128

        # 1. VALUE STREAM
        # Outputs ONE number:
        # "How good is this patient state overall?"
        self.value_stream = nn.Sequential(
            nn.Linear(fc_out, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # single value
        )

        # 2. ADVANTAGE STREAM
        # Outputs one number per action:
        # "How good is each therapy option compared to others?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(fc_out, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)  # 12 scores (one per therapy)
        )
   
    def forward(
        self,
        state: "torch.Tensor",
        history: "torch.Tensor",
    ) -> "torch.Tensor":
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


class PrioritisedReplayBuffer:

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
        probs  = np.array(self._priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), size=batch_size, p=probs)
        batch   = [self._buffer[i] for i in indices]

        n       = len(self._buffer)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return batch, indices, weights.astype(np.float32)

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        for idx, td in zip(indices, td_errors):
            self._priorities[idx] = float(abs(td)) + 1e-5

    def anneal_beta(self, step: int, total_steps: int) -> None:
        self.beta = min(
            self.beta_end,
            self.beta + (self.beta_end - self.beta) * step / total_steps,
        )

    def __len__(self) -> int:
        return len(self._buffer)


class DDQNAgent:
    """Double DQN agent supporting both GRU and no-GRU architectures.

    Parameters
    ----------
    use_gru : bool
        If True (default) uses GRUDuelingQNetwork — the original DAPTA
        architecture.  If False uses DuelingQNetwork (no sequential
        memory) for the ablation study.
    """

    def __init__(
        self,
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

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.replay_buffer = PrioritisedReplayBuffer(capacity=buffer_size)
        self._step = 0

    def select_action(
        self,
        state: np.ndarray,
        history: np.ndarray,
        greedy: bool = False,
    ) -> int:
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

    def update(self, total_steps: int) -> Optional[float]:
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

        self.replay_buffer.update_priorities(indices, td_errors)
        self._soft_update_target()

        self._step += 1
        return float(loss.item())

    def decay_epsilon(self, step: int, total_steps: int) -> None:
        decay_steps = int(total_steps * self.eps_fraction)
        if step < decay_steps:
            self.eps = 1.0 - (1.0 - self.eps_end) * step / decay_steps
        else:
            self.eps = self.eps_end

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
        T       = min(len(state_history), self.history_len)
        history = np.zeros(
            (self.history_len, self.state_dim + 1), dtype=np.float32
        )
        for i in range(T):
            idx = len(state_history) - T + i
            history[i, :self.state_dim]  = state_history[idx]
            history[i,  self.state_dim]  = (
                float(action_history[idx]) / self.n_actions
            )
        return history

    
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