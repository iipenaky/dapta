"""
Neural transition model for the Patient Environment Simulator.

Learns T(s_{t+1} | s_t, a_t): how a patient's discourse state vector
evolves following a given therapy exercise.

Architecture: MLP with Monte Carlo Dropout for uncertainty estimation.
(Gal & Ghahramani, 2016)

Trained on AphasiaBank longitudinal samples.
For patients without multiple sessions, effect size estimates from
published RCTs are used (Gorshkov et al., 2025).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

from dapta.dae.state_builder import STATE_DIM
from dapta.prta.action_space import N_ACTIONS
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

# Effect-size-based delta estimates from published RCT meta-analysis
# (Gorshkov et al., 2025). Indexed by action_id, applied to CIU rate dim.
# Values are approximate normalised effect magnitudes [0, 0.1].
PRIOR_EFFECT_SIZES = {
    0:  0.04,   # SFA: reliable word-level gain
    1:  0.03,
    2:  0.05,
    3:  0.06,
    4:  0.07,   # CILT: higher functional effect
    5:  0.06,
    6:  0.06,
    7:  0.08,   # Conversation partner: highest
    8:  0.03,
    9:  0.03,
    10: 0.02,
    11: 0.09,   # Free conversation: highest
}



# Neural network


class TransitionMLP(nn.Module):
    """
    MLP that predicts the next discourse state vector given current state + action.

    Input  : [state (STATE_DIM) | action one-hot (N_ACTIONS)] = STATE_DIM + N_ACTIONS
    Output : delta_state (STATE_DIM) — predicted change in state vector
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        hidden_sizes: List[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = state_dim + n_actions
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: "torch.Tensor", action_ohe: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([state, action_ohe], dim=-1)
        return self.net(x)  # Predict delta



# Transition model wrapper


class TransitionModel:
    """
    Wraps TransitionMLP with training, inference, and uncertainty estimation.

    Parameters
    ----------
    hidden_sizes    : MLP hidden layer sizes
    dropout         : Dropout probability (used for MC uncertainty at inference)
    mc_samples      : Number of MC Dropout forward passes for uncertainty
    device          : "cuda" | "cpu"
    checkpoint_path : Where to save/load weights
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        mc_samples: int = 20,
        device: Optional[str] = None,
        checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        if not _TORCH:
            raise ImportError("PyTorch is required for TransitionModel.")
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.mc_samples = mc_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model: Optional[TransitionMLP] = None
        self._fitted = False

   
    # Training
   

    def fit(
        self,
        states: np.ndarray,            # (N, STATE_DIM) — states before
        actions: np.ndarray,           # (N,) — integer action indices
        next_states: np.ndarray,       # (N, STATE_DIM) — states after
        val_split: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
    ) -> "TransitionModel":
        """Train the transition model on empirical (s, a, s') triples."""

        N = len(states)
        assert len(actions) == N and len(next_states) == N

        # Targets are deltas, not absolute next states
        deltas = next_states - states  # (N, STATE_DIM)

        # Action one-hot
        action_ohe = np.eye(N_ACTIONS, dtype=np.float32)[actions]  # (N, N_ACTIONS)

        # Train/val split
        split = int(N * (1 - val_split))
        tr_s = torch.tensor(states[:split]).float()
        tr_a = torch.tensor(action_ohe[:split]).float()
        tr_d = torch.tensor(deltas[:split]).float()
        val_s = torch.tensor(states[split:]).float()
        val_a = torch.tensor(action_ohe[split:]).float()
        val_d = torch.tensor(deltas[split:]).float()

        train_loader = DataLoader(
            TensorDataset(tr_s, tr_a, tr_d), batch_size=batch_size, shuffle=True
        )

        self._model = TransitionMLP(
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=list(self.hidden_sizes),
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self._model.train()
            for s_b, a_b, d_b in train_loader:
                s_b, a_b, d_b = s_b.to(self.device), a_b.to(self.device), d_b.to(self.device)
                pred = self._model(s_b, a_b)
                loss = loss_fn(pred, d_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(val_s.to(self.device), val_a.to(self.device))
                val_loss = loss_fn(val_pred, val_d.to(self.device)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.checkpoint_path:
                    self._save()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:4d} | val_loss: {val_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        self._fitted = True
        logger.info(f"Transition model trained. Best val loss: {best_val_loss:.6f}")
        return self

   
    # Inference
   

    def predict_next_state(
        self,
        state: np.ndarray,      # (STATE_DIM,)
        action: int,            # Integer action index
        use_prior: bool = False,
    ) -> np.ndarray:
        """
        Predict next state given current state and action.

        If model is not fitted (not enough longitudinal data), falls back
        to prior effect size estimates from published RCTs.

        Returns
        -------
        np.ndarray, shape (STATE_DIM,), clipped to [0, 1]
        """
        if not self._fitted or use_prior:
            return self._prior_predict(state, action)

        self._model.eval()
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        a_ohe = torch.tensor(
            np.eye(N_ACTIONS, dtype=np.float32)[action]
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            delta = self._model(s_t, a_ohe).cpu().numpy()[0]

        next_state = np.clip(state + delta, 0.0, 1.0)
        return next_state.astype(np.float32)

    def predict_with_uncertainty(
        self,
        state: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty estimation.

        Returns
        -------
        (mean_next_state, std_next_state) — both shape (STATE_DIM,)
        """
        if not self._fitted:
            pred = self._prior_predict(state, action)
            return pred, np.zeros_like(pred)

        # Enable dropout at inference (MC Dropout)
        self._model.train()  # Activates dropout
        s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        a_ohe = torch.tensor(
            np.eye(N_ACTIONS, dtype=np.float32)[action]
        ).unsqueeze(0).to(self.device)

        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                delta = self._model(s_t, a_ohe).cpu().numpy()[0]
                samples.append(np.clip(state + delta, 0.0, 1.0))

        self._model.eval()
        samples = np.stack(samples)  # (mc_samples, STATE_DIM)
        return samples.mean(axis=0).astype(np.float32), samples.std(axis=0).astype(np.float32)

   
    # Prior (fallback)
   

    def _prior_predict(self, state: np.ndarray, action: int) -> np.ndarray:
        effect = PRIOR_EFFECT_SIZES.get(action, 0.03)
        delta = np.zeros(STATE_DIM, dtype=np.float32)
        delta[0] = effect + np.random.normal(0, 0.01)        # CIU rate
        delta[1] = effect * 0.6 + np.random.normal(0, 0.005) # MC score
        delta[2] = effect * 0.4 + np.random.normal(0, 0.005) # MLU
        delta[3] = effect * 0.2 + np.random.normal(0, 0.003) # TTR
        delta[4] = effect * 0.3 + np.random.normal(0, 0.005) # SynComp
        delta[5] = -effect * 0.3 + np.random.normal(0, 0.003) # Surprisal (lower = better)
        next_state = np.clip(state + delta, 0.0, 1.0)
        return next_state.astype(np.float32)

   
    # Persistence
   

    def _save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(self.checkpoint_path))

    def load(self) -> "TransitionModel":
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {self.checkpoint_path}")
        self._model = TransitionMLP(
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=list(self.hidden_sizes),
            dropout=self.dropout,
        ).to(self.device)
        self._model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location=self.device)
        )
        self._model.eval()
        self._fitted = True
        logger.info(f"TransitionModel loaded from {self.checkpoint_path}")
        return self
