"""
Neural transition model for the Patient Environment Simulator.

Learns T(s_{t+1} | s_t, a_t): how a patient's discourse state vector
evolves following a given therapy exercise.

Architecture: MLP with Monte Carlo Dropout for uncertainty estimation.
(Gal & Ghahramani, 2016)

Hyperparameter tuning
---------------------
Call TransitionModel.tune() with a (states, actions, next_states) dataset
to run Optuna-based HPO over hidden layer sizes, dropout, and learning rate.
The best hyperparameters are stored in self.best_params and used automatically
when fit() is called afterwards.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# RCT-derived effect size priors  (Gorshkov et al., 2025)
# Applied to CIU rate (discourse dim 0) and MC score (discourse dim 1)
# when the transition model has not been fitted yet.
# Values are approximate normalised effect magnitudes [0, 0.1].
# ---------------------------------------------------------------------------
PRIOR_EFFECT_SIZES: Dict[int, float] = {
    0:  0.04,
    1:  0.03,
    2:  0.05,
    3:  0.06,
    4:  0.07,
    5:  0.06,
    6:  0.06,
    7:  0.08,
    8:  0.03,
    9:  0.03,
    10: 0.02,
    11: 0.09,
}

# Discourse block dim offsets (ciu_rate=0, mc_score=1 within each task block)
_CIU_DIM = 0
_MC_DIM  = 1


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class TransitionMLP(nn.Module):
    """
    MLP predicting delta_state given current state + action one-hot.

    Input  : [state (STATE_DIM) | action_ohe (N_ACTIONS)]
    Output : delta_state (STATE_DIM)
    """

    def __init__(
        self,
        state_dim:    int,
        n_actions:    int,
        hidden_sizes: List[int],
        dropout:      float,
    ):
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
        return self.net(x)


# ---------------------------------------------------------------------------
# Transition model wrapper
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    Wraps TransitionMLP with training, hyperparameter tuning,
    inference, and uncertainty estimation.

    Parameters
    ----------
    hidden_sizes    : MLP hidden layer sizes. Overridden by tune() if called.
    dropout         : Dropout probability (also used for MC uncertainty).
    mc_samples      : Number of MC Dropout forward passes for uncertainty.
    device          : "cuda" | "cpu" | None (auto-detect).
    checkpoint_path : Where to save/load weights.
    """

    def __init__(
        self,
        hidden_sizes:    List[int]       = (256, 128),
        dropout:         float           = 0.2,
        mc_samples:      int             = 50,
        device:          Optional[str]   = None,
        checkpoint_path: Optional[str]   = None,
    ):
        if not _TORCH:
            raise ImportError("PyTorch is required for TransitionModel.")

        self.hidden_sizes    = list(hidden_sizes)
        self.dropout         = dropout
        self.mc_samples      = mc_samples
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best_params:    Optional[Dict] = None
        self._model:         Optional[TransitionMLP] = None
        self._fitted         = False

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        n_trials:    int   = 30,
        val_split:   float = 0.15,
        num_epochs:  int   = 50,
        batch_size:  int   = 32,
        timeout:     Optional[int] = None,
    ) -> Dict:
        """
        Optuna HPO over hidden sizes, dropout, and learning rate.

        Searches:
          hidden_sizes : one of [(128,64), (256,128), (256,128,64), (512,256,128)]
          dropout      : uniform [0.1, 0.4]
          learning_rate: log-uniform [1e-4, 1e-2]

        Parameters
        ----------
        states, actions, next_states : training data
        n_trials    : number of Optuna trials
        val_split   : fraction of data held out for validation
        num_epochs  : epochs per trial (keep low for speed)
        batch_size  : mini-batch size
        timeout     : stop after this many seconds regardless of n_trials

        Returns
        -------
        dict of best hyperparameters
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        deltas     = next_states - states
        action_ohe = np.eye(N_ACTIONS, dtype=np.float32)[actions]
        split      = int(len(states) * (1 - val_split))

        tr_s = torch.tensor(states[:split]).float()
        tr_a = torch.tensor(action_ohe[:split]).float()
        tr_d = torch.tensor(deltas[:split]).float()
        val_s = torch.tensor(states[split:]).float().to(self.device)
        val_a = torch.tensor(action_ohe[split:]).float().to(self.device)
        val_d = torch.tensor(deltas[split:]).float().to(self.device)

        train_loader = DataLoader(
            TensorDataset(tr_s, tr_a, tr_d),
            batch_size=batch_size,
            shuffle=True,
        )

        hidden_choices = [
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (512, 256, 128),
        ]

        def objective(trial):
            hs  = list(hidden_choices[
                trial.suggest_categorical("hidden_idx", list(range(len(hidden_choices))))
            ])
            dr  = trial.suggest_float("dropout",       0.1,  0.4)
            lr  = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            model = TransitionMLP(
                state_dim=STATE_DIM,
                n_actions=N_ACTIONS,
                hidden_sizes=hs,
                dropout=dr,
            ).to(self.device)

            opt     = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(num_epochs):
                model.train()
                for s_b, a_b, d_b in train_loader:
                    pred = model(s_b.to(self.device), a_b.to(self.device))
                    loss = loss_fn(pred, d_b.to(self.device))
                    opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_s, val_a)
                val_loss = loss_fn(val_pred, val_d).item()

            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = study.best_params
        self.best_params = {
            "hidden_sizes":    list(hidden_choices[best["hidden_idx"]]),
            "dropout":         best["dropout"],
            "learning_rate":   best["learning_rate"],
        }
        # Update model config with best found
        self.hidden_sizes = self.best_params["hidden_sizes"]
        self.dropout      = self.best_params["dropout"]

        logger.info(
            f"HPO complete. Best val_loss={study.best_value:.6f}. "
            f"Params: {self.best_params}"
        )
        return self.best_params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        states:                  np.ndarray,
        actions:                 np.ndarray,
        next_states:             np.ndarray,
        val_split:               float = 0.15,
        learning_rate:           float = 1e-3,
        batch_size:              int   = 32,
        num_epochs:              int   = 200,
        early_stopping_patience: int   = 20,
    ) -> "TransitionModel":
        """
        Train on empirical (s, a, s') triples.

        If tune() was called first, best_params overrides learning_rate,
        hidden_sizes, and dropout.
        """
        # Use tuned hyperparameters if available
        if self.best_params:
            learning_rate    = self.best_params.get("learning_rate", learning_rate)
            self.hidden_sizes = self.best_params.get("hidden_sizes",  self.hidden_sizes)
            self.dropout      = self.best_params.get("dropout",        self.dropout)

        N = len(states)
        assert len(actions) == N and len(next_states) == N, "Mismatched input lengths."

        deltas     = next_states - states
        action_ohe = np.eye(N_ACTIONS, dtype=np.float32)[actions]

        split  = int(N * (1 - val_split))
        tr_s   = torch.tensor(states[:split]).float()
        tr_a   = torch.tensor(action_ohe[:split]).float()
        tr_d   = torch.tensor(deltas[:split]).float()
        val_s  = torch.tensor(states[split:]).float().to(self.device)
        val_a  = torch.tensor(action_ohe[split:]).float().to(self.device)
        val_d  = torch.tensor(deltas[split:]).float().to(self.device)

        train_loader = DataLoader(
            TensorDataset(tr_s, tr_a, tr_d),
            batch_size=batch_size,
            shuffle=True,
        )

        self._model = TransitionMLP(
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn   = nn.MSELoss()

        best_val_loss    = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self._model.train()
            for s_b, a_b, d_b in train_loader:
                pred = self._model(s_b.to(self.device), a_b.to(self.device))
                loss = loss_fn(pred, d_b.to(self.device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(val_s, val_a)
                val_loss = loss_fn(val_pred, val_d).item()

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                if self.checkpoint_path:
                    self.save()
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:4d} | val_loss: {val_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        self._fitted = True
        logger.info(f"TransitionModel trained. Best val loss: {best_val_loss:.6f}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_next_state(
        self,
        state:     np.ndarray,
        action:    int,
        use_prior: bool = False,
    ) -> np.ndarray:
        """
        Predict next state. Falls back to RCT priors if not fitted.

        Returns np.ndarray shape (STATE_DIM,), clipped to [0, 1].
        """
        if not self._fitted or use_prior:
            return self.prior_predict(state, action)

        self._model.eval()
        s_t   = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        a_ohe = torch.tensor(
            np.eye(N_ACTIONS, dtype=np.float32)[action]
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            delta = self._model(s_t, a_ohe).cpu().numpy()[0]

        return np.clip(state + delta, 0.0, 1.0).astype(np.float32)

    def predict_with_uncertainty(
        self,
        state:  np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty estimation.

        Returns
        -------
        (mean_next_state, std_next_state) — both shape (STATE_DIM,)
        """
        if not self._fitted:
            pred = self.prior_predict(state, action)
            return pred, np.zeros_like(pred)

        self._model.train()   # activates dropout at inference
        s_t   = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        a_ohe = torch.tensor(
            np.eye(N_ACTIONS, dtype=np.float32)[action]
        ).unsqueeze(0).to(self.device)

        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                delta = self._model(s_t, a_ohe).cpu().numpy()[0]
                samples.append(np.clip(state + delta, 0.0, 1.0))

        self._model.eval()
        samples = np.stack(samples)
        return (
            samples.mean(axis=0).astype(np.float32),
            samples.std(axis=0).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Prior fallback
    # ------------------------------------------------------------------

    def prior_predict(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Fallback using RCT-derived effect size priors.
        Applies a small positive delta to CIU rate (dim 0) and MC score
        (dim 1) of the cookie_theft task block (the primary elicitation task).
        Gaussian noise reflects uncertainty.
        """
        effect = PRIOR_EFFECT_SIZES.get(action, 0.03)
        delta  = np.zeros(STATE_DIM, dtype=np.float32)
        delta[_CIU_DIM] = effect       + np.random.normal(0, 0.01)
        delta[_MC_DIM]  = effect * 0.6 + np.random.normal(0, 0.005)
        return np.clip(state + delta, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("No checkpoint_path set.")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(self.checkpoint_path))
        logger.info(f"TransitionModel saved to {self.checkpoint_path}")

    def load(self) -> "TransitionModel":
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint at {self.checkpoint_path}")
        self._model = TransitionMLP(
            state_dim=STATE_DIM,
            n_actions=N_ACTIONS,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        ).to(self.device)
        self._model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location=self.device)
        )
        self._model.eval()
        self._fitted = True
        logger.info(f"TransitionModel loaded from {self.checkpoint_path}")
        return self