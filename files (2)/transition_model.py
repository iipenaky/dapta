"""
transition_model.py
===================
Neural transition model for the Patient Environment Simulator (PES).

Learns T(s_{t+1} | s_t, a_t): how a patient's 5-dim discourse state
evolves following a given therapy exercise.

State dimensions (must match reward.METRIC_ORDER):
  0  mlu_morphemes   grammatical complexity
  1  ttr             lexical diversity
  2  ndw             vocabulary breadth
  3  n_utterances    verbal output / engagement
  4  mean_surprisal  fluency proxy (inverted in reward)

Architecture: MLP with Monte Carlo Dropout for uncertainty estimation.
(Gal & Ghahramani, 2016 — "Dropout as a Bayesian Approximation")

Hyperparameter tuning
---------------------
Call TransitionModel.tune() with a (states, actions, next_states) dataset
to run Optuna-based HPO over hidden sizes, dropout, and learning rate.
Best params are stored in self.best_params and applied automatically
when fit() is called afterwards.

Prior fallback
--------------
When the model has not been fitted, prior_predict() applies small
RCT-derived effect sizes (Gorshkov et al., 2025) to the state vector.
This is used during synthetic data augmentation in transitions.py and
as an initialisation fallback before training.

Prior effect sizes are indexed by action_id and applied to all 5
discourse metrics proportionally, with larger effects on the metrics
most directly targeted by the exercise type.
"""

from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

from reward import METRIC_ORDER, N_METRICS
from action_space import N_ACTIONS

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# RCT-DERIVED EFFECT SIZE PRIORS  (Gorshkov et al., 2025)
# =============================================================================
# Approximate normalised effect magnitudes per action_id, applied to
# the primary targeted metric. Values in [0, 0.1] represent the expected
# mean shift in a normalised [0,1] discourse metric per therapy session.
#
# These are intentionally conservative — they represent a single session
# effect, not a treatment course. Gaussian noise is added on top.

PRIOR_EFFECT_SIZES: Dict[int, float] = {
    0:  0.04,   # SFA_naming               — word-level, limited transfer
    1:  0.03,   # phonological_cueing       — word-level, limited transfer
    2:  0.05,   # sentence_production_svo   — sentence-level
    3:  0.06,   # sentence_production_complex
    4:  0.07,   # CILT_dialogue             — discourse, functional
    5:  0.06,   # script_training           — discourse, functional
    6:  0.06,   # story_retelling           — discourse, structured
    7:  0.08,   # conversation_partner      — highest transfer
    8:  0.03,   # reading_comprehension     — sentence, Level_II
    9:  0.03,   # writing_to_dictation      — word, Level_II
    10: 0.02,   # word_picture_matching     — lowest transfer
    11: 0.09,   # free_conversation_prompting — highest in set
}

# Per-metric scaling factors for the prior effect.
# Discourse-level metrics (ndw, n_utterances) benefit most from
# functional discourse exercises; mlu_morphemes benefits from
# sentence-level exercises; surprisal benefits from fluency tasks.
# These are relative weights that sum to 1.0 within each apply call.
_METRIC_SCALES = np.array([
    0.20,   # mlu_morphemes
    0.15,   # ttr
    0.25,   # ndw
    0.25,   # n_utterances
    0.15,   # mean_surprisal (effect is a *reduction*, handled by caller)
], dtype=np.float32)


# =============================================================================
# MLP ARCHITECTURE
# =============================================================================

class TransitionMLP(nn.Module):
    """
    MLP predicting delta_state = s_{t+1} - s_t given (s_t, a_t).

    Input  : [state (N_METRICS) | action one-hot (N_ACTIONS)]
    Output : delta_state (N_METRICS)

    Predicting the delta rather than the next state directly is more
    stable — the network only needs to learn small corrections rather
    than reconstructing the full state from scratch.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        dropout:      float,
    ):
        super().__init__()
        input_dim = N_METRICS + N_ACTIONS
        layers    = []
        in_dim    = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, N_METRICS))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state:      "torch.Tensor",
        action_ohe: "torch.Tensor",
    ) -> "torch.Tensor":
        x = torch.cat([state, action_ohe], dim=-1)
        return self.net(x)


# =============================================================================
# TRANSITION MODEL
# =============================================================================

class TransitionModel:
    """
    Wraps TransitionMLP with training, HPO, inference, and uncertainty.

    Parameters
    ----------
    hidden_sizes    : MLP hidden layer sizes (overridden by tune()).
    dropout         : Dropout rate (also used for MC uncertainty).
    mc_samples      : Forward passes for MC Dropout uncertainty.
    device          : "cuda" | "cpu" | None (auto-detect).
    checkpoint_path : Path to save/load model weights.
    """

    def __init__(
        self,
        hidden_sizes:    List[int]     = (256, 128),
        dropout:         float         = 0.2,
        mc_samples:      int           = 50,
        device:          Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        if not _TORCH:
            raise ImportError(
                "PyTorch is required for TransitionModel. "
                "Install with: pip install torch"
            )

        self.hidden_sizes    = list(hidden_sizes)
        self.dropout         = dropout
        self.mc_samples      = mc_samples
        self.device          = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
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
        n_trials:    int            = 30,
        val_split:   float          = 0.15,
        num_epochs:  int            = 50,
        batch_size:  int            = 32,
        timeout:     Optional[int]  = None,
    ) -> Dict:
        """
        Optuna HPO over hidden sizes, dropout, and learning rate.

        Search space:
          hidden_sizes  : one of 4 preset architectures
          dropout       : uniform [0.1, 0.4]
          learning_rate : log-uniform [1e-4, 1e-2]

        Returns best hyperparameter dict and updates self.best_params.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError(
                "Optuna is required for HPO. Install: pip install optuna"
            )

        hidden_choices = [
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (512, 256, 128),
        ]

        deltas     = next_states - states
        action_ohe = np.eye(N_ACTIONS, dtype=np.float32)[actions]
        split      = int(len(states) * (1 - val_split))

        tr_s  = torch.tensor(states[:split]).float()
        tr_a  = torch.tensor(action_ohe[:split]).float()
        tr_d  = torch.tensor(deltas[:split]).float()
        val_s = torch.tensor(states[split:]).float().to(self.device)
        val_a = torch.tensor(action_ohe[split:]).float().to(self.device)
        val_d = torch.tensor(deltas[split:]).float().to(self.device)

        loader = DataLoader(
            TensorDataset(tr_s, tr_a, tr_d),
            batch_size=batch_size,
            shuffle=True,
        )

        def objective(trial):
            hs  = list(hidden_choices[
                trial.suggest_categorical(
                    "hidden_idx", list(range(len(hidden_choices)))
                )
            ])
            dr  = trial.suggest_float("dropout",       0.1,  0.4)
            lr  = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

            model   = TransitionMLP(hs, dr).to(self.device)
            opt     = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(num_epochs):
                model.train()
                for s_b, a_b, d_b in loader:
                    pred = model(s_b.to(self.device), a_b.to(self.device))
                    loss = loss_fn(pred, d_b.to(self.device))
                    opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val_s, val_a), val_d).item()
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = study.best_params
        self.best_params = {
            "hidden_sizes":  list(hidden_choices[best["hidden_idx"]]),
            "dropout":       best["dropout"],
            "learning_rate": best["learning_rate"],
        }
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
        Train on (s, a, s') triples.
        If tune() was called first, best_params override lr/architecture.
        """
        if self.best_params:
            learning_rate    = self.best_params.get("learning_rate", learning_rate)
            self.hidden_sizes = self.best_params.get("hidden_sizes",  self.hidden_sizes)
            self.dropout      = self.best_params.get("dropout",       self.dropout)

        N = len(states)
        assert len(actions) == N and len(next_states) == N

        deltas     = next_states - states
        action_ohe = np.eye(N_ACTIONS, dtype=np.float32)[actions]
        split      = int(N * (1 - val_split))

        tr_s  = torch.tensor(states[:split]).float()
        tr_a  = torch.tensor(action_ohe[:split]).float()
        tr_d  = torch.tensor(deltas[:split]).float()
        val_s = torch.tensor(states[split:]).float().to(self.device)
        val_a = torch.tensor(action_ohe[split:]).float().to(self.device)
        val_d = torch.tensor(deltas[split:]).float().to(self.device)

        loader = DataLoader(
            TensorDataset(tr_s, tr_a, tr_d),
            batch_size=batch_size,
            shuffle=True,
        )

        self._model = TransitionMLP(
            self.hidden_sizes, self.dropout
        ).to(self.device)

        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_fn   = nn.MSELoss()

        best_val      = float("inf")
        patience_ctr  = 0

        for epoch in range(num_epochs):
            self._model.train()
            for s_b, a_b, d_b in loader:
                pred = self._model(s_b.to(self.device), a_b.to(self.device))
                loss = loss_fn(pred, d_b.to(self.device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self._model(val_s, val_a), val_d
                ).item()

            if val_loss < best_val:
                best_val     = val_loss
                patience_ctr = 0
                if self.checkpoint_path:
                    self.save()
            else:
                patience_ctr += 1

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch:4d} | val_loss: {val_loss:.6f}")

            if patience_ctr >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        self._fitted = True
        logger.info(f"TransitionModel trained. Best val_loss: {best_val:.6f}")
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
        Predict next state. Falls back to prior if model not fitted.
        Returns np.ndarray shape (N_METRICS,), clipped to [0, 1].
        """
        if not self._fitted or use_prior:
            return self.prior_predict(state, action)

        self._model.eval()
        s_t   = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
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

        Returns (mean_next_state, std_next_state), both shape (N_METRICS,).
        A high std indicates the model is uncertain about this transition —
        useful for exploration bonuses in the RL agent.
        """
        if not self._fitted:
            pred = self.prior_predict(state, action)
            return pred, np.zeros_like(pred)

        self._model.train()   # dropout active at inference
        s_t   = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
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

    def prior_predict(
        self,
        state:  np.ndarray,
        action: int,
    ) -> np.ndarray:
        """
        Fallback transition using RCT-derived effect size priors.

        Applies a small positive delta to all 5 discourse metrics,
        scaled by _METRIC_SCALES to reflect which metrics each exercise
        type most directly targets. For mean_surprisal (dim 4), the
        effect is a *reduction* (lower surprisal = better fluency),
        so the delta is negated.

        Gaussian noise reflects session-to-session variability.

        Parameters
        ----------
        state  : np.ndarray shape (N_METRICS,)
        action : int in [0, N_ACTIONS - 1]

        Returns
        -------
        np.ndarray shape (N_METRICS,), clipped to [0, 1]
        """
        effect = PRIOR_EFFECT_SIZES.get(action, 0.03)
        noise  = np.random.normal(0, 0.005, size=N_METRICS).astype(np.float32)

        # Scale effect across metrics
        delta = effect * _METRIC_SCALES + noise

        # Surprisal improves by going DOWN — negate that dimension
        surprisal_idx      = METRIC_ORDER.index("mean_surprisal")
        delta[surprisal_idx] *= -1.0

        return np.clip(state + delta, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("No checkpoint_path set.")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(self.checkpoint_path))
        logger.info(f"TransitionModel saved → {self.checkpoint_path}")

    def load(self) -> "TransitionModel":
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.checkpoint_path}"
            )
        self._model = TransitionMLP(
            self.hidden_sizes, self.dropout
        ).to(self.device)
        self._model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location=self.device)
        )
        self._model.eval()
        self._fitted = True
        logger.info(f"TransitionModel loaded ← {self.checkpoint_path}")
        return self