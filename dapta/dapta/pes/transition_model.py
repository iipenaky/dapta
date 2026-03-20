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

_CIU_DIM = 0
_MC_DIM  = 1


class TransitionMLP(nn.Module):

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



class TransitionModel:

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
        self.hidden_sizes = self.best_params["hidden_sizes"]
        self.dropout      = self.best_params["dropout"]

        logger.info(
            f"HPO complete. Best val_loss={study.best_value:.6f}. "
            f"Params: {self.best_params}"
        )
        return self.best_params

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

    def predict_next_state(
        self,
        state:     np.ndarray,
        action:    int,
        use_prior: bool = False,
    ) -> np.ndarray:
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
        if not self._fitted:
            pred = self.prior_predict(state, action)
            return pred, np.zeros_like(pred)

        self._model.train()  
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

    def prior_predict(self, state: np.ndarray, action: int) -> np.ndarray:
        effect = PRIOR_EFFECT_SIZES.get(action, 0.03)
        delta  = np.zeros(STATE_DIM, dtype=np.float32)
        delta[_CIU_DIM] = effect       + np.random.normal(0, 0.01)
        delta[_MC_DIM]  = effect * 0.6 + np.random.normal(0, 0.005)
        return np.clip(state + delta, 0.0, 1.0).astype(np.float32)

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