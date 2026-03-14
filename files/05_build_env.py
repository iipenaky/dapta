"""
05_build_env.py
===============
Phase 5: Build the RL patient environment.

- Learns transition dynamics from real AphasiaBank trajectories
- Generates synthetic patients for RL training (bootstrapped from real data)
- Saves environment configuration and transition model

Output: data/processed/transition_model.pkl
        data/processed/synthetic_patients.csv
        data/processed/env_config.yaml

Usage:
    python 05_build_env.py
"""

import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import gymnasium as gym
from gymnasium import spaces
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR = Path(CFG["paths"]["processed"])
RES_DIR  = Path(CFG["paths"]["results"])
RES_DIR.mkdir(parents=True, exist_ok=True)
E        = CFG["environment"]
SEED     = CFG["project"]["seed"]
np.random.seed(SEED)

logging.basicConfig(
    filename=Path(CFG["paths"]["logs"]) / "05_build_env.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

STATE_FEATURES  = E["state_features"]
EXERCISES       = E["therapy_exercises"]
N_ACTIONS       = len(EXERCISES)
NOISE           = E["transition_noise"]
N_SIM_PATIENTS  = E["n_simulated_patients"]


# ── Transition model ──────────────────────────────────────────────────────────

def build_transition_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (state, action_proxy, next_state) tuples from longitudinal data.
    Since we don't have real action labels, we use task type as action proxy
    and session-to-session changes as transitions.
    """
    rows = []
    metric_cols = [c for c in STATE_FEATURES
                   if c in df.columns and c not in
                   ["wab_aq", "aphasia_subtype_encoded", "age", "session_number"]]

    # Group by participant, sort by session
    for pid, grp in df.groupby("participant_id"):
        grp = grp.sort_values("session_number")
        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            s1 = grp.iloc[i]
            s2 = grp.iloc[i + 1]

            # Use task as action proxy (encode task name to int)
            task_name = str(s1.get("task", "unknown"))

            state = {f"s_{c}": s1.get(c) for c in metric_cols}
            state.update({
                "s_wab_aq":     s1.get("wab_aq"),
                "s_subtype":    s1.get("aphasia_subtype_encoded"),
                "s_age":        s1.get("age"),
                "s_session":    s1.get("session_number"),
                "task_name":    task_name,
            })
            next_state = {f"ns_{c}": s2.get(c) for c in metric_cols}
            reward = s2.get("composite_reward", 0) or 0

            row = {**state, **next_state, "reward": reward}
            rows.append(row)

    return pd.DataFrame(rows)


def train_transition_model(df_trans: pd.DataFrame, metric_cols: list) -> dict:
    """
    Train a GradientBoosting multi-output regressor to predict
    next-state metric values given current state + action.
    """
    feature_cols = [f"s_{c}" for c in metric_cols] + [
        "s_wab_aq", "s_subtype", "s_age", "s_session", "action_encoded"
    ]
    target_cols  = [f"ns_{c}" for c in metric_cols]

    # Encode task as action
    le = LabelEncoder()
    df_trans["action_encoded"] = le.fit_transform(
        df_trans["task_name"].fillna("unknown")
    )

    available_features = [c for c in feature_cols if c in df_trans.columns]
    available_targets  = [c for c in target_cols  if c in df_trans.columns]

    df_clean = df_trans[available_features + available_targets].dropna()
    if len(df_clean) < 10:
        console.print("[yellow]Warning: fewer than 10 transition pairs found. "
                      "Transition model will rely more on synthetic data.[/yellow]")

    X = df_clean[available_features].values
    y = df_clean[available_targets].values

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=SEED,
        ),
        n_jobs=-1,
    )
    model.fit(X, y)

    # CV score
    if len(df_clean) >= 20:
        scores = cross_val_score(model, X, y, cv=min(5, len(df_clean)//4),
                                  scoring="r2")
        console.print(f"Transition model CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
        logging.info(f"Transition model CV R²: {scores.mean():.3f}")

    return {
        "model":            model,
        "label_encoder":    le,
        "feature_cols":     available_features,
        "target_cols":      available_targets,
        "metric_cols":      metric_cols,
    }


# ── Synthetic patient generation ──────────────────────────────────────────────

def generate_synthetic_patients(df_real: pd.DataFrame,
                                  n: int,
                                  metric_cols: list) -> pd.DataFrame:
    """
    Bootstrap synthetic patients from real data distribution.
    Each synthetic patient is a perturbed sample from the real distribution.
    """
    real_metrics = df_real[metric_cols].dropna()
    if len(real_metrics) < 5:
        console.print("[yellow]Very few real patients — synthetic data will be low quality[/yellow]")

    rows = []
    for i in range(n):
        # Sample a real patient as base
        base = real_metrics.sample(1, random_state=i).iloc[0]

        # Add Gaussian noise proportional to each metric's std
        patient = {}
        for col in metric_cols:
            std = real_metrics[col].std()
            val = base[col] + np.random.normal(0, NOISE * (std or 1.0))
            # Clip to plausible range
            col_min = real_metrics[col].quantile(0.01)
            col_max = real_metrics[col].quantile(0.99)
            patient[col] = float(np.clip(val, col_min, col_max))

        # Sample metadata from real distribution
        meta_cols = ["wab_aq", "aphasia_subtype_encoded", "age"]
        for mc in meta_cols:
            if mc in df_real.columns:
                vals = df_real[mc].dropna().values
                patient[mc] = float(np.random.choice(vals)) if len(vals) > 0 else 0.0

        patient["participant_id"] = f"synthetic_{i:04d}"
        patient["session_number"] = 1
        rows.append(patient)

    return pd.DataFrame(rows)


# ── Gymnasium Environment ─────────────────────────────────────────────────────

class AphasiaTherapyEnv(gym.Env):
    """
    Custom Gymnasium environment for aphasia therapy sequencing.

    State:  discourse metric vector + patient metadata
    Action: therapy exercise index (0 to N_ACTIONS-1)
    Reward: change in composite discourse score after exercise
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 patients_df:       pd.DataFrame,
                 transition_model:  dict,
                 metric_cols:       list,
                 state_features:    list,
                 exercises:         list,
                 scaler_params:     dict,
                 max_steps:         int,
                 noise:             float,
                 seed:              int = 42):

        super().__init__()

        self.patients_df      = patients_df.reset_index(drop=True)
        self.transition       = transition_model
        self.metric_cols      = metric_cols
        self.state_features   = state_features
        self.exercises        = exercises
        self.scaler_params    = scaler_params
        self.max_steps        = max_steps
        self.noise            = noise
        self.rng              = np.random.default_rng(seed)

        n_state = len(state_features)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_state,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(exercises))

        self._current_patient = None
        self._state           = None
        self._step_count      = 0

    def _get_state_vector(self, patient_dict: dict) -> np.ndarray:
        vec = []
        for feat in self.state_features:
            val = patient_dict.get(feat, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            # Z-score normalize if scaler available
            if feat in self.scaler_params["means"]:
                mean = self.scaler_params["means"][feat]
                std  = self.scaler_params["stds"][feat] or 1.0
                val  = (float(val) - mean) / std
            vec.append(float(val))
        return np.array(vec, dtype=np.float32)

    def _compute_reward(self, before: dict, after: dict) -> float:
        """Composite reward = weighted improvement in discourse metrics."""
        weights = CFG["metrics"]["reward_weights"]
        reward  = 0.0
        w_sum   = 0.0
        for metric, w in weights.items():
            if metric not in before or metric not in after:
                continue
            b = before.get(metric) or 0.0
            a = after.get(metric)  or 0.0
            delta = a - b
            if metric == "maze_ratio":
                delta = -delta  # lower maze = better
            reward += w * delta
            w_sum  += w
        return float(reward / w_sum) if w_sum > 0 else 0.0

    def _simulate_transition(self, state_dict: dict, action_idx: int) -> dict:
        """
        Simulate patient state after an exercise using learned transition model.
        Falls back to rule-based heuristic if model not available.
        """
        tm = self.transition
        if tm is None:
            return self._heuristic_transition(state_dict, action_idx)

        model      = tm["model"]
        feat_cols  = tm["feature_cols"]
        target_cols = tm["target_cols"]
        le         = tm["label_encoder"]
        metric_cols = tm["metric_cols"]

        # Build feature vector
        exercise_name = self.exercises[action_idx]["name"]
        try:
            action_enc = le.transform([exercise_name])[0]
        except ValueError:
            action_enc = 0

        feat_vec = []
        for fc in feat_cols:
            if fc == "action_encoded":
                feat_vec.append(float(action_enc))
            elif fc.startswith("s_"):
                key = fc[2:]
                val = state_dict.get(key, 0.0) or 0.0
                feat_vec.append(float(val))
            else:
                feat_vec.append(0.0)

        X = np.array(feat_vec).reshape(1, -1)
        y_pred = model.predict(X)[0]

        new_state = dict(state_dict)
        for i, tc in enumerate(target_cols):
            if tc.startswith("ns_"):
                key = tc[3:]
                # Add small noise
                noise_val = self.rng.normal(0, self.noise * abs(float(y_pred[i]) + 1e-6))
                new_state[key] = float(y_pred[i]) + noise_val

        return new_state

    def _heuristic_transition(self, state_dict: dict, action_idx: int) -> dict:
        """
        Fallback: exercise targets specific metrics with small random improvement.
        """
        exercise  = self.exercises[action_idx]
        targets   = exercise.get("targets", [])
        new_state = dict(state_dict)

        for metric in targets:
            if metric in new_state and new_state[metric] is not None:
                improvement = self.rng.normal(0.05, 0.1)
                new_state[metric] = float(new_state[metric]) + improvement

        return new_state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a random patient
        idx = self.rng.integers(0, len(self.patients_df))
        self._current_patient = self.patients_df.iloc[idx].to_dict()
        self._step_count = 0
        obs = self._get_state_vector(self._current_patient)
        return obs, {}

    def step(self, action: int):
        before = dict(self._current_patient)
        after  = self._simulate_transition(self._current_patient, action)

        reward = self._compute_reward(before, after)
        self._current_patient = after
        self._step_count     += 1

        terminated = False
        truncated  = self._step_count >= self.max_steps
        obs        = self._get_state_vector(after)

        info = {
            "exercise":   self.exercises[action]["name"],
            "step":       self._step_count,
            "before":     before,
            "after":      after,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 5: Building RL Environment")

    master_path = PROC_DIR / "master_metrics.csv"
    if not master_path.exists():
        console.print("[red]master_metrics.csv not found. Run 03_compute_metrics.py first.[/red]")
        return

    df = pd.read_csv(master_path)
    console.print(f"Loaded master dataset: {len(df)} rows, "
                  f"{df['participant_id'].nunique()} participants")

    # Load scaler params
    scaler_path = PROC_DIR / "scaler_params.yaml"
    with open(scaler_path) as f:
        scaler_params = yaml.safe_load(f)

    # Identify available metric columns
    metric_cols = [c for c in E["state_features"]
                   if c in df.columns and c not in
                   ["wab_aq", "aphasia_subtype_encoded", "age", "session_number"]]
    console.print(f"Metric columns for state: {metric_cols}")

    # ── Build transition dataset ───────────────────────────────────────────
    console.print("\nBuilding transition dataset from longitudinal trajectories...")
    df_trans = build_transition_dataset(df)
    console.print(f"Transition pairs found: {len(df_trans)}")

    # ── Train transition model ─────────────────────────────────────────────
    console.print("\nTraining transition model...")
    transition_model = train_transition_model(df_trans, metric_cols)

    # Save transition model
    trans_path = PROC_DIR / "transition_model.pkl"
    joblib.dump(transition_model, trans_path)
    console.print(f"[green]✓ Transition model → {trans_path}[/green]")

    # ── Generate synthetic patients ────────────────────────────────────────
    console.print(f"\nGenerating {N_SIM_PATIENTS} synthetic patients...")
    df_synthetic = generate_synthetic_patients(df, N_SIM_PATIENTS, metric_cols)
    syn_path = PROC_DIR / "synthetic_patients.csv"
    df_synthetic.to_csv(syn_path, index=False)
    console.print(f"[green]✓ Synthetic patients → {syn_path}[/green]")

    # ── Verify environment ─────────────────────────────────────────────────
    console.print("\nVerifying Gymnasium environment...")
    env = AphasiaTherapyEnv(
        patients_df=df_synthetic,
        transition_model=transition_model,
        metric_cols=metric_cols,
        state_features=E["state_features"],
        exercises=EXERCISES,
        scaler_params=scaler_params,
        max_steps=E["max_episode_steps"],
        noise=NOISE,
        seed=SEED,
    )

    obs, _ = env.reset()
    console.print(f"Observation shape: {obs.shape}")
    console.print(f"Action space: Discrete({env.action_space.n})")

    # Run a quick sanity check
    total_reward = 0
    for _ in range(E["max_episode_steps"]):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    console.print(f"Sanity check episode total reward: {total_reward:.4f}")

    # Save environment config for training phase
    env_cfg = {
        "metric_cols":   metric_cols,
        "state_features": E["state_features"],
        "n_actions":     N_ACTIONS,
        "max_steps":     E["max_episode_steps"],
        "noise":         NOISE,
        "obs_dim":       len(E["state_features"]),
    }
    env_cfg_path = PROC_DIR / "env_config.yaml"
    with open(env_cfg_path, "w") as f:
        yaml.dump(env_cfg, f)

    # Save the env class itself for import in training
    joblib.dump(env, PROC_DIR / "env_instance.pkl")

    console.print(f"[green]✓ Environment config → {env_cfg_path}[/green]")

    # ── Plot synthetic vs real distributions ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(metric_cols[:6]):
        if col not in df.columns or col not in df_synthetic.columns:
            continue
        ax = axes[i]
        real_vals = df[col].dropna()
        syn_vals  = df_synthetic[col].dropna()
        ax.hist(real_vals, bins=20, alpha=0.5, label="Real", color="steelblue")
        ax.hist(syn_vals,  bins=20, alpha=0.5, label="Synthetic", color="coral")
        ax.set_title(col)
        ax.legend()
    plt.suptitle("Real vs Synthetic Patient Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(RES_DIR / "synthetic_vs_real_distributions.png", dpi=150)
    plt.close()
    console.print(f"[green]✓ Distribution plot → {RES_DIR / 'synthetic_vs_real_distributions.png'}[/green]")


if __name__ == "__main__":
    main()
