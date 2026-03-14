"""
aphasia_env.py
==============
Gymnasium environment for aphasia therapy sequencing.
Imported by 05_build_env.py and 06_train_rl.py.
"""

import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

with open("config.yaml") as f:
    _CFG = yaml.safe_load(f)


class AphasiaTherapyEnv(gym.Env):
    """
    Custom Gymnasium environment for aphasia therapy sequencing.

    Observation: z-scored discourse metric vector + patient metadata
    Action:      therapy exercise index (Discrete)
    Reward:      weighted improvement in composite discourse score
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 patients_df:       pd.DataFrame,
                 transition_model:  dict | None,
                 metric_cols:       list[str],
                 state_features:    list[str],
                 exercises:         list[dict],
                 scaler_params:     dict,
                 max_steps:         int,
                 noise:             float,
                 seed:              int = 42):

        super().__init__()

        self.patients_df     = patients_df.reset_index(drop=True)
        self.transition      = transition_model
        self.metric_cols     = metric_cols
        self.state_features  = state_features
        self.exercises       = exercises
        self.scaler_params   = scaler_params
        self.max_steps       = max_steps
        self.noise           = noise
        self.rng             = np.random.default_rng(seed)

        n_state = len(state_features)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_state,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(exercises))

        self._current_patient: dict = {}
        self._step_count:      int  = 0

    # ── Helpers ───────────────────────────────────────────────────────────

    def _normalize(self, key: str, val: float) -> float:
        means = self.scaler_params.get("means", {})
        stds  = self.scaler_params.get("stds",  {})
        mean  = means.get(key, 0.0)
        std   = stds.get(key, 1.0) or 1.0
        return (val - mean) / std

    def _get_obs(self, patient: dict) -> np.ndarray:
        vec = []
        for feat in self.state_features:
            val = patient.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            if feat in self.scaler_params.get("means", {}):
                val = self._normalize(feat, float(val))
            vec.append(float(val))
        return np.array(vec, dtype=np.float32)

    def _compute_reward(self, before: dict, after: dict) -> float:
        weights = _CFG["metrics"]["reward_weights"]
        reward, w_sum = 0.0, 0.0
        for metric, w in weights.items():
            b = before.get(metric) or 0.0
            a = after.get(metric)  or 0.0
            if b is None or a is None:
                continue
            delta = float(a) - float(b)
            if metric == "maze_ratio":
                delta = -delta
            reward += w * delta
            w_sum  += w
        return float(reward / w_sum) if w_sum > 0 else 0.0

    def _simulate_transition(self, state: dict, action_idx: int) -> dict:
        tm = self.transition
        if tm is None:
            return self._heuristic_transition(state, action_idx)

        model       = tm["model"]
        feat_cols   = tm["feature_cols"]
        target_cols = tm["target_cols"]
        le          = tm["label_encoder"]

        exercise_name = self.exercises[action_idx]["name"]
        try:
            action_enc = int(le.transform([exercise_name])[0])
        except ValueError:
            action_enc = 0

        feat_vec = []
        for fc in feat_cols:
            if fc == "action_encoded":
                feat_vec.append(float(action_enc))
            elif fc.startswith("s_"):
                key = fc[2:]
                val = state.get(key, 0.0) or 0.0
                feat_vec.append(float(val))
            else:
                feat_vec.append(0.0)

        import numpy as np
        X      = np.array(feat_vec).reshape(1, -1)
        y_pred = model.predict(X)[0]

        new_state = dict(state)
        for i, tc in enumerate(target_cols):
            if tc.startswith("ns_"):
                key = tc[3:]
                noise = self.rng.normal(0, self.noise * (abs(float(y_pred[i])) + 1e-6))
                new_state[key] = float(y_pred[i]) + noise

        return new_state

    def _heuristic_transition(self, state: dict, action_idx: int) -> dict:
        """Rule-based fallback transition."""
        exercise  = self.exercises[action_idx]
        targets   = exercise.get("targets", [])
        new_state = dict(state)
        for metric in targets:
            if metric in new_state and new_state[metric] is not None:
                improvement = float(self.rng.normal(0.05, 0.10))
                new_state[metric] = float(new_state[metric]) + improvement
        return new_state

    # ── Gym interface ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = int(self.rng.integers(0, len(self.patients_df)))
        self._current_patient = self.patients_df.iloc[idx].to_dict()
        self._step_count      = 0
        obs = self._get_obs(self._current_patient)
        return obs, {}

    def step(self, action: int):
        before = dict(self._current_patient)
        after  = self._simulate_transition(self._current_patient, int(action))

        reward           = self._compute_reward(before, after)
        self._current_patient = after
        self._step_count += 1

        terminated = False
        truncated  = self._step_count >= self.max_steps
        obs        = self._get_obs(after)

        info = {
            "exercise":     self.exercises[int(action)]["name"],
            "step":         self._step_count,
            "reward":       reward,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
