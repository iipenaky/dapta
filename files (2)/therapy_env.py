"""
therapy_env.py
==============
Gymnasium-compatible MDP environment for DAPTA.

Implements the therapy sequencing MDP:
  M = (S, A, T, R, gamma)

where:
  S = patient discourse state (N_METRICS = 5 dims, normalised [0,1])
  A = 12 therapy exercise types  (action_space.py)
  T = neural transition model    (transition_model.py)
  R = discourse-level reward     (reward.py) — weights are learned, not fixed
  gamma = 0.95, horizon T = 20 sessions

Reward weights are passed in at construction time so that the Optuna
HPO loop in run_rq4.py can vary them across trials without rebuilding
the transition model or patient state.

State dimensions (must match reward.METRIC_ORDER):
  0  mlu_morphemes   grammatical complexity
  1  ttr             lexical diversity
  2  ndw             vocabulary breadth
  3  n_utterances    verbal output / engagement
  4  mean_surprisal  fluency proxy (inverted in reward)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from transition_model import TransitionModel
from action_space     import N_ACTIONS, get_exercise
from reward           import (
    METRIC_ORDER,
    N_METRICS,
    compute_reward,
    validate_weights,
)
from logger           import get_logger

logger = get_logger(__name__)


class TherapyEnv(gym.Env):
    """
    Gymnasium environment for personalised aphasia therapy sequencing.

    Observation space : Box(0, 1, shape=(N_METRICS,))  — 5-dim discourse state
    Action space      : Discrete(N_ACTIONS=12)
    Reward            : Weighted discourse improvement (reward.py)
    Episode horizon   : 20 steps (therapy sessions)
    Discount factor   : 0.95  (set in agent config, not here)

    Parameters
    ----------
    transition_model : Fitted TransitionModel instance.
    initial_state    : Starting state vector, shape (N_METRICS,).
    reward_weights   : np.ndarray shape (N_METRICS,) summing to 1.0.
                       Learned by Optuna in run_rq4.py via
                       reward.weights_from_trial(trial).
    episode_horizon  : Number of therapy sessions per episode (default 20).
    reward_clip      : (min, max) for reward clipping.
    noise_std        : Gaussian noise std added to transitions to simulate
                       day-to-day variability in aphasia performance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_state:    np.ndarray,
        reward_weights:   np.ndarray,
        episode_horizon:  int   = 20,
        reward_clip:      tuple = (-1.0, 1.0),
        noise_std:        float = 0.01,
    ):
        super().__init__()

        validate_weights(reward_weights)

        self.transition_model = transition_model
        self.initial_state    = np.array(initial_state, dtype=np.float32)
        self.reward_weights   = np.array(reward_weights, dtype=np.float32)
        self.episode_horizon  = episode_horizon
        self.reward_clip      = reward_clip
        self.noise_std        = noise_std

        assert self.initial_state.shape == (N_METRICS,), (
            f"initial_state shape {self.initial_state.shape} != ({N_METRICS},). "
            f"Expected metrics: {METRIC_ORDER}"
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_METRICS,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self._state:           np.ndarray  = self.initial_state.copy()
        self._step_count:      int         = 0
        self._episode_history: List[Dict]  = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._state           = self.initial_state.copy()
        self._step_count      = 0
        self._episode_history = []
        return self._state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one therapy session.

        Parameters
        ----------
        action : int in [0, N_ACTIONS - 1]

        Returns
        -------
        observation  : np.ndarray shape (N_METRICS,)
        reward       : float
        terminated   : bool — True when episode_horizon reached
        truncated    : bool — always False (no time limit beyond horizon)
        info         : dict with step diagnostics
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        state_before = self._state.copy()

        # Transition: neural model predicts next state
        next_state = self.transition_model.predict_next_state(state_before, action)

        # Gaussian noise simulates day-to-day variability in aphasia
        if self.noise_std > 0:
            noise      = self.np_random.normal(0, self.noise_std, size=next_state.shape)
            next_state = np.clip(next_state + noise, 0.0, 1.0).astype(np.float32)

        # Reward using learned weights
        reward = compute_reward(
            state_before  = state_before,
            state_after   = next_state,
            weights       = self.reward_weights,
            clip          = self.reward_clip,
        )

        self._state       = next_state.astype(np.float32)
        self._step_count += 1

        exercise = get_exercise(action)
        self._episode_history.append({
            "step":         self._step_count,
            "action":       action,
            "exercise":     exercise.name,
            "target_level": exercise.target_level,
            "context":      exercise.context,
            "state_before": state_before.tolist(),
            "state_after":  next_state.tolist(),
            "reward":       reward,
        })

        terminated = self._step_count >= self.episode_horizon
        truncated  = False

        # Per-metric deltas for diagnostics
        delta = next_state - state_before
        info  = {
            "step":        self._step_count,
            "action_name": exercise.name,
            "target_level": exercise.target_level,
            "context":     exercise.context,
            "reward":      reward,
            "delta":       {m: float(delta[i]) for i, m in enumerate(METRIC_ORDER)},
        }

        return self._state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_episode_history(self) -> List[Dict]:
        """Return step-by-step history of the current/last episode."""
        return self._episode_history.copy()

    def get_cumulative_improvement(self) -> Dict[str, float]:
        """
        Total per-metric improvement across the episode.
        Returns a dict keyed by metric name.
        Returns zeros if no steps taken yet.
        """
        if not self._episode_history:
            return {m: 0.0 for m in METRIC_ORDER}

        first = np.array(self._episode_history[0]["state_before"])
        last  = np.array(self._episode_history[-1]["state_after"])
        delta = last - first

        return {m: float(delta[i]) for i, m in enumerate(METRIC_ORDER)}

    def get_action_distribution(self) -> Dict[str, int]:
        """Count of each exercise selected in the current episode."""
        counts: Dict[str, int] = {}
        for step in self._episode_history:
            name = step["exercise"]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def render(self) -> None:
        """Print current state summary to logger."""
        logger.info(f"Step {self._step_count}/{self.episode_horizon}")
        for i, metric in enumerate(METRIC_ORDER):
            logger.info(f"  {metric}: {self._state[i]:.4f}")


# =============================================================================
# Factory
# =============================================================================

def build_env_population(
    initial_states:   List[np.ndarray],
    transition_model: TransitionModel,
    reward_weights:   np.ndarray,
    episode_horizon:  int   = 20,
    reward_clip:      tuple = (-1.0, 1.0),
    noise_std:        float = 0.01,
) -> List[TherapyEnv]:
    """
    Build one TherapyEnv per patient.

    All environments share the same transition model and reward weights.
    In run_rq4.py, this is called once per Optuna trial with the
    trial's sampled weights — enabling per-trial reward shaping.

    Parameters
    ----------
    initial_states   : List of N_METRICS-dim state vectors, one per patient.
    transition_model : Fitted TransitionModel (shared across envs).
    reward_weights   : np.ndarray shape (N_METRICS,) from weights_from_trial().
    episode_horizon  : Sessions per episode.
    reward_clip      : Reward clipping bounds.
    noise_std        : Transition noise std.

    Returns
    -------
    List[TherapyEnv], one per patient.
    """
    envs = [
        TherapyEnv(
            transition_model = transition_model,
            initial_state    = s,
            reward_weights   = reward_weights,
            episode_horizon  = episode_horizon,
            reward_clip      = reward_clip,
            noise_std        = noise_std,
        )
        for s in initial_states
    ]
    logger.info(f"Built {len(envs)} patient environments.")
    return envs