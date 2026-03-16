"""
Gymnasium-compatible MDP environment for DAPTA.

Implements the therapy sequencing MDP:
  M = (S, A, T, R, γ)

where:
  S = patient discourse state (43-dim)
  A = 12 therapy exercise types
  T = neural transition model (TransitionModel)
  R = discourse-level reward (reward.py)
  γ = 0.95, horizon T = 20
"""

from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dapta.dae.state_builder import STATE_DIM, SLICE_DISCOURSE, N_METRICS_PER_TASK, METRIC_NAMES, TASKS
from dapta.pes.transition_model import TransitionModel
from dapta.prta.action_space import N_ACTIONS, get_exercise
from dapta.utils.reward import compute_reward
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class TherapyEnv(gym.Env):
    """
    Gymnasium environment for personalised aphasia therapy sequencing.

    Observation space : Box(0, 1, shape=(STATE_DIM,)) = 43-dim patient state
    Action space      : Discrete(N_ACTIONS=12)
    Reward            : Discourse-level improvement (reward.py)
    Episode horizon   : 20 steps (therapy sessions)
    Discount factor   : 0.95 (set in agent config)

    Parameters
    ----------
    transition_model : Fitted TransitionModel instance.
    initial_state    : Starting state vector, shape (STATE_DIM,).
    episode_horizon  : Number of therapy sessions per episode (default 20).
    reward_clip      : (min, max) for reward clipping.
    noise_std        : Gaussian noise std added to transitions (simulates
                       day-to-day variability in aphasia performance).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_state:    np.ndarray,
        episode_horizon:  int   = 20,
        reward_clip:      tuple = (-1.0, 1.0),
        noise_std:        float = 0.01,
    ):
        super().__init__()

        self.transition_model = transition_model
        self.initial_state    = np.array(initial_state, dtype=np.float32)
        self.episode_horizon  = episode_horizon
        self.reward_clip      = reward_clip
        self.noise_std        = noise_std

        assert self.initial_state.shape == (STATE_DIM,), (
            f"initial_state shape {self.initial_state.shape} != ({STATE_DIM},)"
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self._state:          np.ndarray  = self.initial_state.copy()
        self._step_count:     int         = 0
        self._episode_history: List[Dict] = []

        # Rolling CIU history for regression penalty in reward.py
        # Stores the mean CIU rate (across all tasks) per session.
        self._ciu_history: List[float] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state           = self.initial_state.copy()
        self._step_count      = 0
        self._episode_history = []
        self._ciu_history     = []
        return self._state.copy(), {}

    def step(self, action: int):
        """
        Execute one therapy session.

        Parameters
        ----------
        action : int in [0, N_ACTIONS - 1]

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        state_before = self._state.copy()

        # Transition
        next_state = self.transition_model.predict_next_state(state_before, action)

        # Gaussian noise simulates day-to-day variability
        if self.noise_std > 0:
            noise = self.np_random.normal(0, self.noise_std, size=next_state.shape)
            next_state = np.clip(next_state + noise, 0.0, 1.0).astype(np.float32)

        # Update rolling CIU history — mean of all ciu_rate dims in discourse block
        # ciu_rate is offset 0 within each task's 4-dim block
        ciu_dims  = [i * N_METRICS_PER_TASK + 0 for i in range(len(TASKS))]
        mean_ciu  = float(np.mean(next_state[ciu_dims]))
        self._ciu_history.append(mean_ciu)

        # Reward — full 43-dim state vectors, rolling CIU history for penalty
        reward = compute_reward(
            state_before=state_before,
            state_after=next_state,
            clip=self.reward_clip,
        )

        self._state      = next_state.astype(np.float32)
        self._step_count += 1

        self._episode_history.append({
            "step":         self._step_count,
            "action":       action,
            "exercise":     get_exercise(action).name,
            "state_before": state_before.tolist(),
            "state_after":  next_state.tolist(),
            "reward":       reward,
        })

        terminated = self._step_count >= self.episode_horizon
        truncated  = False

        info = {
            "step":        self._step_count,
            "action_name": get_exercise(action).name,
            "reward":      reward,
            "mean_ciu":    mean_ciu,
        }

        return self._state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_episode_history(self) -> List[Dict]:
        """Return the step history of the current/last episode."""
        return self._episode_history.copy()

    def get_cumulative_discourse_improvement(self) -> np.ndarray:
        """
        Total improvement in the discourse block across the episode.
        Returns shape (20,) — 4 metrics × 5 tasks.
        Returns zeros if no steps taken yet.
        """
        if not self._episode_history:
            return np.zeros(SLICE_DISCOURSE.stop - SLICE_DISCOURSE.start,
                            dtype=np.float32)
        first = np.array(self._episode_history[0]["state_before"])
        last  = np.array(self._episode_history[-1]["state_after"])
        return (last - first)[SLICE_DISCOURSE].astype(np.float32)

    def render(self):
        """Print current discourse state summary."""
        logger.info(f"Step {self._step_count}/{self.episode_horizon}")
        for t_idx, task in enumerate(TASKS):
            for m_idx, metric in enumerate(METRIC_NAMES):
                dim = t_idx * N_METRICS_PER_TASK + m_idx
                logger.info(f"  {task}__{metric}: {self._state[dim]:.3f}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_env_population(
    initial_states:  List[np.ndarray],
    transition_model: TransitionModel,
    episode_horizon: int   = 20,
    reward_clip:     tuple = (-1.0, 1.0),
    noise_std:       float = 0.01,
) -> List[TherapyEnv]:
    """
    Build one TherapyEnv per patient.

    Parameters
    ----------
    initial_states   : List of 43-dim state vectors, one per patient.
    transition_model : Fitted TransitionModel.
    episode_horizon  : Sessions per episode.
    reward_clip      : Reward clipping bounds.
    noise_std        : Transition noise std.

    Returns
    -------
    List[TherapyEnv]
    """
    envs = [
        TherapyEnv(
            transition_model=transition_model,
            initial_state=s,
            episode_horizon=episode_horizon,
            reward_clip=reward_clip,
            noise_std=noise_std,
        )
        for s in initial_states
    ]
    logger.info(f"Built {len(envs)} patient environments.")
    return envs