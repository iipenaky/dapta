from typing import Dict, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from dapta.dae.state_builder import (
    STATE_DIM, SLICE_DISCOURSE, N_METRICS_PER_TASK, METRIC_NAMES, TASKS,
)
from dapta.pes.transition_model import TransitionModel
from dapta.prta.action_space import N_ACTIONS, get_exercise
from dapta.utils.reward import compute_reward
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class TherapyEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_state:    np.ndarray,
        episode_horizon:  int   = 20,
        reward_clip:      tuple = (-1.0, 1.0),
        noise_std:        float = 0.005,   # FIX 3: reduced from 0.01 → 0.005
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

        self._state:           np.ndarray  = self.initial_state.copy()
        self._step_count:      int         = 0
        self._episode_history: List[Dict]  = []
        self._ciu_history:     List[float] = []

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state           = self.initial_state.copy()
        self._step_count      = 0
        self._episode_history = []
        self._ciu_history     = []
        return self._state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        state_before = self._state.copy()

        next_state = self.transition_model.predict_next_state(state_before, action)

        # FIX 3: noise_std is now 0.005 by default (halved from 0.01).
        # The transition model already has stochasticity from MC Dropout
        # and the prior noise terms, so 0.01 was double-counting randomness
        # and preventing the RL agent from learning a stable signal.
        if self.noise_std > 0:
            noise      = self.np_random.normal(0, self.noise_std, size=next_state.shape)
            next_state = np.clip(next_state + noise, 0.0, 1.0).astype(np.float32)

        ciu_dims  = [i * N_METRICS_PER_TASK + 0 for i in range(len(TASKS))]
        mean_ciu  = float(np.mean(next_state[ciu_dims]))
        self._ciu_history.append(mean_ciu)

        reward = compute_reward(
            state_before = state_before,
            state_after  = next_state,
            clip         = self.reward_clip,
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
    def get_episode_history(self) -> List[Dict]:
        return self._episode_history.copy()

    def get_cumulative_discourse_improvement(self) -> np.ndarray:
        if not self._episode_history:
            return np.zeros(
                SLICE_DISCOURSE.stop - SLICE_DISCOURSE.start,
                dtype=np.float32,
            )
        first = np.array(self._episode_history[0]["state_before"])
        last  = np.array(self._episode_history[-1]["state_after"])
        return (last - first)[SLICE_DISCOURSE].astype(np.float32)

    def render(self):
        logger.info(f"Step {self._step_count}/{self.episode_horizon}")
        for t_idx, task in enumerate(TASKS):
            for m_idx, metric in enumerate(METRIC_NAMES):
                dim = t_idx * N_METRICS_PER_TASK + m_idx
                logger.info(f"  {task}__{metric}: {self._state[dim]:.3f}")


# ------------------------------------------------------------------
def build_env_population(
    initial_states:   List[np.ndarray],
    transition_model: TransitionModel,
    episode_horizon:  int   = 20,
    reward_clip:      tuple = (-1.0, 1.0),
    noise_std:        float = 0.005,   # FIX 3: matches new default above
) -> List[TherapyEnv]:

    envs = [
        TherapyEnv(
            transition_model = transition_model,
            initial_state    = s,
            episode_horizon  = episode_horizon,
            reward_clip      = reward_clip,
            noise_std        = noise_std,
        )
        for s in initial_states
    ]
    logger.info(f"Built {len(envs)} patient environments.")
    return envs