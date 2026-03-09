"""
Gymnasium-compatible MDP environment for DAPTA.

Implements the therapy sequencing MDP:
  M = (S, A, T, R, γ)

where:
  S = patient discourse state (14-dim)
  A = 12 therapy exercise types
  T = neural transition model (TransitionModel)
  R = discourse-level reward (reward.py)
  γ = 0.95, horizon T = 20

This environment is compatible with Stable-Baselines3 and standard
Gymnasium-based RL agents.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:
    _GYM = False
    # Minimal stub so the rest of the code can be imported
    class gym:
        class Env:
            pass
    class spaces:
        @staticmethod
        def Discrete(n): return None
        @staticmethod
        def Box(**kwargs): return None

from dapta.dae.state_builder import STATE_DIM
from dapta.pes.transition_model import TransitionModel
from dapta.prta.action_space import N_ACTIONS, get_exercise
from dapta.utils.reward import compute_reward, DEFAULT_WEIGHTS
from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class TherapyEnv(gym.Env):
    """
    Gymnasium environment for personalised aphasia therapy sequencing.

    Observation space : Box(0, 1, shape=(STATE_DIM,)) = 14-dim patient state
    Action space      : Discrete(N_ACTIONS=12)
    Reward            : Discourse-level improvement (reward.py)
    Episode horizon   : 20 steps (therapy sessions)
    Discount factor   : 0.95 (set in agent config)

    Parameters
    ----------
    transition_model : Fitted TransitionModel instance
    initial_state    : Starting discourse state vector (shape STATE_DIM)
    episode_horizon  : Number of therapy sessions per episode (default 20)
    reward_weights   : Discourse metric weights (defaults to DEFAULT_WEIGHTS)
    reward_clip      : (min, max) for reward clipping
    noise_std        : Gaussian noise std added to transitions (simulates variability)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_state: np.ndarray,
        episode_horizon: int = 20,
        reward_weights: Optional[dict] = None,
        reward_clip: Tuple[float, float] = (-1.0, 1.0),
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.transition_model = transition_model
        self.initial_state = np.array(initial_state, dtype=np.float32)
        self.episode_horizon = episode_horizon
        self.reward_weights = reward_weights or DEFAULT_WEIGHTS
        self.reward_clip = reward_clip
        self.noise_std = noise_std

        # Gymnasium spaces
        if _GYM:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self._state: np.ndarray = self.initial_state.copy()
        self._step_count: int = 0
        self._episode_history: List[Dict] = []

   
    # Gymnasium interface
   

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = self.initial_state.copy()
        self._step_count = 0
        self._episode_history = []
        return self._state.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one therapy session.

        Parameters
        ----------
        action : int, therapy exercise index [0, N_ACTIONS - 1]

        Returns
        -------
        (observation, reward, terminated, truncated, info)
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        state_before = self._state.copy()

        # Transition
        next_state = self.transition_model.predict_next_state(
            state_before, action
        )

        # Add noise for stochasticity
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=next_state.shape)
            next_state = np.clip(next_state + noise, 0.0, 1.0)

        # Reward — only on discourse dims (first 6)
        reward = compute_reward(
            state_before=state_before[:6],
            state_after=next_state[:6],
            weights=self.reward_weights,
            clip=self.reward_clip,
        )

        self._state = next_state.astype(np.float32)
        self._step_count += 1

        # Record step
        self._episode_history.append({
            "step": self._step_count,
            "action": action,
            "exercise": get_exercise(action).name,
            "state_before": state_before[:6].tolist(),
            "state_after": next_state[:6].tolist(),
            "reward": reward,
        })

        terminated = self._step_count >= self.episode_horizon
        truncated = False

        info = {
            "step": self._step_count,
            "action_name": get_exercise(action).name,
            "reward": reward,
        }

        return self._state.copy(), reward, terminated, truncated, info

    def get_episode_history(self) -> List[Dict]:
        """Return the step history of the current/last episode."""
        return self._episode_history.copy()

    def get_cumulative_discourse_improvement(self) -> np.ndarray:
        """
        Return the total discourse-level improvement across the episode.
        Shape: (6,) — one value per discourse metric.
        """
        if not self._episode_history:
            return np.zeros(6, dtype=np.float32)
        first = np.array(self._episode_history[0]["state_before"])
        last = np.array(self._episode_history[-1]["state_after"])
        return last - first

    def render(self) -> None:
        """Optional: print current state summary."""
        metric_names = ["CIU", "MC", "MLU", "TTR", "SynComp", "Surprisal"]
        logger.info(f"Step {self._step_count}/{self.episode_horizon}")
        for name, val in zip(metric_names, self._state[:6]):
            logger.info(f"  {name}: {val:.3f}")


# ---------------------------------------------------------------------------
# Factory: build environment population from patient data
# ---------------------------------------------------------------------------

def build_env_population(
    initial_states: List[np.ndarray],
    transition_model: TransitionModel,
    episode_horizon: int = 20,
    reward_weights: Optional[dict] = None,
) -> List[TherapyEnv]:
    """
    Build one TherapyEnv per patient in the dataset.

    Parameters
    ----------
    initial_states    : One state vector per patient
    transition_model  : Fitted TransitionModel

    Returns
    -------
    List[TherapyEnv]
    """
    envs = [
        TherapyEnv(
            transition_model=transition_model,
            initial_state=s,
            episode_horizon=episode_horizon,
            reward_weights=reward_weights,
        )
        for s in initial_states
    ]
    logger.info(f"Built {len(envs)} patient environments.")
    return envs
