from typing import Dict, List

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
    """A simulated therapy session for one aphasia patient.
    
    The AI agent lives inside this environment. At each step it picks a therapy
    exercise, the environment predicts how the patient responds, and hands back
    a reward signal so the agent can learn which exercises actually help.

    Implementing the Gymnasium interface (reset / step / render) means any
    standard RL library (Stable Baselines 3, RLlib, etc.) can train against
    this environment without modification.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_state:    np.ndarray,
        episode_horizon:  int   = 20,
        reward_clip:      tuple = (-1.0, 1.0),
        noise_std:        float = 0.005,  
    ):
        super().__init__()

        self.transition_model = transition_model
        # The patient's baseline speech scores at the start of the episode.
        self.initial_state    = np.array(initial_state, dtype=np.float32)
        # How many therapy sessions (steps) make up one full episode.
        # 20 steps mirrors the typical therapy course duration in AphasiaBank studies.
        self.episode_horizon  = episode_horizon
        # Keeps rewards in a sensible range so training stays stable.
        self.reward_clip      = reward_clip
        # A small amount of randomness added to predictions so the agent
        # learns to handle real-world variability in patient responses.
        # Default sigma of 0.005 represents the medium noise level used in
        # the main experiments; varied during sensitivity analysis.
        self.noise_std        = noise_std

        # Verify the state vector has the expected dimensionality before training begins.
        assert self.initial_state.shape == (STATE_DIM,), (
            f"initial_state shape {self.initial_state.shape} != ({STATE_DIM},)"
        )

        # The state is a vector of speech scores, all normalised between 0 and 1.
        # Bounding the space helps the neural network learn stable value estimates.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        # The agent can choose from a fixed set of therapy exercises.
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal state variables; reset to initial values at the start of each episode.
        self._state:           np.ndarray  = self.initial_state.copy()
        self._step_count:      int         = 0
        # Full record of every session in this episode (used for analysis later).
        self._episode_history: List[Dict]  = []
        # Tracks the patient's CIU score over time (a measure of speech informativeness).
        self._ciu_history:     List[float] = []

    
    def reset(self, *, seed=None, options=None):
        """Resets the environment back to the patient's starting state.
        Called at the beginning of every new episode so training can start fresh.

        Passing seed to super() ensures the Gymnasium RNG is properly seeded,
        which matters for reproducibility when noise_std > 0.
        """
        super().reset(seed=seed)
        self._state           = self.initial_state.copy()
        self._step_count      = 0
        self._episode_history = []
        self._ciu_history     = []
        # Return (observation, info) as required by the Gymnasium API.
        return self._state.copy(), {}

    
    def step(self, action: int):
        """Runs one therapy session.
        
        The agent picks an exercise, the transition model predicts the patient's
        new speech scores, a reward is calculated, and everything is logged.
        Returns the new state, the reward, and whether the episode is finished.

        Parameters
        ----------
        action : integer index into the therapy exercise action space.

        Returns
        -------
        observation : updated patient state vector after the exercise.
        reward      : scalar signal indicating whether discourse scores improved.
        terminated  : True when the episode horizon is reached.
        truncated   : always False (no time-limit truncation separate from horizon).
        info        : dictionary with per-step diagnostics for logging and analysis.
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        state_before = self._state.copy()

        # Ask the transition model: if this patient does this exercise, what
        # will their speech scores look like afterwards?
        next_state = self.transition_model.predict_next_state(state_before, action)

        # Add Gaussian noise to simulate day-to-day variability in patient performance.
        # This prevents the agent from overfitting to the deterministic transition model
        # and makes the learned policy more robust to real-world fluctuations.
        if self.noise_std > 0:
            noise      = self.np_random.normal(0, self.noise_std, size=next_state.shape)
            # Clip keeps all scores within the valid 0 to 1 range after noise is added.
            next_state = np.clip(next_state + noise, 0.0, 1.0).astype(np.float32)

        # CIU (Correct Information Units) is the first metric for each task.
        # We average it across all tasks to get one overall speech quality score.
        # This is used to track progress independently of the reward function.
        ciu_dims  = [i * N_METRICS_PER_TASK + 0 for i in range(len(TASKS))]
        mean_ciu  = float(np.mean(next_state[ciu_dims]))
        self._ciu_history.append(mean_ciu)

        # The reward tells the agent whether this exercise helped or hurt.
        # Positive reward means the patient improved, negative means they got worse.
        # Reward is computed from the change in discourse metrics between steps.
        reward = compute_reward(
            state_before = state_before,
            state_after  = next_state,
            clip         = self.reward_clip,
        )

        self._state      = next_state.astype(np.float32)
        self._step_count += 1

        # Save a human-readable record of what happened this step.
        # Storing state_before and state_after as lists allows later analysis
        # of the full trajectory without re-running the environment.
        self._episode_history.append({
            "step":         self._step_count,
            "action":       action,
            "exercise":     get_exercise(action).name,
            "state_before": state_before.tolist(),
            "state_after":  next_state.tolist(),
            "reward":       reward,
        })

        # The episode ends when we reach the maximum number of sessions.
        terminated = self._step_count >= self.episode_horizon
        truncated  = False

        info = {
            "step":        self._step_count,
            "action_name": get_exercise(action).name,
            "reward":      reward,
            "mean_ciu":    mean_ciu,
        }

        return self._state.copy(), reward, terminated, truncated, info

    
    def get_episode_history(self) -> List[Dict]:
        """Returns the full log of every session in this episode.
        
        Called after an episode completes to extract trajectories for
        offline analysis, plotting, and evaluation reporting.
        """
        return self._episode_history.copy()

    def get_cumulative_discourse_improvement(self) -> np.ndarray:
        """Measures how much the patient's discourse scores improved from
        the very first session to the very last one in this episode.
        
        This is the primary outcome metric used in the held-out evaluation:
        the element-wise difference between the final and initial discourse
        blocks of the state vector, computed across all 140 test patients.
        """
        if not self._episode_history:
            # Return a zero vector if the episode was empty or never stepped.
            return np.zeros(
                SLICE_DISCOURSE.stop - SLICE_DISCOURSE.start,
                dtype=np.float32,
            )
        first = np.array(self._episode_history[0]["state_before"])
        last  = np.array(self._episode_history[-1]["state_after"])
        # Subtract starting scores from ending scores to get the net change.
        # Slicing with SLICE_DISCOURSE isolates only the discourse metric dimensions,
        # excluding static features like subtype encoding and WAB-AQ.
        return (last - first)[SLICE_DISCOURSE].astype(np.float32)

    def render(self):
        """Prints the patient's current speech scores to the log, broken down
        by task and metric. Useful for debugging and monitoring training.
        
        Iterates over all task-metric combinations in the same order they
        appear in the state vector to make the output easy to interpret.
        """
        logger.info(f"Step {self._step_count}/{self.episode_horizon}")
        for t_idx, task in enumerate(TASKS):
            for m_idx, metric in enumerate(METRIC_NAMES):
                # Reconstruct the flat state vector index from task and metric indices.
                dim = t_idx * N_METRICS_PER_TASK + m_idx
                logger.info(f"  {task}__{metric}: {self._state[dim]:.3f}")



def build_env_population(
    initial_states:   List[np.ndarray],
    transition_model: TransitionModel,
    episode_horizon:  int   = 20,
    reward_clip:      tuple = (-1.0, 1.0),
    noise_std:        float = 0.005,   # FIX 3: matches new default above
) -> List[TherapyEnv]:
    """Creates one therapy environment per patient.
    
    This is how the system trains across many patients at once rather than
    just one, which makes the learned therapy plan more general and robust.

    Each environment is independent: it holds its own copy of the patient's
    initial state and episode history, so rollouts from different patients
    do not interfere with each other during training.
    """

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