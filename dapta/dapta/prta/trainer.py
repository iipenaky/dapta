"""
Training orchestrator for all DAPTA agents and baselines.

Trains:
  1. DDQN agent (patient-specific per cluster)
  2. DDQN generalised (pooled across all clusters) — G-DDQN
  3. PPO agent (via Stable-Baselines3)
  4. Rule-Based Difficulty Escalation (RBDE) baseline
  5. Random Therapy Sequencing (RTS) baseline
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dapta.pes.environment import TherapyEnv
from dapta.prta.action_space import N_ACTIONS, THERAPY_EXERCISES
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3 = True
except ImportError:
    _SB3 = False
    logger.warning("stable-baselines3 not installed. PPO training unavailable.")



# DDQN Training loop


def train_ddqn(
    agent: DDQNAgent,
    envs: List[TherapyEnv],
    total_steps: int = 50_000,
    eval_freq: int = 1_000,
    n_eval_episodes: int = 10,
    log_freq: int = 500,
) -> Dict[str, List[float]]:
    """
    Train a DDQN agent on a population of patient environments.

    Each episode samples a random environment from the population,
    simulating training across the patient cluster.

    Parameters
    ----------
    agent        : DDQNAgent instance
    envs         : List of TherapyEnv (one per patient in cluster)
    total_steps  : Total environment steps
    eval_freq    : Steps between evaluations

    Returns
    -------
    training_logs: {
        "steps": [...],
        "mean_reward": [...],
        "mean_cumulative_improvement": [...],
        "loss": [...],
    }
    """
    logs: Dict[str, List] = {
        "steps": [], "mean_reward": [], "mean_ciu_improvement": [], "loss": []
    }

    step = 0
    episode_rewards = []
    losses = []

    logger.info(f"Training DDQN on {len(envs)} patient environments for {total_steps} steps.")

    while step < total_steps:
        # Sample a random patient environment
        env = np.random.choice(envs)
        state, _ = env.reset()
        state_history: List[np.ndarray] = []
        action_history: List[int] = []
        episode_reward = 0.0
        done = False

        while not done and step < total_steps:
            history_tensor = agent.build_history_tensor(state_history, action_history)
            action = agent.select_action(state, history_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            next_history = agent.build_history_tensor(
                state_history + [state], action_history + [action]
            )
            agent.replay_buffer.push(
                state, history_tensor, action, reward,
                next_state, next_history, float(done)
            )

            # Update history
            state_history.append(state)
            action_history.append(action)
            state = next_state

            # Train
            loss = agent.update(total_steps)
            if loss is not None:
                losses.append(loss)

            agent.decay_epsilon(step, total_steps)
            step += 1

            # Evaluation
            if step % eval_freq == 0:
                mean_r, mean_ciu = evaluate_agent(agent, envs[:n_eval_episodes])
                logs["steps"].append(step)
                logs["mean_reward"].append(mean_r)
                logs["mean_ciu_improvement"].append(mean_ciu)
                logs["loss"].append(np.mean(losses[-100:]) if losses else 0.0)
                logger.info(
                    f"Step {step:6d} | "
                    f"mean_reward={mean_r:.3f} | "
                    f"mean_ciu_improvement={mean_ciu:.4f} | "
                    f"eps={agent.eps:.3f}"
                )

        episode_rewards.append(episode_reward)

    logger.info(f"DDQN training complete. Steps: {step}")
    return logs



# Evaluation helper


def evaluate_agent(
    agent: DDQNAgent,
    envs: List[TherapyEnv],
    n_episodes: int = 10,
) -> Tuple[float, float]:
    """
    Evaluate agent greedily on a set of environments.

    Returns
    -------
    (mean_cumulative_reward, mean_ciu_rate_improvement)
    """
    all_rewards = []
    all_ciu_improvements = []

    for env in envs[:n_episodes]:
        state, _ = env.reset()
        state_history: List[np.ndarray] = []
        action_history: List[int] = []
        total_reward = 0.0

        for _ in range(env.episode_horizon):
            history_tensor = agent.build_history_tensor(state_history, action_history)
            action = agent.select_action(state, history_tensor, greedy=True)
            next_state, reward, done, _, _ = env.step(action)
            state_history.append(state)
            action_history.append(action)
            state = next_state
            total_reward += reward
            if done:
                break

        all_rewards.append(total_reward)
        improvement = env.get_cumulative_discourse_improvement()
        all_ciu_improvements.append(float(improvement[0]))  # CIU rate dim

    return float(np.mean(all_rewards)), float(np.mean(all_ciu_improvements))



# PPO Training


def train_ppo(
    env: TherapyEnv,
    total_steps: int = 50_000,
    save_path: Optional[str | Path] = None,
    **ppo_kwargs,
) -> Optional["PPO"]:
    """
    Train a PPO agent using Stable-Baselines3.

    Parameters
    ----------
    env         : Single TherapyEnv (SB3 wraps it internally)
    total_steps : Total training timesteps
    save_path   : Where to save the trained model

    Returns
    -------
    Trained PPO model, or None if SB3 not available.
    """
    if not _SB3:
        logger.error("stable-baselines3 not installed. Cannot train PPO.")
        return None

    default_ppo_cfg = dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/ppo",
    )
    default_ppo_cfg.update(ppo_kwargs)

    model = PPO("MlpPolicy", env, **default_ppo_cfg)
    logger.info(f"Training PPO for {total_steps} steps...")
    model.learn(total_timesteps=total_steps)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info(f"PPO model saved to {save_path}")

    return model



# Baseline: Rule-Based Difficulty Escalation (RBDE)


class RuleBasedBaseline:
    """
    Represents current state-of-the-art digital aphasia therapy platforms:
    exercises are delivered in a fixed order of increasing linguistic complexity.

    Order: word (drill) → sentence → discourse (structured) → discourse (functional)
    This matches platforms like Constant Therapy and TalkPath (Privitera et al., 2024).
    """

    # Fixed sequence indexed by step number (repeats after all 12 used)
    FIXED_SEQUENCE = [10, 1, 0, 9, 8, 2, 3, 6, 5, 4, 7, 11]

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        action = self.FIXED_SEQUENCE[self._step % len(self.FIXED_SEQUENCE)]
        self._step += 1
        return action

    def run_episode(self, env: TherapyEnv) -> Tuple[float, np.ndarray]:
        """Run one full episode. Returns (total_reward, discourse_improvement)."""
        self.reset()
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.episode_horizon):
            action = self.select_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward, env.get_cumulative_discourse_improvement()



# Baseline: Random Therapy Sequencing (RTS)


class RandomBaseline:
    """
    Lower-bound baseline: selects exercises uniformly at random.
    """

    def select_action(self, state: np.ndarray, **kwargs) -> int:
        return np.random.randint(0, N_ACTIONS)

    def run_episode(self, env: TherapyEnv) -> Tuple[float, np.ndarray]:
        state, _ = env.reset()
        total_reward = 0.0
        for _ in range(env.episode_horizon):
            action = self.select_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward, env.get_cumulative_discourse_improvement()
