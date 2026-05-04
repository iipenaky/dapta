from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dapta.pes.environment import TherapyEnv
from dapta.prta.action_space import N_ACTIONS
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _SB3 = True
except ImportError:
    _SB3 = False
    logger.warning("stable-baselines3 not installed. PPO training unavailable.")


def train_ddqn(
    agent: DDQNAgent,
    envs: List[TherapyEnv],
    total_steps: int = 50_000,
    eval_freq: int = 1_000,
    n_eval_episodes: int = 10,
    log_freq: int = 500,
) -> Dict[str, List[float]]:
    logs: Dict[str, List] = {"steps": [], "mean_reward": [], "mean_ciu_improvement": [], "loss": []}

    step = 0
    episode_rewards = []
    losses = []

    logger.info(f"Training DDQN on {len(envs)} patient environments for {total_steps} steps.")

    while step < total_steps:
        env = envs[np.random.randint(len(envs))]
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

            temp_states = state_history + [state]
            temp_actions = action_history + [action]

            next_history = agent.build_history_tensor(
                temp_states, temp_actions
            )
            agent.replay_buffer.push(
                state, history_tensor, action, reward,
                next_state, next_history, float(done)
            )

            state_history.append(state)
            action_history.append(action)
            state = next_state

            loss = agent.update(total_steps)
            if loss is not None:
                losses.append(loss)

            agent.decay_epsilon(step, total_steps)
            step += 1

            if step % eval_freq == 0:
                mean_r, mean_ciu = evaluate_agent(agent, envs, n_episodes=n_eval_episodes)
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



def evaluate_agent(
    agent: DDQNAgent,
    envs: List[TherapyEnv],
    n_episodes: int = 10,
) -> Tuple[float, float]:
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
        all_ciu_improvements.append(float(np.mean(improvement[:5])))  

    return float(np.mean(all_rewards)), float(np.mean(all_ciu_improvements))


def train_ppo(
    envs,
    total_steps: int = 50_000,
    save_path: Optional[str | Path] = None,
    **ppo_kwargs,
) -> Optional["PPO"]:
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

    def make_env_factory(transition_model, initial_state, episode_horizon, reward_clip, noise_std):
        def _init():
            return TherapyEnv(
                transition_model=transition_model,
                initial_state=initial_state,
                episode_horizon=episode_horizon,
                reward_clip=reward_clip,
                noise_std=noise_std,
            )
        return _init

    vec_env = DummyVecEnv([
        make_env_factory(
            env.transition_model,
            env.initial_state,
            env.episode_horizon,
            env.reward_clip,
            env.noise_std,
        )
        for env in envs
    ])


    model = PPO("MlpPolicy", vec_env, **default_ppo_cfg)
    logger.info(f"Training PPO for {total_steps} steps...")
    model.learn(total_timesteps=total_steps)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info(f"PPO model saved to {save_path}")

    return model

class RuleBasedBaseline:
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


class RandomBaseline:
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
