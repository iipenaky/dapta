"""
ddqn.py
=======
Double Deep Q-Network (DDQN) implementation for DAPTA.

Used by both run_rq2.py (G-DDQN, generalised) and run_rq4.py
(P-DDQN, personalised per cluster).

Architecture
------------
- Online network  : selects actions (argmax Q)
- Target network  : evaluates selected actions (reduces overestimation)
- Experience replay buffer : breaks temporal correlations
- Epsilon-greedy exploration with linear decay

Double DQN update (Van Hasselt et al., 2016):
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a))

This decouples action selection from action evaluation, reducing
the maximisation bias present in vanilla DQN.

State  : 5-dim normalised discourse vector (METRIC_ORDER)
Actions: 12 therapy exercises (N_ACTIONS)

References
----------
Van Hasselt et al. (2016). Deep reinforcement learning with double
    Q-learning. AAAI-16.
Mnih et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518, 529-533.
"""

from collections import deque
from pathlib     import Path
from typing      import Dict, List, Optional, Tuple

import numpy as np
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

from reward      import N_METRICS
from action_space import N_ACTIONS

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Q-NETWORK
# =============================================================================

class QNetwork(nn.Module):
    def __init__(self, hidden_sizes, dropout=0.1, history_len=5):
        super().__init__()
        self.history_len = history_len
        
        # GRU encodes sequence of past states+actions
        self.gru = nn.GRU(
            input_size=N_METRICS + N_ACTIONS,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        
        # Shared MLP after GRU
        layers = []
        in_dim = 128 + N_METRICS  # GRU output + current state
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.shared = nn.Sequential(*layers)
        
        # Dueling streams
        self.value_stream     = nn.Linear(in_dim, 1)
        self.advantage_stream = nn.Linear(in_dim, N_ACTIONS)

    def forward(self, state, history=None):
        batch = state.shape[0]
        
        if history is None:
            # No history — use zeros
            history = torch.zeros(
                batch, self.history_len,
                N_METRICS + N_ACTIONS,
                device=state.device
            )
        
        _, hidden = self.gru(history)
        gru_out = hidden[-1]  # Last GRU layer output
        
        combined = torch.cat([gru_out, state], dim=-1)
        features  = self.shared(combined)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
# =============================================================================
# EXPERIENCE REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """
    Uniform experience replay buffer.

    Stores (s, a, r, s', done) transitions and samples random
    mini-batches for training. Breaking temporal correlation
    between consecutive transitions is essential for stable
    Q-learning with neural networks.

    Parameters
    ----------
    capacity  : maximum number of transitions stored
    seed      : RNG seed for reproducibility
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42):
        self.buffer   = deque(maxlen=capacity)
        self.rng      = random.Random(seed)

    # In ReplayBuffer.push():
    def push(self, state, action, reward, next_state, done, history=None):
        self.buffer.append((state, action, reward, next_state, done, history))

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch      = self.rng.sample(self.buffer, batch_size)
        states     = np.stack([b[0] for b in batch])
        actions    = np.array([b[1] for b in batch], dtype=np.int64)
        rewards    = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones      = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DDQN AGENT
# =============================================================================

class DDQNAgent:
    """
    Double DQN agent with dueling architecture and experience replay.

    Parameters
    ----------
    hidden_sizes         : Q-network hidden layer sizes
    dropout              : dropout rate in Q-network
    learning_rate        : Adam optimiser learning rate
    gamma                : discount factor
    epsilon_start        : initial exploration rate
    epsilon_end          : minimum exploration rate
    epsilon_decay_steps  : steps over which epsilon decays linearly
    batch_size           : mini-batch size for training
    buffer_capacity      : replay buffer size
    target_update_freq   : steps between target network hard updates
    device               : "cuda" | "cpu" | None (auto)
    seed                 : RNG seed
    checkpoint_path      : where to save best model weights
    """

    def __init__(
        self,
        hidden_sizes:        List[int]     = (128, 64),
        dropout:             float         = 0.1,
        learning_rate:       float         = 1e-3,
        gamma:               float         = 0.95,
        epsilon_start:       float         = 1.0,
        epsilon_end:         float         = 0.05,
        epsilon_decay_steps: int           = 5_000,
        batch_size:          int           = 32,
        buffer_capacity:     int           = 10_000,
        target_update_freq:  int           = 100,
        device:              Optional[str] = None,
        seed:                int           = 42,
        checkpoint_path:     Optional[str] = None,
    ):
        if not _TORCH:
            raise ImportError("PyTorch required. pip install torch")

        self.hidden_sizes        = list(hidden_sizes)
        self.dropout             = dropout
        self.learning_rate       = learning_rate
        self.gamma               = gamma
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = (
            (epsilon_start - epsilon_end) / epsilon_decay_steps
        )
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.device              = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed                = seed
        self.checkpoint_path     = (
            Path(checkpoint_path) if checkpoint_path else None
        )

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Online and target networks
        self.online_net = QNetwork(
            self.hidden_sizes, self.dropout
        ).to(self.device)
        self.target_net = QNetwork(
            self.hidden_sizes, self.dropout
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=self.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()   # Huber loss — robust to outliers

        self.replay_buffer = ReplayBuffer(buffer_capacity, seed)

        self._step_count  = 0
        self._best_reward = float("-inf")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, greedy=False, history=None):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        h = torch.FloatTensor(history).unsqueeze(0).to(self.device) \
            if history is not None else None
        
        with torch.no_grad():
            q = self.online_net(s, h)
        return q.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one DDQN update.

        Returns
        -------
        float loss value, or None if buffer not yet large enough.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )

        s  = torch.tensor(states,      dtype=torch.float32).to(self.device)
        a  = torch.tensor(actions,     dtype=torch.int64  ).to(self.device)
        r  = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        ns = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        d  = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        # Current Q values
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # online net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(ns).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(ns).gather(
                1, next_actions
            ).squeeze(1)
            target_q     = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - self.epsilon_decay
        )

        # Hard update target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        envs:            list,
        n_episodes:      int,
        eval_envs:       Optional[list] = None,
        eval_every:      int            = 50,
        eval_episodes:   int            = 10,
    ) -> Dict:
        """
        Train the agent across a population of environments.

        Each episode selects one environment (patient) at random,
        runs one full episode, stores transitions, and learns.

        Parameters
        ----------
        envs          : list of TherapyEnv — training population
        n_episodes    : total training episodes
        eval_envs     : held-out environments for evaluation
                        (if None, evaluates on training envs)
        eval_every    : evaluate every N episodes
        eval_episodes : number of eval episodes per evaluation

        Returns
        -------
        dict with training history:
            episode_rewards  : list of per-episode total reward
            eval_scores      : list of (episode, mean_improvement) tuples
            losses           : list of loss values
        """
        rng             = random.Random(self.seed)
        episode_rewards = []
        eval_scores     = []
        losses          = []

        _eval_envs = eval_envs if eval_envs is not None else envs

        for episode in range(n_episodes):

            # Sample one patient environment
            env   = rng.choice(envs)
            state, _ = env.reset()
            total_reward = 0.0

            while True:
                action     = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done       = terminated or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                loss = self.learn()
                if loss is not None:
                    losses.append(loss)

                state        = next_state
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)

            # Periodic evaluation on held-out patients
            if (episode + 1) % eval_every == 0:
                score = self.evaluate(
                    _eval_envs, n_episodes=eval_episodes
                )
                eval_scores.append((episode + 1, score))

                if score > self._best_reward:
                    self._best_reward = score
                    if self.checkpoint_path:
                        self.save()

                logger.info(
                    f"Episode {episode+1:5d} | "
                    f"reward: {total_reward:6.3f} | "
                    f"eval: {score:6.4f} | "
                    f"eps: {self.epsilon:.3f}"
                )

        return {
            "episode_rewards": episode_rewards,
            "eval_scores":     eval_scores,
            "losses":          losses,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        envs:       list,
        n_episodes: int = 20,
    ) -> float:
        """
        Evaluate agent greedily on a set of environments.

        Returns mean cumulative discourse improvement across all
        episodes. This is the primary metric for comparing agents
        and for the Optuna objective function.

        Parameters
        ----------
        envs       : list of TherapyEnv
        n_episodes : evaluation episodes (samples envs with replacement)

        Returns
        -------
        float : mean cumulative improvement score
        """
        rng    = random.Random(self.seed + 1)
        scores = []

        for _ in range(n_episodes):
            env   = rng.choice(envs)
            state, _ = env.reset()

            while True:
                action = self.select_action(state, greedy=True)
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            # Sum of all per-metric improvements across episode
            improvement = env.get_cumulative_improvement()
            scores.append(sum(improvement.values()))

        return float(np.mean(scores))

    def evaluate_detailed(
        self,
        envs:       list,
        n_episodes: int = 20,
    ) -> Dict:
        """
        Detailed evaluation — returns per-metric improvements,
        action distributions, and per-patient scores.

        Used for thesis analysis tables and plots.
        """
        rng             = random.Random(self.seed + 1)
        all_improvements = []
        all_actions      = []
        per_patient      = []

        for ep in range(n_episodes):
            env   = rng.choice(envs)
            state, _ = env.reset()

            while True:
                action = self.select_action(state, greedy=True)
                state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

            improvement = env.get_cumulative_improvement()
            action_dist = env.get_action_distribution()

            all_improvements.append(improvement)
            all_actions.append(action_dist)
            per_patient.append({
                "episode":     ep,
                "improvement": improvement,
                "actions":     action_dist,
                "total":       sum(improvement.values()),
            })

        # Aggregate per-metric means
        from reward import METRIC_ORDER
        metric_means = {
            m: float(np.mean([imp[m] for imp in all_improvements]))
            for m in METRIC_ORDER
        }

        return {
            "mean_improvement":    float(np.mean([p["total"] for p in per_patient])),
            "std_improvement":     float(np.std([p["total"]  for p in per_patient])),
            "per_metric":          metric_means,
            "per_patient":         per_patient,
            "action_distribution": _aggregate_action_counts(all_actions),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("No checkpoint_path set.")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net":   self.online_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "epsilon":      self.epsilon,
            "step_count":   self._step_count,
            "hidden_sizes": self.hidden_sizes,
            "dropout":      self.dropout,
        }, str(self.checkpoint_path))
        logger.info(f"DDQNAgent saved -> {self.checkpoint_path}")

    def load(self) -> "DDQNAgent":
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.checkpoint_path}"
            )
        ckpt = torch.load(
            str(self.checkpoint_path), map_location=self.device
        )
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon      = ckpt["epsilon"]
        self._step_count  = ckpt["step_count"]
        logger.info(f"DDQNAgent loaded <- {self.checkpoint_path}")
        return self


# =============================================================================
# STATIC BASELINES
# =============================================================================

class GreedyBaseline:
    """
    Always selects the action with the highest generalisation_potential.
    Action 11 (free_conversation_prompting, potential=0.90) is always chosen.
    This is the strongest rule-based baseline.
    """

    def __init__(self):
        from action_space import get_generalisation_potentials
        potentials   = get_generalisation_potentials()
        self._action = int(np.argmax(potentials))

    def evaluate(self, envs: list, n_episodes: int = 20) -> Dict:
        return _run_static_baseline(
            envs, n_episodes, lambda s: self._action, "greedy"
        )


class RoundRobinBaseline:
    """
    Cycles through all 12 exercises in fixed order.
    Represents a structured but non-adaptive therapy protocol.
    """

    def evaluate(self, envs: list, n_episodes: int = 20) -> Dict:
        counter = [0]
        def action_fn(state):
            a = counter[0] % N_ACTIONS
            counter[0] += 1
            return a
        return _run_static_baseline(
            envs, n_episodes, action_fn, "round_robin"
        )


class RandomBaseline:
    """
    Selects actions uniformly at random.
    Lower bound baseline — any trained agent should beat this.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def evaluate(self, envs: list, n_episodes: int = 20) -> Dict:
        return _run_static_baseline(
            envs, n_episodes,
            lambda s: self.rng.randint(0, N_ACTIONS - 1),
            "random"
        )


def _run_static_baseline(
    envs:       list,
    n_episodes: int,
    action_fn,
    name:       str,
) -> Dict:
    """Run a static policy and return detailed evaluation results."""
    rng    = random.Random(0)
    scores = []
    all_improvements = []
    all_actions      = []

    for _ in range(n_episodes):
        env   = rng.choice(envs)
        state, _ = env.reset()

        while True:
            action = action_fn(state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        improvement = env.get_cumulative_improvement()
        action_dist = env.get_action_distribution()
        scores.append(sum(improvement.values()))
        all_improvements.append(improvement)
        all_actions.append(action_dist)

    from reward import METRIC_ORDER
    metric_means = {
        m: float(np.mean([imp[m] for imp in all_improvements]))
        for m in METRIC_ORDER
    }

    return {
        "baseline":            name,
        "mean_improvement":    float(np.mean(scores)),
        "std_improvement":     float(np.std(scores)),
        "per_metric":          metric_means,
        "action_distribution": _aggregate_action_counts(all_actions),
    }


# =============================================================================
# HELPERS
# =============================================================================

def _aggregate_action_counts(action_dists: list) -> Dict[str, float]:
    """Average action counts across episodes."""
    totals: Dict[str, float] = {}
    for dist in action_dists:
        for name, count in dist.items():
            totals[name] = totals.get(name, 0.0) + count
    n = len(action_dists)
    return {k: v / n for k, v in totals.items()}


def hyperparams_from_trial(trial) -> Dict:
    """
    Sample DDQN hyperparameters from an Optuna trial.
    Called inside the Optuna objective in run_rq2.py.

    Searches:
      hidden_sizes         : 4 preset architectures
      dropout              : [0.05, 0.3]
      learning_rate        : log-uniform [1e-4, 1e-2]
      gamma                : [0.90, 0.99]
      epsilon_decay_steps  : [2000, 10000]
      batch_size           : categorical [32, 64, 128]
      target_update_freq   : [50, 200]
      buffer_capacity      : categorical [5000, 10000, 20000]
    """
    hidden_choices = [
        (64, 32),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ]
    return {
        "hidden_sizes": list(hidden_choices[
            trial.suggest_categorical(
                "hidden_idx", list(range(len(hidden_choices)))
            )
        ]),
        "dropout":             trial.suggest_float("dropout",      0.05, 0.30),
        "learning_rate":       trial.suggest_float("lr",           1e-4, 1e-2, log=True),
        "gamma":               trial.suggest_float("gamma",        0.90, 0.99),
        "epsilon_decay_steps": trial.suggest_int(  "eps_decay",    2000, 10_000),
        "batch_size":          trial.suggest_categorical("batch",  [32, 64, 128]),
        "target_update_freq":  trial.suggest_int(  "tuf",          50,   200),
        "buffer_capacity":     trial.suggest_categorical("buf",    [5_000, 10_000, 20_000]),
    }

class PPOAgent:
    """
    Proximal Policy Optimisation agent.
    Comparison agent for RQ2 alongside DDQN.
    """
    def __init__(self, lr=3e-4, gamma=0.99, clip_eps=0.2,
                 hidden_sizes=[128, 64], dropout=0.1,
                 seed=42, checkpoint_path="models/ppo.pt"):
        
        self.gamma    = gamma
        self.clip_eps = clip_eps
        self.device   = torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self._best_reward = -np.inf
        
        torch.manual_seed(seed)
        
        # Actor network (policy)
        actor_layers = []
        in_dim = N_METRICS
        for h in hidden_sizes:
            actor_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, N_ACTIONS))
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        # Critic network (value)
        critic_layers = []
        in_dim = N_METRICS
        for h in hidden_sizes:
            critic_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, greedy=False):
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(s)
        probs  = torch.softmax(logits, dim=-1)
        if greedy:
            return probs.argmax(dim=-1).item()
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def _get_log_prob(self, states, actions):
        logits = self.actor(states)
        probs  = torch.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

    def train(self, envs, n_episodes=2000, eval_envs=None,
              eval_every=100, eval_episodes=50, n_epochs=4):
        
        rewards_log = []
        
        for ep in range(n_episodes):
            env = envs[ep % len(envs)]
            obs, _ = env.reset()
            
            states, actions, rewards, log_probs, values = [], [], [], [], []
            done = False
            
            while not done:
                s = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                logits = self.actor(s)
                probs  = torch.softmax(logits, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                a      = dist.sample()
                lp     = dist.log_prob(a)
                v      = self.critic(s)
                
                obs, r, term, trunc, _ = env.step(a.item())
                done = term or trunc
                
                states.append(s)
                actions.append(a)
                rewards.append(r)
                log_probs.append(lp)
                values.append(v)
            
            # Compute returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            
            returns    = torch.FloatTensor(returns).to(self.device)
            states_t   = torch.cat(states)
            actions_t  = torch.cat(actions)
            old_lps    = torch.cat(log_probs).detach()
            values_t   = torch.cat(values).squeeze().detach()
            advantages = (returns - values_t)
            advantages = (advantages - advantages.mean()) / \
                         (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(n_epochs):
                new_lps = self._get_log_prob(states_t, actions_t)
                ratio   = torch.exp(new_lps - old_lps)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                ) * advantages
                
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(
                    self.critic(states_t).squeeze(), returns
                )
                
                self.opt_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.opt_actor.step()
                
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()
            
            rewards_log.append(sum(rewards))
            
            # Eval
            if eval_envs and (ep + 1) % eval_every == 0:
                mean_r = np.mean([
                    self._run_eval(eval_envs[i % len(eval_envs)])
                    for i in range(eval_episodes)
                ])
                if mean_r > self._best_reward:
                    self._best_reward = mean_r
                    self.save()
                print(f"  PPO ep {ep+1}/{n_episodes} "
                      f"eval={mean_r:.4f}")
        
        return rewards_log

    def _run_eval(self, env):
        obs, _ = env.reset()
        total  = 0
        while True:
            a = self.select_action(obs, greedy=True)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        return total

    def save(self):
        torch.save({
            'actor':  self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, self.checkpoint_path)

    def load(self):
        ck = torch.load(self.checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(ck['actor'])
        self.critic.load_state_dict(ck['critic'])

    def evaluate_detailed(self, envs, n_episodes=50):
        """Match DDQNAgent.evaluate_detailed() interface."""
        from reward import METRIC_ORDER
        scores = []
        per_patient = []
        per_metric = {m: [] for m in METRIC_ORDER}

        for i in range(min(n_episodes, len(envs))):
            env = envs[i % len(envs)]
            obs, _ = env.reset()
            total = 0
            while True:
                a = self.select_action(obs, greedy=True)
                obs, r, term, trunc, _ = env.step(a)
                total += r
                if term or trunc:
                    break
            scores.append(total)
            per_patient.append({"total": total})

            imp = env.get_cumulative_improvement()
            for m in METRIC_ORDER:
                per_metric[m].append(imp.get(m, 0.0))

        return {
            "mean_improvement": float(np.mean(scores)),
            "std_improvement":  float(np.std(scores)),
            "per_patient":      per_patient,
            "per_metric":       {m: float(np.mean(v)) 
                                for m, v in per_metric.items()},
            "action_distribution": {},
        }