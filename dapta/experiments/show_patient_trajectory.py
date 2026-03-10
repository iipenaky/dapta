"""Generate a single patient trajectory (state, action) using a trained DAPTA agent.

This script is meant to provide a qualitative example for the thesis results chapter.

Usage:
  python experiments/show_patient_trajectory.py
  python experiments/show_patient_trajectory.py --patient_idx 0 --out_file outputs/figures/patient_trajectory.txt
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from dapta.pes.environment import build_env_population
from dapta.pes.transition_model import TransitionModel
from dapta.prta.action_space import THERAPY_EXERCISES
from dapta.prta.ddqn_agent import DDQNAgent


def load_test_split(dae_dir: Path) -> tuple[np.ndarray, list[str]]:
    dae_path = dae_dir / "state_vectors.npz"
    if not dae_path.exists():
        raise FileNotFoundError(f"Missing DAE state vectors: {dae_path}")

    dae_data = np.load(dae_path, allow_pickle=True)
    state_vectors = dae_data["state_vectors"]
    session_ids = list(dae_data["session_ids"])

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    test_ids = splits.get("test", [])
    test_indices = [session_ids.index(sid) for sid in test_ids if sid in session_ids]
    return state_vectors[test_indices], [session_ids[i] for i in test_indices]


def load_cluster_labels(pes_dir: Path, all_session_ids: list[str]) -> np.ndarray:
    with open(pes_dir / "cluster_assignments.json") as f:
        cluster_info = json.load(f)
    return np.array([cluster_info["assignments"].get(sid, 0) for sid in all_session_ids])


def load_ddqn_agents(rl_dir: Path, state_dim: int, n_actions: int, cluster_labels: np.ndarray) -> tuple[dict[int, DDQNAgent], DDQNAgent]:
    cluster_agents = {}
    n_clusters = int(cluster_labels.max()) + 1

    for cluster_id in range(n_clusters):
        agent_path = rl_dir / f"ddqn_cluster_{cluster_id}.pt"
        if not agent_path.exists():
            continue
        agent = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device="cpu")
        agent.load(agent_path)
        agent.epsilon = 0.0
        cluster_agents[cluster_id] = agent

    g_path = rl_dir / "ddqn_generalised.pt"
    g_agent = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device="cpu")
    g_agent.load(g_path)
    g_agent.epsilon = 0.0

    return cluster_agents, g_agent


def format_action(action: int) -> str:
    if 0 <= action < len(THERAPY_EXERCISES):
        ex = THERAPY_EXERCISES[action]
        return f"{action} ({ex.name})"
    return str(action)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an example patient trajectory.")
    parser.add_argument("--dae_dir", default="outputs/dae", help="DAE output folder")
    parser.add_argument("--pes_dir", default="outputs/pes", help="PES output folder")
    parser.add_argument("--rl_dir", default="outputs/rl", help="RL output folder")
    parser.add_argument("--patient_idx", type=int, default=0, help="Index of a patient in the test split")
    parser.add_argument("--out_file", default="outputs/figures/patient_trajectory.txt", help="Output text file")
    args = parser.parse_args()

    dae_dir = Path(args.dae_dir)
    pes_dir = Path(args.pes_dir)
    rl_dir = Path(args.rl_dir)

    states, session_ids = load_test_split(dae_dir)
    if args.patient_idx < 0 or args.patient_idx >= len(states):
        raise ValueError(f"patient_idx must be between 0 and {len(states)-1}")

    # Load cluster labels (for all sessions) and select this patient's cluster
    all_session_ids = list(json.load(open(dae_dir / "splits.json"))['test'])
    cluster_labels = load_cluster_labels(pes_dir, all_session_ids)
    patient_cluster = int(cluster_labels[args.patient_idx])

    # Load transition model and build one environment
    transition_model = TransitionModel(checkpoint_path=pes_dir / "transition_model.pt")
    transition_model.load()

    envs = build_env_population(
        transition_model=transition_model,
        initial_states=[states[args.patient_idx]],
        episode_horizon=20,
    )
    env = envs[0]

    # Load agents
    state_dim = states.shape[1]
    n_actions = env.action_space.n
    cluster_agents, g_ddqn = load_ddqn_agents(rl_dir, state_dim, n_actions, cluster_labels)

    agent = cluster_agents.get(patient_cluster, g_ddqn)

    # Run one episode, tracking trajectory
    state, _ = env.reset()
    history_states = []
    history_actions = []
    history_rewards = []

    for step in range(env.episode_horizon):
        history_tensor = agent.build_history_tensor(history_states, history_actions)
        action = agent.select_action(state, history_tensor, greedy=True)
        next_state, reward, done, _, _ = env.step(action)

        history_states.append(state)
        history_actions.append(action)
        history_rewards.append(reward)

        state = next_state
        if done:
            break

    cum_improvement = env.get_cumulative_discourse_improvement()

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"Patient trajectory (test split index {args.patient_idx}, cluster {patient_cluster})\n")
        f.write("Action meanings taken from dapta.prta.action_space.THERAPY_EXERCISES\n\n")
        f.write("Step\tAction\tReward\tCumulative improvement (CIU, MC, MLU, TTR, SynComp, Surprisal)\n")
        cum_reward = 0.0
        for t, (a, r) in enumerate(zip(history_actions, history_rewards)):
            cum_reward += r
            f.write(f"{t+1}\t{format_action(a)}\t{r:.4f}\t{cum_reward:.4f}\n")

        f.write("\nFinal cumulative discourse improvement:\n")
        f.write("\t" + ", ".join([f"{v:.4f}" for v in cum_improvement]) + "\n")

    print(f"Saved trajectory to: {out_path}")


if __name__ == "__main__":
    main()
