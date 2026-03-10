"""
Phase 2b: RL Agent Training.

What this script does:
  1. Loads PES outputs (transition model, cluster assignments, initial states)
  2. Trains patient-specific DDQN agents (one per cluster)
  3. Trains generalised DDQN (G-DDQN) on pooled data
  4. Trains PPO agent as comparison
  5. Evaluates all agents + baselines (RBDE, RTS)
  6. Saves agents, training logs, and results tables

Requires:
  outputs/pes/  (from run_pes.py)

Outputs (used by run_evaluation.py):
  outputs/rl/ddqn_cluster_{i}.pt      - patient-specific DDQN per cluster
  outputs/rl/ddqn_generalised.pt      - G-DDQN
  outputs/rl/ppo_agent/               - PPO model
  outputs/rl/training_logs.json       - reward curves
  outputs/rl/results_table.json       - final evaluation metrics

Usage:
  python experiments/run_rl.py
  python experiments/run_rl.py --total_steps 10000  # faster for testing
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from dapta.pes.transition_model import TransitionModel
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import (
    train_ddqn,
    train_ppo,
    evaluate_agent,
    RuleBasedBaseline,
    RandomBaseline,
)
from dapta.dae.state_builder import STATE_DIM
from dapta.utils.logger import get_logger
from dapta.utils.config import Config

logger = get_logger(__name__, log_file="logs/run_rl.log")


def load_environments(
    transition_model: TransitionModel,
    initial_states: np.ndarray,
    cluster_labels: np.ndarray,
    episode_horizon: int = 20,
) -> Dict[int, List[TherapyEnv]]:
    """Build TherapyEnv instances grouped by cluster."""
    cluster_envs: Dict[int, List[TherapyEnv]] = {}
    n_clusters = int(cluster_labels.max()) + 1

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_states = initial_states[mask]
        if len(cluster_states) == 0:
            logger.warning(f"Cluster {cluster_id} has no patients.")
            cluster_envs[cluster_id] = []
            continue
        envs = build_env_population(
            initial_states=list(cluster_states),
            transition_model=transition_model,
            episode_horizon=episode_horizon,
        )
        cluster_envs[cluster_id] = envs
        logger.info(f"  Cluster {cluster_id}: {len(envs)} environments")

    return cluster_envs


def make_ddqn_agent(checkpoint_path: str = None) -> DDQNAgent:
    """Instantiate a DDQN agent with standard hyperparameters."""
    return DDQNAgent(
        state_dim=STATE_DIM,
        gru_hidden=128,
        gru_layers=2,
        learning_rate=1e-4,
        gamma=0.95,
        tau=0.005,
        buffer_size=10_000,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_fraction=0.3,
        history_len=10,
        checkpoint_path=checkpoint_path,
    )


def evaluate_all_baselines(
    test_envs: List[TherapyEnv],
    ddqn_specific: DDQNAgent,
    ddqn_general: DDQNAgent,
    n_eval: int = 20,
) -> dict:
    """
    Evaluate DDQN (specific), G-DDQN, RBDE, and RTS on test environments.
    Returns mean discourse improvement per agent per metric.
    """
    rbde = RuleBasedBaseline()
    rts = RandomBaseline()
    eval_envs = test_envs[:n_eval]

    results = {}

    # DAPTA (patient-specific DDQN)
    mean_r, mean_ciu = evaluate_agent(ddqn_specific, eval_envs)
    results["DAPTA"] = {"mean_reward": mean_r, "mean_ciu_improvement": mean_ciu}

    # G-DDQN (generalised)
    mean_r, mean_ciu = evaluate_agent(ddqn_general, eval_envs)
    results["G_DDQN"] = {"mean_reward": mean_r, "mean_ciu_improvement": mean_ciu}

    # RBDE
    rbde_rewards, rbde_cius = [], []
    for env in eval_envs:
        r, improvement = rbde.run_episode(env)
        rbde_rewards.append(r)
        rbde_cius.append(float(improvement[0]))
    results["RBDE"] = {
        "mean_reward": float(np.mean(rbde_rewards)),
        "mean_ciu_improvement": float(np.mean(rbde_cius)),
    }

    # RTS
    rts_rewards, rts_cius = [], []
    for env in eval_envs:
        r, improvement = rts.run_episode(env)
        rts_rewards.append(r)
        rts_cius.append(float(improvement[0]))
    results["RTS"] = {
        "mean_reward": float(np.mean(rts_rewards)),
        "mean_ciu_improvement": float(np.mean(rts_cius)),
    }

    return results


def main(args) -> None:
    cfg = Config.load()
    output_dir = Path("outputs/rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 2b: RL Agent Training")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load PES outputs
    # ------------------------------------------------------------------
    logger.info("\n[1/4] Loading PES outputs...")
    pes_dir = Path("outputs/pes")

    if not (pes_dir / "transition_model.pt").exists():
        logger.error("PES outputs not found. Run experiments/run_pes.py first.")
        return

    pes_data = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    initial_states = pes_data["initial_states"]
    cluster_labels = pes_data["cluster_labels"]

    with open(pes_dir / "cluster_assignments.json") as f:
        cluster_info = json.load(f)

    # Load transition model
    transition_model = TransitionModel(
        hidden_sizes=(128, 64),
        dropout=0.2,
        checkpoint_path=str(pes_dir / "transition_model.pt"),
        device=args.device,
    )
    transition_model.load()
    logger.info("Transition model loaded.")

    # ------------------------------------------------------------------
    # Build environments
    # ------------------------------------------------------------------
    logger.info("\n[2/4] Building environments per cluster...")
    cluster_envs = load_environments(
        transition_model=transition_model,
        initial_states=initial_states,
        cluster_labels=cluster_labels,
        episode_horizon=20,
    )

    all_envs = [env for envs in cluster_envs.values() for env in envs]
    logger.info(f"Total environments: {len(all_envs)}")

    # ------------------------------------------------------------------
    # Train patient-specific DDQN per cluster
    # ------------------------------------------------------------------
    logger.info(f"\n[3/4] Training patient-specific DDQN agents ({args.total_steps} steps each)...")
    training_logs = {}
    cluster_agents: Dict[int, DDQNAgent] = {}

    for cluster_id, envs in cluster_envs.items():
        if not envs:
            logger.warning(f"Skipping cluster {cluster_id} — no environments.")
            continue

        logger.info(f"\n  Training DDQN for cluster {cluster_id} ({len(envs)} patients)...")
        ckpt = str(output_dir / f"ddqn_cluster_{cluster_id}.pt")
        agent = make_ddqn_agent(checkpoint_path=ckpt)

        logs = train_ddqn(
            agent=agent,
            envs=envs,
            total_steps=args.total_steps,
            eval_freq=max(100, args.total_steps // 50),
            n_eval_episodes=min(10, len(envs)),
            log_freq=max(100, args.total_steps // 20),
        )

        agent.save()
        cluster_agents[cluster_id] = agent
        training_logs[f"ddqn_cluster_{cluster_id}"] = logs
        logger.info(f"  Cluster {cluster_id} agent saved to {ckpt}")

    # ------------------------------------------------------------------
    # Train generalised DDQN (G-DDQN) on all environments
    # ------------------------------------------------------------------
    logger.info(f"\n  Training G-DDQN on all {len(all_envs)} environments...")
    g_ddqn_ckpt = str(output_dir / "ddqn_generalised.pt")
    g_ddqn = make_ddqn_agent(checkpoint_path=g_ddqn_ckpt)

    g_logs = train_ddqn(
        agent=g_ddqn,
        envs=all_envs,
        total_steps=args.total_steps,
        eval_freq=max(100, args.total_steps // 50),
        n_eval_episodes=min(20, len(all_envs)),
    )
    g_ddqn.save()
    training_logs["ddqn_generalised"] = g_logs
    logger.info(f"  G-DDQN saved to {g_ddqn_ckpt}")

    # ------------------------------------------------------------------
    # Train PPO (on all environments, first env as representative)
    # ------------------------------------------------------------------
    if all_envs and not args.skip_ppo:
        logger.info(f"\n  Training PPO...")
        ppo_save = str(output_dir / "ppo_agent")
        ppo_model = train_ppo(
            env=all_envs[0],
            total_steps=args.total_steps,
            save_path=ppo_save,
        )
        if ppo_model:
            training_logs["ppo"] = {"status": "trained", "steps": args.total_steps}
        else:
            training_logs["ppo"] = {"status": "skipped_no_sb3"}
    else:
        logger.info("  Skipping PPO (--skip_ppo or no environments).")

    # ------------------------------------------------------------------
    # Evaluate all agents
    # ------------------------------------------------------------------
    logger.info("\n[4/4] Evaluating all agents vs baselines...")

    # Use the best cluster agent (largest cluster) as representative DAPTA
    best_cluster = max(cluster_agents.keys(), key=lambda c: len(cluster_envs[c]))
    dapta_agent = cluster_agents[best_cluster]

    results = evaluate_all_baselines(
        test_envs=all_envs,
        ddqn_specific=dapta_agent,
        ddqn_general=g_ddqn,
        n_eval=min(20, len(all_envs)),
    )

    # Also evaluate per-cluster
    per_cluster_results = {}
    for cluster_id, agent in cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if envs:
            mean_r, mean_ciu = evaluate_agent(agent, envs)
            per_cluster_results[str(cluster_id)] = {
                "mean_reward": mean_r,
                "mean_ciu_improvement": mean_ciu,
                "n_patients": len(envs),
            }

    # ------------------------------------------------------------------
    # Save all outputs
    # ------------------------------------------------------------------
    with open(output_dir / "training_logs.json", "w") as f:
        # Convert numpy values to Python floats for JSON serialisation
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(training_logs), f, indent=2)

    results_table = {
        "overall": results,
        "per_cluster": per_cluster_results,
        "config": {
            "total_steps": args.total_steps,
            "episode_horizon": 20,
            "n_clusters": 8,
            "n_environments": len(all_envs),
        }
    }
    with open(output_dir / "results_table.json", "w") as f:
        json.dump(results_table, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 2b Complete — Results Summary")
    logger.info("=" * 60)
    for agent_name, res in results.items():
        logger.info(
            f"  {agent_name:12s}: reward={res['mean_reward']:.3f}, "
            f"CIU_improvement={res['mean_ciu_improvement']:.4f}"
        )
    logger.info(f"\nOutputs saved to: {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 2b: RL Agent Training")
    parser.add_argument(
        "--total_steps", type=int, default=50_000,
        help="Training steps per agent (default: 50000). Use 5000 for a quick test."
    )
    parser.add_argument(
        "--skip_ppo", action="store_true",
        help="Skip PPO training (faster)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()
    main(args)
