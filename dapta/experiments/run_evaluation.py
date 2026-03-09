"""
Phase 3: Full evaluation and generalisation test.

Produces the results tables and statistical analysis for the thesis
Chapter 4. Runs all agents on the held-out test set and computes:

  RQ2: Does RL outperform rule-based sequencing?
  RQ3: Do discourse-level gains transfer to naturalised speech?
  RQ4: Does patient-specific adaptation outperform G-DDQN?

Statistical analysis:
  - Paired Wilcoxon signed-rank test (Bonferroni corrected)
  - Cohen's d effect size (clinically meaningful threshold: d >= 0.40)
  - 95% Bootstrap confidence intervals
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from dapta.utils.config import Config
from dapta.utils.logger import get_logger
from dapta.utils.metrics_eval import (
    cohens_d_paired,
    wilcoxon_test,
    bonferroni_correction,
    bootstrap_ci,
    effect_size_label,
)
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.pes.transition_model import TransitionModel
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import RuleBasedBaseline, RandomBaseline

logger = get_logger(__name__, log_file="logs/evaluation.log")

METRIC_NAMES = ["CIU_rate", "MC_score", "MLU_morphemes", "TTR", "SynComp", "Surprisal"]
CLINICAL_THRESHOLD = 0.40  # iTalkBetter benchmark Cohen's d



# Agent runners


def run_ddqn_episode(agent: DDQNAgent, env: TherapyEnv) -> np.ndarray:
    """Run one DDQN episode. Returns discourse improvement vector (6,)."""
    state, _ = env.reset()
    state_history = []
    action_history = []
    for _ in range(env.episode_horizon):
        history_tensor = agent.build_history_tensor(state_history, action_history)
        action = agent.select_action(state, history_tensor, greedy=True)
        state_history.append(state)
        action_history.append(action)
        state, _, done, _, _ = env.step(action)
        if done:
            break
    return env.get_cumulative_discourse_improvement()


def run_ppo_episode(agent, env: TherapyEnv) -> np.ndarray:
    """Run one PPO episode. Returns discourse improvement vector (6,)."""
    state, _ = env.reset()
    for _ in range(env.episode_horizon):
        action, _ = agent.predict(state.reshape(1, -1), deterministic=True)
        state, _, done, _, _ = env.step(int(action))
        if done:
            break
    return env.get_cumulative_discourse_improvement()


def run_baseline_episode(baseline, env: TherapyEnv) -> np.ndarray:
    """Run one baseline episode. Returns discourse improvement vector (6,)."""
    _, improvements = baseline.run_episode(env)
    return improvements



# Load agents


def load_ddqn_agents(
    rl_dir: Path,
    cluster_labels: np.ndarray,
    state_dim: int,
    n_actions: int,
    device: str,
) -> Tuple[Dict[int, DDQNAgent], DDQNAgent]:
    """
    Load patient-specific DDQN agents (one per cluster) and G-DDQN.

    Returns
    -------
    cluster_agents : dict mapping cluster_id -> DDQNAgent
    g_ddqn         : generalised DDQNAgent
    """
    n_clusters = int(cluster_labels.max()) + 1
    cluster_agents = {}

    for cluster_id in range(n_clusters):
        agent_path = rl_dir / f"ddqn_cluster_{cluster_id}.pt"
        if not agent_path.exists():
            logger.warning(f"No saved agent for cluster {cluster_id}, skipping.")
            continue
        agent = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device=device)
        agent.load(agent_path)
        agent.epsilon = 0.0  # greedy at evaluation
        cluster_agents[cluster_id] = agent
        logger.info(f"  Loaded DDQN cluster {cluster_id} from {agent_path}")

    g_ddqn_path = rl_dir / "ddqn_generalised.pt"
    g_ddqn = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device=device)
    g_ddqn.load(g_ddqn_path)
    g_ddqn.epsilon = 0.0
    logger.info(f"  Loaded G-DDQN from {g_ddqn_path}")

    return cluster_agents, g_ddqn



# Main evaluation loop


def evaluate_all_agents(
    test_envs: List[TherapyEnv],
    test_cluster_labels: np.ndarray,
    cluster_agents: Dict[int, DDQNAgent],
    g_ddqn: DDQNAgent,
    rbde: RuleBasedBaseline,
    rts: RandomBaseline,
    ppo_agent=None,
) -> Dict[str, np.ndarray]:
    """
    Run every agent on every test environment.

    Returns
    -------
    improvements : dict mapping agent_name -> (N_patients, 6) array
    """
    results = {k: [] for k in ["DAPTA", "G_DDQN", "RBDE", "RTS"]}
    if ppo_agent is not None:
        results["PPO"] = []

    for i, env in enumerate(test_envs):
        cluster_id = int(test_cluster_labels[i])

        # DAPTA: use the patient-specific agent for this cluster
        if cluster_id in cluster_agents:
            dapta_imp = run_ddqn_episode(cluster_agents[cluster_id], env)
        else:
            # fallback to G-DDQN if cluster agent missing
            dapta_imp = run_ddqn_episode(g_ddqn, env)
        results["DAPTA"].append(dapta_imp)

        # G-DDQN: generalised policy
        results["G_DDQN"].append(run_ddqn_episode(g_ddqn, env))

        # Baselines
        results["RBDE"].append(run_baseline_episode(rbde, env))
        results["RTS"].append(run_baseline_episode(rts, env))

        # PPO (optional)
        if ppo_agent is not None:
            results["PPO"].append(run_ppo_episode(ppo_agent, env))

    return {k: np.stack(v) for k, v in results.items()}



# Statistical analysis


def compute_statistics(
    improvements: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """
    Compute all statistical comparisons for the thesis results chapter.

    Returns a nested dict:
    {
      "per_agent_means": {agent: {metric: mean}},
      "cohens_d": {comparison: {metric: d}},
      "wilcoxon": {comparison: {metric: {stat, p_raw, p_corrected, significant}}},
      "rq3_generalisation": {...},
      "rq4_personalisation": {...},
    }
    """
    out = {}

    # --- Per-agent mean improvement per metric ---
    out["per_agent_means"] = {}
    out["per_agent_stds"] = {}
    for agent, matrix in improvements.items():
        out["per_agent_means"][agent] = {
            m: float(np.mean(matrix[:, i]))
            for i, m in enumerate(METRIC_NAMES)
        }
        out["per_agent_stds"][agent] = {
            m: float(np.std(matrix[:, i], ddof=1))
            for i, m in enumerate(METRIC_NAMES)
        }

    # --- Cohen's d: DAPTA vs each baseline ---
    out["cohens_d"] = {}
    baselines = [k for k in improvements if k != "DAPTA"]
    for baseline in baselines:
        key = f"DAPTA_vs_{baseline}"
        out["cohens_d"][key] = {}
        for i, m in enumerate(METRIC_NAMES):
            d = cohens_d_paired(
                improvements[baseline][:, i],
                improvements["DAPTA"][:, i],
            )
            out["cohens_d"][key][m] = {
                "d": round(d, 3),
                "label": effect_size_label(abs(d)),
                "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
            }

    # --- Wilcoxon signed-rank tests with Bonferroni correction ---
    out["wilcoxon"] = {}
    n_tests = len(METRIC_NAMES) * len(baselines)

    for baseline in baselines:
        key = f"DAPTA_vs_{baseline}"
        out["wilcoxon"][key] = {}
        p_values = []
        metric_order = []

        for i, m in enumerate(METRIC_NAMES):
            diff = improvements["DAPTA"][:, i] - improvements[baseline][:, i]
            stat, p = wilcoxon_test(np.zeros_like(diff), diff)
            p_values.append(p)
            metric_order.append(m)

        p_corrected, _ = bonferroni_correction(p_values, alpha=alpha)

        for m, p_raw, p_corr in zip(metric_order, p_values, p_corrected):
            out["wilcoxon"][key][m] = {
                "p_raw": round(float(p_raw), 4),
                "p_corrected": round(float(p_corr), 4),
                "significant": bool(p_corr < alpha),
            }

    # --- RQ3: Generalisation test (DAPTA vs RBDE on discourse metrics) ---
    dapta = improvements["DAPTA"]
    rbde = improvements["RBDE"]
    rq3 = {}
    for i, m in enumerate(METRIC_NAMES[:5]):  # exclude surprisal for naturalness
        d = cohens_d_paired(rbde[:, i], dapta[:, i])
        lo, hi = bootstrap_ci(dapta[:, i] - rbde[:, i])
        diff = dapta[:, i] - rbde[:, i]
        _, p = wilcoxon_test(np.zeros_like(diff), diff)
        rq3[m] = {
            "dapta_mean": round(float(np.mean(dapta[:, i])), 4),
            "rbde_mean": round(float(np.mean(rbde[:, i])), 4),
            "cohens_d": round(d, 3),
            "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
            "p_value": round(float(p), 4),
            "ci_95": (round(lo, 4), round(hi, 4)),
            "null_rejected": bool(p < alpha and abs(d) >= CLINICAL_THRESHOLD),
        }
    out["rq3_generalisation"] = rq3

    # --- RQ4: Personalisation test (DAPTA vs G-DDQN) ---
    g_ddqn = improvements["G_DDQN"]
    ciu_dapta = dapta[:, 0]
    ciu_gddqn = g_ddqn[:, 0]
    d_ciu = cohens_d_paired(ciu_gddqn, ciu_dapta)
    lo, hi = bootstrap_ci(ciu_dapta - ciu_gddqn)
    diff = ciu_dapta - ciu_gddqn
    _, p = wilcoxon_test(np.zeros_like(diff), diff)

    out["rq4_personalisation"] = {
        "dapta_ciu_mean": round(float(np.mean(ciu_dapta)), 4),
        "gddqn_ciu_mean": round(float(np.mean(ciu_gddqn)), 4),
        "cohens_d_ciu": round(d_ciu, 3),
        "p_value": round(float(p), 4),
        "ci_95": (round(lo, 4), round(hi, 4)),
        "dapta_variance": round(float(np.var(ciu_dapta, ddof=1)), 4),
        "gddqn_variance": round(float(np.var(ciu_gddqn, ddof=1)), 4),
        "variance_reduction_pct": round(
            100 * (np.var(ciu_gddqn, ddof=1) - np.var(ciu_dapta, ddof=1))
            / max(np.var(ciu_gddqn, ddof=1), 1e-8), 2
        ),
        "personalisation_beneficial": bool(
            np.mean(ciu_dapta) > np.mean(ciu_gddqn) and p < alpha
        ),
    }

    return out



# Pretty print results table


def print_results_table(stats: dict) -> None:
    """Print a thesis-ready results summary to the console."""

    print("\n" + "=" * 70)
    print("DAPTA EVALUATION RESULTS — Chapter 4 Summary")
    print("=" * 70)

    # Table 1: Mean discourse improvement per agent
    print("\nTable 1: Mean Discourse Metric Improvement by Agent")
    print("-" * 70)
    header = f"{'Metric':<22}" + "".join(
        f"{a:>10}" for a in ["DAPTA", "G_DDQN", "RBDE", "RTS"]
    )
    print(header)
    print("-" * 70)
    for m in METRIC_NAMES:
        row = f"{m:<22}"
        for agent in ["DAPTA", "G_DDQN", "RBDE", "RTS"]:
            val = stats["per_agent_means"].get(agent, {}).get(m, float("nan"))
            row += f"{val:>10.4f}"
        print(row)
    print("-" * 70)

    # Table 2: Cohen's d vs baselines
    print("\nTable 2: Cohen's d Effect Size (DAPTA vs Baselines)")
    print("-" * 70)
    print(f"{'Metric':<22}{'vs RBDE':>12}{'vs RTS':>12}{'vs G_DDQN':>12}")
    print("-" * 70)
    for m in METRIC_NAMES:
        row = f"{m:<22}"
        for baseline in ["RBDE", "RTS", "G_DDQN"]:
            key = f"DAPTA_vs_{baseline}"
            d = stats["cohens_d"].get(key, {}).get(m, {}).get("d", float("nan"))
            star = "*" if stats["cohens_d"].get(key, {}).get(m, {}).get(
                "clinically_meaningful", False) else " "
            row += f"{d:>10.3f}{star:>2}"
        print(row)
    print("-" * 70)
    print("* = clinically meaningful (d >= 0.40, iTalkBetter benchmark)")

    # Table 3: RQ3 Generalisation
    print("\nTable 3: RQ3 — Generalisation Test (DAPTA vs RBDE)")
    print("-" * 70)
    print(f"{'Metric':<22}{'DAPTA':>10}{'RBDE':>10}{'Cohen d':>10}{'p (corr)':>10}{'Sig?':>8}")
    print("-" * 70)
    for m, v in stats["rq3_generalisation"].items():
        sig = "YES ✓" if v["null_rejected"] else "no"
        print(
            f"{m:<22}{v['dapta_mean']:>10.4f}{v['rbde_mean']:>10.4f}"
            f"{v['cohens_d']:>10.3f}{v['p_value']:>10.4f}{sig:>8}"
        )
    print("-" * 70)

    # RQ4 Personalisation
    rq4 = stats["rq4_personalisation"]
    print("\nRQ4 — Personalisation Test (DAPTA vs G-DDQN on CIU rate)")
    print("-" * 70)
    print(f"  DAPTA mean CIU improvement  : {rq4['dapta_ciu_mean']:.4f}")
    print(f"  G-DDQN mean CIU improvement : {rq4['gddqn_ciu_mean']:.4f}")
    print(f"  Cohen's d                   : {rq4['cohens_d_ciu']:.3f} ({effect_size_label(abs(rq4['cohens_d_ciu']))})")
    print(f"  p-value                     : {rq4['p_value']:.4f}")
    print(f"  95% CI                      : ({rq4['ci_95'][0]:.4f}, {rq4['ci_95'][1]:.4f})")
    print(f"  Variance reduction          : {rq4['variance_reduction_pct']:.1f}%")
    print(f"  Personalisation beneficial  : {'YES ✓' if rq4['personalisation_beneficial'] else 'no'}")
    print("=" * 70)



# Main


def parse_args():
    p = argparse.ArgumentParser(description="DAPTA Phase 3: Evaluation")
    p.add_argument("--dae_dir", default="outputs/dae", help="DAE outputs directory")
    p.add_argument("--pes_dir", default="outputs/pes", help="PES outputs directory")
    p.add_argument("--rl_dir", default="outputs/rl", help="RL outputs directory")
    p.add_argument("--out_dir", default="outputs/evaluation", help="Output directory")
    p.add_argument("--device", default="cpu", help="torch device")
    p.add_argument("--n_actions", type=int, default=12, help="Number of therapy exercises")
    p.add_argument("--skip_ppo", action="store_true", help="Skip PPO evaluation")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dae_dir = Path(args.dae_dir)
    pes_dir = Path(args.pes_dir)
    rl_dir = Path(args.rl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 3: Generalisation & Personalisation Evaluation")
    logger.info("=" * 60)

   
    # 1. Load DAE outputs
   
    logger.info("\n[1/5] Loading DAE outputs...")
    dae_data = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    state_vectors = dae_data["state_vectors"]       # (N, 14)
    state_dim = state_vectors.shape[1]

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)
    # splits.json contains session IDs, not integer indices
    # map session_id -> row index in state_vectors
    all_session_ids = list(dae_data["session_ids"])
    sid_to_idx = {sid: i for i, sid in enumerate(all_session_ids)}

    test_ids = splits["test"]
    test_indices = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    test_states = state_vectors[test_indices]
    logger.info(f"  Test set: {len(test_indices)} patients, state_dim={state_dim}")

   
    # 2. Load PES outputs
   
    logger.info("\n[2/5] Loading PES outputs...")
    pes_data = np.load(pes_dir / "env_initial_states.npz", allow_pickle=True)
    with open(pes_dir / "cluster_assignments.json") as _f:
        _cluster_info = json.load(_f)
    cluster_labels = np.array([_cluster_info["assignments"].get(sid, 0) for sid in all_session_ids])
    transition_model = TransitionModel(checkpoint_path=pes_dir / "transition_model.pt")
    transition_model.load()
    test_cluster_labels = cluster_labels[test_indices]
    logger.info(f"  Transition model loaded. Test cluster distribution: "
                f"{np.bincount(test_cluster_labels.astype(int))}")

   
    # 3. Build test environments
   
    logger.info("\n[3/5] Building test environments...")
    test_envs = build_env_population(
        transition_model=transition_model,
        initial_states=list(test_states),
        episode_horizon=20,
    )
    logger.info(f"  Built {len(test_envs)} test environments.")

   
    # 4. Load agents
   
    logger.info("\n[4/5] Loading trained agents...")
    cluster_agents, g_ddqn = load_ddqn_agents(
        rl_dir=rl_dir,
        cluster_labels=cluster_labels,
        state_dim=state_dim,
        n_actions=args.n_actions,
        device=args.device,
    )
    rbde = RuleBasedBaseline()
    rts = RandomBaseline()

    ppo_agent = None
    if not args.skip_ppo:
        try:
            from stable_baselines3 import PPO as SB3PPO
            ppo_path = rl_dir / "ppo_agent"
            if ppo_path.exists():
                ppo_agent = SB3PPO.load(str(ppo_path))
                logger.info(f"  PPO agent loaded from {ppo_path}")
        except Exception as e:
            logger.warning(f"  Could not load PPO agent: {e}. Skipping.")

   
    # 5. Run evaluation
   
    logger.info("\n[5/5] Running evaluation on test set...")
    improvements = evaluate_all_agents(
        test_envs=test_envs,
        test_cluster_labels=test_cluster_labels,
        cluster_agents=cluster_agents,
        g_ddqn=g_ddqn,
        rbde=rbde,
        rts=rts,
        ppo_agent=ppo_agent,
    )
    logger.info(f"  Evaluated {len(test_envs)} patients across "
                f"{len(improvements)} agents.")

   
    # 6. Statistical analysis
   
    logger.info("\n[6/6] Computing statistics...")
    stats = compute_statistics(improvements, alpha=0.05)

    # Save raw improvements
    for agent, matrix in improvements.items():
        np.save(out_dir / f"improvements_{agent}.npy", matrix)

    # Save statistics
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"  Results saved to {out_dir}/evaluation_results.json")

    # Print results table
    print_results_table(stats)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 Complete.")
    logger.info(f"  Outputs saved to: {out_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()