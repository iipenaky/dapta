"""
add_ppo_results.py
==================
Loads the saved ppo_agent.zip and runs it on the same held-out test set
used to evaluate DAPTA, G-DDQN, RBDE, and RTS.

Produces:
  - improvements_PPO.npy          (same format as improvements_DAPTA.npy)
  - evaluation_results_with_ppo.json

Run from your project root:
    python add_ppo_results.py

Assumes your outputs are structured as:
    outputs/
      dae/
        state_vectors.npz
        splits.json
      pes/
        transition_model.pt
        cluster_assignments.json
      rl/
        ppo_agent.zip
      evaluation/
        improvements_DAPTA.npy
        improvements_G_DDQN.npy
        improvements_RBDE.npy
        improvements_RTS.npy
        cluster_performance.json
        evaluation_results.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Try importing SB3 ─────────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    _SB3 = True
except ImportError:
    _SB3 = False
    print("[ERROR] stable-baselines3 not installed.")
    print("        Run:  pip install stable-baselines3")
    exit(1)

# ── Try importing your DAPTA modules ──────────────────────────────────────────
try:
    from dapta.pes.environment import TherapyEnv, build_env_population
    from dapta.pes.transition_model import TransitionModel
    _DAPTA = True
except ImportError:
    _DAPTA = False
    print("[ERROR] dapta package not found on sys.path.")
    print("        Run this script from your project root, e.g.:")
    print("            cd /path/to/your/project")
    print("            python add_ppo_results.py")
    exit(1)

METRIC_NAMES  = ["CIU_rate", "MC_score", "MLU_morphemes", "TTR", "SynComp", "Surprisal"]
METRIC_SHORT  = ["CIU Rate", "MC Score", "MLU-m", "TTR", "SynComp", "Surprisal"]
BENCHMARK_D   = 0.42


# ── Stats helpers ─────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def wilcoxon_p(a, b):
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def bonferroni(pvals):
    k = len(pvals)
    return [min(p * k, 1.0) for p in pvals]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    dae_dir  = Path(args.dae_dir)
    pes_dir  = Path(args.pes_dir)
    rl_dir   = Path(args.rl_dir)
    eval_dir = Path(args.eval_dir)

    # ── 1. Load test set ──────────────────────────────────────────────────────
    print("[1/4] Loading test set...")

    dae_data      = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    state_vectors = dae_data["state_vectors"]

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    all_session_ids = list(dae_data["session_ids"])
    sid_to_idx      = {sid: i for i, sid in enumerate(all_session_ids)}
    test_ids        = splits["test"]
    test_indices    = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    test_states     = state_vectors[test_indices]

    print(f"    Test patients: {len(test_indices)}, state_dim: {test_states.shape[1]}")

    # ── 2. Build test environments ────────────────────────────────────────────
    print("[2/4] Building test environments...")

    transition_model = TransitionModel(checkpoint_path=pes_dir / "transition_model.pt")
    transition_model.load()

    test_envs = build_env_population(
        transition_model=transition_model,
        initial_states=list(test_states),
        episode_horizon=20,
    )

    print(f"    Built {len(test_envs)} environments.")

    # ── 3. Load PPO and run episodes ──────────────────────────────────────────
    print("[3/4] Loading PPO model and running evaluation...")

    ppo_path = rl_dir / "ppo_agent.zip"
    if not ppo_path.exists():
        print(f"[ERROR] ppo_agent.zip not found at {ppo_path}")
        print("        Check your --rl_dir argument.")
        exit(1)

    ppo_model = PPO.load(str(ppo_path))
    print(f"    Loaded PPO from {ppo_path}")

    ppo_improvements = []

    for i, env in enumerate(test_envs):
        state, _ = env.reset()

        for _ in range(env.episode_horizon):
            # PPO uses MlpPolicy — takes flat state, no GRU history needed
            action, _ = ppo_model.predict(state.reshape(1, -1), deterministic=True)
            state, _, done, _, _ = env.step(int(action))
            if done:
                break

        improvement = env.get_cumulative_discourse_improvement()
        ppo_improvements.append(improvement)

        if (i + 1) % 20 == 0:
            print(f"    Evaluated {i+1}/{len(test_envs)} patients...")

    ppo_improvements = np.stack(ppo_improvements)   # (133, 6)
    print(f"    PPO shape: {ppo_improvements.shape}")
    print(f"    PPO mean CIU improvement: {np.mean(ppo_improvements[:, 0]):.4f}")

    # ── 4. Save and report ────────────────────────────────────────────────────
    print("[4/4] Saving results...")

    # Save improvements array
    out_npy = eval_dir / "improvements_PPO.npy"
    np.save(out_npy, ppo_improvements)
    print(f"    Saved: {out_npy}")

    # Load existing results and other agent arrays for comparison
    dapta = np.load(eval_dir / "improvements_DAPTA.npy")
    gddqn = np.load(eval_dir / "improvements_G_DDQN.npy")
    rbde  = np.load(eval_dir / "improvements_RBDE.npy")
    rts   = np.load(eval_dir / "improvements_RTS.npy")

    with open(eval_dir / "evaluation_results.json") as f:
        existing_results = json.load(f)

    # Build PPO stats block in the same format as existing results
    ppo_stats = {
        "per_metric_means": {},
        "vs_RBDE": {},
        "vs_DAPTA": {},
    }

    raw_p_vs_rbde  = []
    raw_p_vs_dapta = []

    for mi, metric in enumerate(METRIC_NAMES):
        ppo_mean  = float(np.mean(ppo_improvements[:, mi]))
        rbde_mean = float(np.mean(rbde[:, mi]))
        dapta_mean = float(np.mean(dapta[:, mi]))

        d_vs_rbde  = cohens_d(ppo_improvements[:, mi], rbde[:, mi])
        d_vs_dapta = cohens_d(ppo_improvements[:, mi], dapta[:, mi])

        p_vs_rbde  = wilcoxon_p(ppo_improvements[:, mi], rbde[:, mi])
        p_vs_dapta = wilcoxon_p(ppo_improvements[:, mi], dapta[:, mi])

        raw_p_vs_rbde.append(p_vs_rbde)
        raw_p_vs_dapta.append(p_vs_dapta)

        ppo_stats["per_metric_means"][metric] = round(ppo_mean, 4)
        ppo_stats["vs_RBDE"][metric] = {
            "ppo_mean":   round(ppo_mean, 4),
            "rbde_mean":  round(rbde_mean, 4),
            "cohens_d":   round(d_vs_rbde, 3),
            "clinically_meaningful": abs(d_vs_rbde) >= BENCHMARK_D,
        }
        ppo_stats["vs_DAPTA"][metric] = {
            "ppo_mean":   round(ppo_mean, 4),
            "dapta_mean": round(dapta_mean, 4),
            "cohens_d":   round(d_vs_dapta, 3),
            "clinically_meaningful": abs(d_vs_dapta) >= BENCHMARK_D,
        }

    # Bonferroni correction
    p_corr_rbde  = bonferroni(raw_p_vs_rbde)
    p_corr_dapta = bonferroni(raw_p_vs_dapta)

    for mi, metric in enumerate(METRIC_NAMES):
        ppo_stats["vs_RBDE"][metric]["p_corrected"]  = round(p_corr_rbde[mi], 4)
        ppo_stats["vs_RBDE"][metric]["significant"]  = p_corr_rbde[mi] < 0.05
        ppo_stats["vs_DAPTA"][metric]["p_corrected"] = round(p_corr_dapta[mi], 4)
        ppo_stats["vs_DAPTA"][metric]["significant"] = p_corr_dapta[mi] < 0.05

    # Inject into existing results and save
    existing_results["ppo_comparison"] = ppo_stats
    existing_results["per_agent_means"]["PPO"] = ppo_stats["per_metric_means"]

    out_json = eval_dir / "evaluation_results_with_ppo.json"
    with open(out_json, "w") as f:
        json.dump(existing_results, f, indent=2)
    print(f"    Saved: {out_json}")

    # ── Print summary table ───────────────────────────────────────────────────
    SEP = "─" * 72

    print()
    print("=" * 72)
    print("  PPO EVALUATION RESULTS")
    print("=" * 72)
    print()
    print(f"  {'Metric':<14} {'PPO':>8} {'DAPTA':>8} {'G-DDQN':>8} {'RBDE':>8} {'RTS':>8}")
    print(SEP)

    for mi, short in enumerate(METRIC_SHORT):
        print(
            f"  {short:<14}"
            f" {np.mean(ppo_improvements[:,mi]):>8.4f}"
            f" {np.mean(dapta[:,mi]):>8.4f}"
            f" {np.mean(gddqn[:,mi]):>8.4f}"
            f" {np.mean(rbde[:,mi]):>8.4f}"
            f" {np.mean(rts[:,mi]):>8.4f}"
        )

    print(SEP)
    print()
    print("  PPO vs RBDE (Cohen's d, Bonferroni-corrected p):")
    print(SEP)
    print(f"  {'Metric':<14} {'d':>8} {'p-corr':>8} {'>=0.42?':>10}")
    print(SEP)

    for mi, (short, metric) in enumerate(zip(METRIC_SHORT, METRIC_NAMES)):
        d   = ppo_stats["vs_RBDE"][metric]["cohens_d"]
        p   = ppo_stats["vs_RBDE"][metric]["p_corrected"]
        sig = "YES ✓" if abs(d) >= BENCHMARK_D else "no"
        print(f"  {short:<14} {d:>8.3f} {p:>8.4f} {sig:>10}")

    print(SEP)
    print()
    print("  PPO vs DAPTA (Cohen's d):")
    print(SEP)
    print(f"  {'Metric':<14} {'d (PPO-DAPTA)':>14}  Interpretation")
    print(SEP)

    for mi, (short, metric) in enumerate(zip(METRIC_SHORT, METRIC_NAMES)):
        d = ppo_stats["vs_DAPTA"][metric]["cohens_d"]
        interp = "PPO better" if d > 0.2 else ("DAPTA better" if d < -0.2 else "similar")
        print(f"  {short:<14} {d:>14.3f}  {interp}")

    print(SEP)
    print()
    print(f"  improvements_PPO.npy              → {out_npy}")
    print(f"  evaluation_results_with_ppo.json  → {out_json}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add PPO results to DAPTA evaluation")
    parser.add_argument("--dae_dir",  default="outputs/dae")
    parser.add_argument("--pes_dir",  default="outputs/pes")
    parser.add_argument("--rl_dir",   default="outputs/rl")
    parser.add_argument("--eval_dir", default="outputs/evaluation")
    args = parser.parse_args()
    main(args)