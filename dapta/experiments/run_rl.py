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

<<<<<<< Updated upstream
=======
METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
SURPRISAL_DIM      = 38
CLINICAL_THRESHOLD = 0.40

# ─────────────────────────────────────────────
# Statistical helpers (unchanged)
# ─────────────────────────────────────────────

def cohens_d_paired(before: np.ndarray, after: np.ndarray) -> float:
    diff = after - before
    std  = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 0 else 0.0


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return 0.0, 1.0
    try:
        stat, p = scipy_stats.wilcoxon(diff, alternative="greater")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def bonferroni(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    n         = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    sig       = [p <= alpha for p in corrected]
    return corrected, sig


def bootstrap_ci(
    diff: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng  = np.random.default_rng(seed)
    boot = [
        float(np.mean(rng.choice(diff, size=len(diff), replace=True)))
        for _ in range(n_resamples)
    ]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

>>>>>>> Stashed changes

# ─────────────────────────────────────────────
# Environment helpers (unchanged)
# ─────────────────────────────────────────────

def load_environments(
    transition_model: TransitionModel,
    initial_states: np.ndarray,
    cluster_labels: np.ndarray,
    episode_horizon: int = 20,
) -> Dict[int, List[TherapyEnv]]:
<<<<<<< Updated upstream
    """Build TherapyEnv instances grouped by cluster."""
    cluster_envs: Dict[int, List[TherapyEnv]] = {}
    n_clusters = int(cluster_labels.max()) + 1

=======
    n_clusters   = int(cluster_labels.max()) + 1
    cluster_envs = {}
>>>>>>> Stashed changes
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


<<<<<<< Updated upstream
def make_ddqn_agent(checkpoint_path: str = None) -> DDQNAgent:
    """Instantiate a DDQN agent with standard hyperparameters."""
=======
# ─────────────────────────────────────────────
# Agent factories — one for each configuration
# ─────────────────────────────────────────────

def make_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    """GRU-Dueling DDQN — original DAPTA architecture."""
>>>>>>> Stashed changes
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
        use_gru=True,
        checkpoint_path=checkpoint_path,
    )

<<<<<<< Updated upstream

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
=======

def make_no_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    """Standard Dueling DDQN without GRU — ablation agent."""
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
        use_gru=False,
        checkpoint_path=checkpoint_path,
    )


# ─────────────────────────────────────────────
# Episode runner (unchanged)
# ─────────────────────────────────────────────

def run_episodes(
    agent,
    envs:     List[TherapyEnv],
    is_ddqn:  bool = True,
    greedy:   bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    disc_imps = []
    surp_imps = []
    rewards   = []

    for env in envs:
        state, _       = env.reset()
        state_history  = []
        action_history = []
        total_reward   = 0.0
        initial_surp   = float(state[SURPRISAL_DIM])

        for _ in range(env.episode_horizon):
            if is_ddqn:
                history_tensor = agent.build_history_tensor(
                    state_history, action_history
                )
                action = agent.select_action(
                    state, history_tensor, greedy=greedy
                )
            else:
                action = agent.select_action(state)

            state_history.append(state.copy())
            action_history.append(action)

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break

        disc_imp   = env.get_cumulative_discourse_improvement()[:5]
        final_surp = float(state[SURPRISAL_DIM])

        disc_imps.append(disc_imp)
        surp_imps.append(initial_surp - final_surp)
        rewards.append(total_reward)

    return (
        np.stack(disc_imps),
        np.array(surp_imps, dtype=np.float32),
        np.array(rewards,   dtype=np.float32),
    )


# ─────────────────────────────────────────────
# RQ helpers
# ─────────────────────────────────────────────

def answer_rq2(
    dapta_disc: np.ndarray,
    rbde_disc:  np.ndarray,
    rts_disc:   np.ndarray,
    alpha:      float = 0.05,
) -> dict:
    out = {"DAPTA_vs_RBDE": {}, "DAPTA_vs_RTS": {}, "summary": {}}

    for comparison, baseline_disc in [
        ("DAPTA_vs_RBDE", rbde_disc),
        ("DAPTA_vs_RTS",  rts_disc),
    ]:
        p_values = []
        entries  = {}
        for i, metric in enumerate(METRIC_NAMES):
            d            = cohens_d_paired(baseline_disc[:, i], dapta_disc[:, i])
            stat, p      = wilcoxon_test(dapta_disc[:, i], baseline_disc[:, i])
            ci_lo, ci_hi = bootstrap_ci(dapta_disc[:, i] - baseline_disc[:, i])
            p_values.append(p)
            entries[metric] = {
                "dapta_mean":            round(float(np.mean(dapta_disc[:, i])),    4),
                "baseline_mean":         round(float(np.mean(baseline_disc[:, i])), 4),
                "cohens_d":              round(d, 3),
                "effect_size":           effect_size_label(d),
                "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
                "wilcoxon_stat":         round(stat, 4),
                "p_raw":                 round(float(p), 4),
                "ci_95":                 [round(ci_lo, 4), round(ci_hi, 4)],
            }
        p_corr, sig = bonferroni(p_values, alpha)
        for i, metric in enumerate(METRIC_NAMES):
            entries[metric]["p_corrected"] = round(p_corr[i], 4)
            entries[metric]["significant"] = sig[i]
        out[comparison] = entries
        n_sig  = sum(1 for v in entries.values() if v["significant"])
        n_clin = sum(1 for v in entries.values() if v["clinically_meaningful"])
        out["summary"][comparison] = {
            "n_significant":           n_sig,
            "n_clinically_meaningful": n_clin,
            "rq2_answered_positively": n_clin >= 2,
        }
    return out


def answer_rq3_cross_task(output_dir: Path) -> dict:
    """
    RQ3 — cross-task transfer analysis.

    Tests whether discourse metrics targeted by DAPTA in structured
    tasks (Cookie Theft, Cinderella, Sandwich) correlate with the same
    metrics in naturalistic free conversation segments.  A significant
    correlation indicates that DAPTA targets generalised discourse
    deficits rather than task-specific performance, supporting
    ecological validity of the reward signal.

    Surprisal is excluded from this analysis due to computational
    constraints (see Section 3.4.3).  LLM-based naturalness scoring
    is identified as future work.
    """
    csv_path = Path("outputs/dae/per_task_metrics.csv")
    if not csv_path.exists():
        logger.warning(
            "per_task_metrics.csv not found — skipping cross-task RQ3. "
            "Run experiments/extract_per_task_csv.py first."
        )
        return {"rq3_answered_positively": False, "note": "csv not found"}

    try:
        import pandas as pd
        from scipy import stats as scipy_stats

        df      = pd.read_csv(str(csv_path))
        test_df = df[df["split"] == "test"]

        structured_tasks   = ["cookie_theft", "cinderella", "sandwich"]
        naturalistic_tasks = ["conversation"]

        structured = (
            test_df[test_df["task"].isin(structured_tasks)]
            .groupby("participant_id")[["ciu_rate", "mlu_morphemes"]]
            .mean()
            .add_suffix("_structured")
        )
        naturalistic = (
            test_df[test_df["task"].isin(naturalistic_tasks)]
            .groupby("participant_id")[["ciu_rate", "mlu_morphemes"]]
            .mean()
            .add_suffix("_naturalistic")
        )

        combined = structured.join(naturalistic, how="inner")
        n        = len(combined)

        if n < 5:
            return {
                "rq3_answered_positively": False,
                "note": f"insufficient matched participants (n={n})",
            }

        results = {}
        any_sig = False

        for col_s, col_n, label in [
            ("ciu_rate_structured",   "ciu_rate_naturalistic",   "ciu_rate"),
            ("mlu_morphemes_structured", "mlu_morphemes_naturalistic", "mlu_morphemes"),
        ]:
            r, p = scipy_stats.spearmanr(combined[col_s], combined[col_n])
            sig  = bool(p < 0.05)
            if sig:
                any_sig = True
            results[label] = {
                "spearman_r":  round(float(r), 3),
                "p_value":     round(float(p), 4),
                "significant": sig,
                "n":           int(n),
                "interpretation": (
                    "structured task gains reflect generalised discourse deficit"
                    if r > 0.3 and sig
                    else "weak or no cross-task transfer detected"
                ),
            }

        out = {
            "method":      "cross_task_correlation",
            "n_matched":   int(n),
            "correlations": results,
            "rq3_answered_positively": any_sig,
            "note": (
                "Surprisal excluded due to computational constraints. "
                "Transfer assessed via structured-vs-naturalistic discourse "
                "metric correlation on held-out test set."
            ),
        }

        with open(output_dir / "rq3_transfer.json", "w") as f:
            json.dump(out, f, indent=2)

        logger.info(
            f"  RQ3 cross-task: n={n}, "
            f"CIU r={results['ciu_rate']['spearman_r']}, "
            f"p={results['ciu_rate']['p_value']} — "
            f"answered: {any_sig}"
        )
        return out

    except Exception as e:
        logger.warning(f"RQ3 cross-task analysis failed: {e}")
        return {"rq3_answered_positively": False, "error": str(e)}


def answer_rq4(
    dapta_disc:     np.ndarray,
    g_ddqn_disc:    np.ndarray,
    dapta_surp:     np.ndarray,
    g_ddqn_surp:    np.ndarray,
    cluster_labels: np.ndarray,
    profiles_data:  list,
    alpha:          float = 0.05,
) -> dict:
    out      = {}
    p_values = []
    entries  = {}

    for i, metric in enumerate(METRIC_NAMES):
        d            = cohens_d_paired(g_ddqn_disc[:, i], dapta_disc[:, i])
        stat, p      = wilcoxon_test(dapta_disc[:, i], g_ddqn_disc[:, i])
        ci_lo, ci_hi = bootstrap_ci(dapta_disc[:, i] - g_ddqn_disc[:, i])
        p_values.append(p)
        entries[metric] = {
            "dapta_mean":            round(float(np.mean(dapta_disc[:, i])),   4),
            "gddqn_mean":            round(float(np.mean(g_ddqn_disc[:, i])), 4),
            "cohens_d":              round(d, 3),
            "effect_size":           effect_size_label(d),
            "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
            "wilcoxon_stat":         round(stat, 4),
            "p_raw":                 round(float(p), 4),
            "ci_95":                 [round(ci_lo, 4), round(ci_hi, 4)],
        }

    p_corr, sig = bonferroni(p_values, alpha)
    for i, metric in enumerate(METRIC_NAMES):
        entries[metric]["p_corrected"] = round(p_corr[i], 4)
        entries[metric]["significant"] = sig[i]

    ciu_dapta = dapta_disc[:, 0]
    ciu_gddqn = g_ddqn_disc[:, 0]
    var_reduction = (
        (np.var(ciu_gddqn, ddof=1) - np.var(ciu_dapta, ddof=1))
        / max(np.var(ciu_gddqn, ddof=1), 1e-8) * 100
    )

    out["pooled_dapta_vs_gddqn"]      = entries
    out["ciu_variance_reduction_pct"] = round(float(var_reduction), 2)
    out["personalisation_beneficial"] = bool(
        np.mean(ciu_dapta) > np.mean(ciu_gddqn) and any(sig)
    )

    unique_clusters = np.unique(cluster_labels)
    per_cluster     = {}
    for c in unique_clusters:
        mask     = cluster_labels == c
        n        = int(mask.sum())
        if n == 0:
            continue
        d_ciu    = dapta_disc[mask, 0]
        g_ciu    = g_ddqn_disc[mask, 0]
        d_val    = cohens_d_paired(g_ciu, d_ciu)
        subtypes = [
            p.get("aphasia_subtype", "Other")
            for p, m in zip(profiles_data, mask) if m
        ]
        dominant      = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
        subtype_counts = {s: subtypes.count(s) for s in set(subtypes)}
        per_cluster[str(c)] = {
            "n":                int(n),
            "dominant_subtype": dominant,
            "subtype_counts":   subtype_counts,
            "dapta_ciu_mean":   round(float(np.mean(d_ciu)), 4),
            "gddqn_ciu_mean":   round(float(np.mean(g_ciu)), 4),
            "cohens_d_ciu":     round(d_val, 3),
            "dapta_wins":       bool(np.mean(d_ciu) > np.mean(g_ciu)),
        }

    wab_aqs = np.array([
        float(p.get("wab_aq") or 55.0) for p in profiles_data
    ], dtype=np.float32)

    personalisation_benefit = ciu_dapta - ciu_gddqn
    if len(wab_aqs) >= 5:
        r_wab, p_wab = scipy_stats.pearsonr(wab_aqs, personalisation_benefit)
        out["wabaq_moderates_personalisation"] = {
            "pearson_r":   round(float(r_wab), 3),
            "p_value":     round(float(p_wab), 4),
            "significant": bool(p_wab < 0.05),
            "interpretation": (
                "higher WAB-AQ patients benefit more from personalisation"
                if r_wab > 0.2 and p_wab < 0.05
                else "lower WAB-AQ patients benefit more from personalisation"
                if r_wab < -0.2 and p_wab < 0.05
                else "WAB-AQ does not significantly moderate personalisation benefit"
            ),
        }
        low_mask  = wab_aqs < 50
        high_mask = wab_aqs >= 50
        out["severity_subgroup_analysis"] = {
            "low_wabaq_n":           int(low_mask.sum()),
            "high_wabaq_n":          int(high_mask.sum()),
            "low_wabaq_dapta_mean":  round(float(np.mean(ciu_dapta[low_mask])),  4) if low_mask.sum() > 0 else None,
            "high_wabaq_dapta_mean": round(float(np.mean(ciu_dapta[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_gddqn_mean":  round(float(np.mean(ciu_gddqn[low_mask])),  4) if low_mask.sum() > 0 else None,
            "high_wabaq_gddqn_mean": round(float(np.mean(ciu_gddqn[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_d":  round(cohens_d_paired(ciu_gddqn[low_mask],  ciu_dapta[low_mask]),  3) if low_mask.sum() > 1 else None,
            "high_wabaq_d": round(cohens_d_paired(ciu_gddqn[high_mask], ciu_dapta[high_mask]), 3) if high_mask.sum() > 1 else None,
        }

    out["per_cluster"]            = per_cluster
    out["rq4_answered_positively"] = out["personalisation_beneficial"]
    return out


def answer_ablation(
    all_agent_results: Dict[str, np.ndarray],
    rbde_disc:         np.ndarray,
) -> dict:
    """
    Ablation study — compares all six agents across three axes:
      1. Algorithm:       GRU-DDQN vs No-GRU DDQN vs PPO
      2. Personalisation: cluster-specific vs generalised
      3. Memory:          GRU vs no-GRU (within DDQN family)
    """
    out = {}

    # Per-agent summary statistics
    per_agent = {}
    for agent_name, disc in all_agent_results.items():
        d_ciu = cohens_d_paired(rbde_disc[:, 0], disc[:, 0])
        per_agent[agent_name] = {
            metric: {
                "mean":     round(float(np.mean(disc[:, i])), 4),
                "cohens_d_vs_rbde": round(
                    cohens_d_paired(rbde_disc[:, i], disc[:, i]), 3
                ),
                "effect_size": effect_size_label(
                    cohens_d_paired(rbde_disc[:, i], disc[:, i])
                ),
            }
            for i, metric in enumerate(METRIC_NAMES)
        }
        per_agent[agent_name]["ciu_cohens_d_vs_rbde"] = round(d_ciu, 3)

    out["per_agent_summary"] = per_agent

    # Axis 1 — does GRU help? (personalised: DAPTA vs No-GRU-DAPTA)
    if "DAPTA" in all_agent_results and "NO_GRU_DAPTA" in all_agent_results:
        d = cohens_d_paired(
            all_agent_results["NO_GRU_DAPTA"][:, 0],
            all_agent_results["DAPTA"][:, 0],
        )
        stat, p = wilcoxon_test(
            all_agent_results["DAPTA"][:, 0],
            all_agent_results["NO_GRU_DAPTA"][:, 0],
        )
        out["gru_benefit_personalised"] = {
            "cohens_d":    round(d, 3),
            "wilcoxon_p":  round(float(p), 4),
            "significant": bool(p < 0.05),
            "gru_helps":   bool(d > 0.1 and p < 0.05),
            "interpretation": (
                "GRU adds meaningful benefit in personalised setting"
                if d > 0.1 and p < 0.05
                else "GRU does not meaningfully improve performance"
            ),
        }

    # Axis 1 — does GRU help? (generalised: G-DDQN vs No-GRU-G-DDQN)
    if "G_DDQN" in all_agent_results and "NO_GRU_G_DDQN" in all_agent_results:
        d = cohens_d_paired(
            all_agent_results["NO_GRU_G_DDQN"][:, 0],
            all_agent_results["G_DDQN"][:, 0],
        )
        stat, p = wilcoxon_test(
            all_agent_results["G_DDQN"][:, 0],
            all_agent_results["NO_GRU_G_DDQN"][:, 0],
        )
        out["gru_benefit_generalised"] = {
            "cohens_d":    round(d, 3),
            "wilcoxon_p":  round(float(p), 4),
            "significant": bool(p < 0.05),
            "gru_helps":   bool(d > 0.1 and p < 0.05),
        }

    # Axis 2 — does algorithm matter? (PPO-generalised vs G-DDQN)
    if "PPO_GENERALISED" in all_agent_results and "G_DDQN" in all_agent_results:
        d = cohens_d_paired(
            all_agent_results["G_DDQN"][:, 0],
            all_agent_results["PPO_GENERALISED"][:, 0],
        )
        stat, p = wilcoxon_test(
            all_agent_results["PPO_GENERALISED"][:, 0],
            all_agent_results["G_DDQN"][:, 0],
        )
        out["algorithm_effect_generalised"] = {
            "ppo_vs_gddqn_cohens_d": round(d, 3),
            "wilcoxon_p":            round(float(p), 4),
            "significant":           bool(p < 0.05),
            "ppo_wins":              bool(np.mean(all_agent_results["PPO_GENERALISED"][:, 0]) >
                                         np.mean(all_agent_results["G_DDQN"][:, 0])),
            "interpretation": (
                "PPO outperforms DDQN — algorithm choice matters"
                if d > 0.1 and p < 0.05
                else "No significant difference between PPO and DDQN"
            ),
        }

    # Axis 2 — algorithm in personalised setting (PPO-personalised vs DAPTA)
    if "PPO_PERSONALISED" in all_agent_results and "DAPTA" in all_agent_results:
        d = cohens_d_paired(
            all_agent_results["DAPTA"][:, 0],
            all_agent_results["PPO_PERSONALISED"][:, 0],
        )
        out["algorithm_effect_personalised"] = {
            "ppo_vs_dapta_cohens_d": round(d, 3),
            "ppo_wins": bool(
                np.mean(all_agent_results["PPO_PERSONALISED"][:, 0]) >
                np.mean(all_agent_results["DAPTA"][:, 0])
            ),
        }

    # Axis 3 — recommended configuration
    ciu_means = {
        name: float(np.mean(disc[:, 0]))
        for name, disc in all_agent_results.items()
    }
    best = max(ciu_means, key=ciu_means.get)
    out["recommended_configuration"] = {
        "agent":         best,
        "ciu_rate_mean": round(ciu_means[best], 4),
        "all_ciu_means": {k: round(v, 4) for k, v in ciu_means.items()},
    }

    return out


# ─────────────────────────────────────────────
# Main training + evaluation
# ─────────────────────────────────────────────

def main(args) -> None:
    cfg        = Config.load()
    output_dir = Path("outputs/rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 2b: RL Agent Training — Full Ablation")
    logger.info("=" * 60)

    # ── 1. Load PES outputs ──────────────────────────────────────
    logger.info("\n[1/6] Loading PES outputs...")
    pes_dir = Path("outputs/pes")
    if not (pes_dir / "transition_model.pt").exists():
        logger.error("PES outputs not found. Run experiments/run_pes.py first.")
        return

    pes_data       = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    initial_states = pes_data["initial_states"]
    cluster_labels = pes_data["cluster_labels"]

    with open("outputs/dae/patient_profiles.json") as f:
        profiles_data = json.load(f)

    transition_model = TransitionModel(
        hidden_sizes=(128, 64),
        dropout=0.2,
        checkpoint_path=str(pes_dir / "transition_model.pt"),
        device=args.device,
    )
    transition_model.load()
    logger.info("Transition model loaded.")

    # ── 2. Build environments ────────────────────────────────────
    logger.info("\n[2/6] Building environments per cluster...")
    cluster_envs = load_environments(
        transition_model=transition_model,
        initial_states=initial_states,
        cluster_labels=cluster_labels,
        episode_horizon=20,
    )
    all_envs   = [env for envs in cluster_envs.values() for env in envs]
    n_clusters = len(cluster_envs)
    logger.info(f"Total environments: {len(all_envs)}, clusters: {n_clusters}")

    # ── 3. Train DDQN agents (GRU + No-GRU) ─────────────────────
    logger.info(f"\n[3/6] Training DDQN agents ({args.total_steps} steps each)...")
    training_logs: Dict = {}

    # 3a — DAPTA: GRU-DDQN cluster-specific (personalised)
    logger.info("\n  3a. DAPTA — GRU-DDQN cluster-specific")
    cluster_agents: Dict[int, DDQNAgent] = {}
    for cluster_id, envs in cluster_envs.items():
        if not envs:
            continue
        ckpt  = str(output_dir / f"ddqn_cluster_{cluster_id}.pt")
        agent = make_gru_ddqn(checkpoint_path=ckpt)
        logs  = train_ddqn(
            agent=agent,
            envs=envs,
            total_steps=args.total_steps,
            eval_freq=max(100, args.total_steps // 50),
            n_eval_episodes=min(10, len(envs)),
        )
        agent.save()
        cluster_agents[cluster_id] = agent
        training_logs[f"dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    Cluster {cluster_id} saved.")

    # 3b — G-DDQN: GRU-DDQN generalised
    logger.info("\n  3b. G-DDQN — GRU-DDQN generalised")
    g_ddqn = make_gru_ddqn(checkpoint_path=str(output_dir / "ddqn_generalised.pt"))
    training_logs["g_ddqn"] = train_ddqn(
        agent=g_ddqn,
        envs=all_envs,
        total_steps=args.total_steps,
        eval_freq=max(100, args.total_steps // 50),
        n_eval_episodes=min(20, len(all_envs)),
    )
    g_ddqn.save()
    logger.info("    G-DDQN saved.")

    # 3c — No-GRU DAPTA: No-GRU DDQN cluster-specific (personalised)
    logger.info("\n  3c. No-GRU DAPTA — No-GRU DDQN cluster-specific")
    no_gru_cluster_agents: Dict[int, DDQNAgent] = {}
    for cluster_id, envs in cluster_envs.items():
        if not envs:
            continue
        ckpt  = str(output_dir / f"no_gru_ddqn_cluster_{cluster_id}.pt")
        agent = make_no_gru_ddqn(checkpoint_path=ckpt)
        logs  = train_ddqn(
            agent=agent,
            envs=envs,
            total_steps=args.total_steps,
            eval_freq=max(100, args.total_steps // 50),
            n_eval_episodes=min(10, len(envs)),
        )
        agent.save()
        no_gru_cluster_agents[cluster_id] = agent
        training_logs[f"no_gru_dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    No-GRU cluster {cluster_id} saved.")

    # 3d — No-GRU G-DDQN: No-GRU DDQN generalised
    logger.info("\n  3d. No-GRU G-DDQN — No-GRU DDQN generalised")
    no_gru_g_ddqn = make_no_gru_ddqn(
        checkpoint_path=str(output_dir / "no_gru_ddqn_generalised.pt")
    )
    training_logs["no_gru_g_ddqn"] = train_ddqn(
        agent=no_gru_g_ddqn,
        envs=all_envs,
        total_steps=args.total_steps,
        eval_freq=max(100, args.total_steps // 50),
        n_eval_episodes=min(20, len(all_envs)),
    )
    no_gru_g_ddqn.save()
    logger.info("    No-GRU G-DDQN saved.")

    # ── 4. Train PPO agents (generalised + personalised) ─────────
    logger.info(f"\n[4/6] Training PPO agents...")

    # 4a — PPO generalised (existing)
    ppo_generalised = None
    if not args.skip_ppo:
        logger.info("\n  4a. PPO generalised — pooled environments")
        ppo_save        = str(output_dir / "ppo_generalised")
        ppo_generalised = train_ppo(
            env=all_envs[0],
            total_steps=args.total_steps,
            save_path=ppo_save,
        )
        training_logs["ppo_generalised"] = {
            "status": "trained" if ppo_generalised else "skipped_no_sb3",
            "steps":  args.total_steps,
        }
        logger.info("    PPO generalised saved.")

        # 4b — PPO personalised — train per cluster
        logger.info("\n  4b. PPO personalised — cluster-specific environments")
        ppo_cluster_models = {}
        for cluster_id, envs in cluster_envs.items():
            if not envs:
                continue
            ppo_c_save  = str(output_dir / f"ppo_cluster_{cluster_id}")
            ppo_cluster = train_ppo(
                env=envs[0],
                total_steps=args.total_steps,
                save_path=ppo_c_save,
            )
            if ppo_cluster:
                ppo_cluster_models[cluster_id] = ppo_cluster
                training_logs[f"ppo_cluster_{cluster_id}"] = {
                    "status": "trained",
                    "steps":  args.total_steps,
                }
                logger.info(f"    PPO cluster {cluster_id} saved.")
    else:
        logger.info("  Skipping PPO (--skip_ppo flag set).")
        ppo_cluster_models = {}

    # ── 5. Collect improvement arrays for all agents ─────────────
    logger.info("\n[5/6] Collecting per-patient improvement arrays...")

    # DAPTA (GRU personalised)
    dapta_disc_list, dapta_surp_list, dapta_label_list = [], [], []
    for cluster_id, agent in cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        dapta_disc_list.append(disc)
        dapta_surp_list.append(surp)
        dapta_label_list.extend([cluster_id] * len(envs))
    dapta_disc   = np.concatenate(dapta_disc_list,  axis=0) if dapta_disc_list  else np.zeros((0, 5))
    dapta_surp   = np.concatenate(dapta_surp_list,  axis=0) if dapta_surp_list  else np.zeros(0)
    dapta_labels = np.array(dapta_label_list, dtype=int)

    # G-DDQN (GRU generalised)
    g_ddqn.eps = 0.0
    g_ddqn_disc, g_ddqn_surp, _ = run_episodes(g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # No-GRU DAPTA (personalised)
    no_gru_dapta_disc_list, no_gru_dapta_surp_list = [], []
    for cluster_id, agent in no_gru_cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        no_gru_dapta_disc_list.append(disc)
        no_gru_dapta_surp_list.append(surp)
    no_gru_dapta_disc = np.concatenate(no_gru_dapta_disc_list, axis=0) if no_gru_dapta_disc_list else np.zeros((0, 5))

    # No-GRU G-DDQN (generalised)
    no_gru_g_ddqn.eps = 0.0
    no_gru_g_ddqn_disc, _, _ = run_episodes(no_gru_g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # PPO generalised
    ppo_gen_disc = np.zeros((0, 5))
    if ppo_generalised:
        ppo_gen_disc_list = []
        for env in all_envs:
            disc, _, _ = run_episodes(ppo_generalised, [env], is_ddqn=False, greedy=True)
            ppo_gen_disc_list.append(disc)
        ppo_gen_disc = np.concatenate(ppo_gen_disc_list, axis=0)

    # PPO personalised
    ppo_pers_disc = np.zeros((0, 5))
    if ppo_cluster_models:
        ppo_pers_disc_list = []
        for cluster_id, ppo_model in ppo_cluster_models.items():
            envs = cluster_envs[cluster_id]
            if not envs:
                continue
            for env in envs:
                disc, _, _ = run_episodes(ppo_model, [env], is_ddqn=False, greedy=True)
                ppo_pers_disc_list.append(disc)
        if ppo_pers_disc_list:
            ppo_pers_disc = np.concatenate(ppo_pers_disc_list, axis=0)

    # Baselines
>>>>>>> Stashed changes
    rbde = RuleBasedBaseline()
    rts = RandomBaseline()
<<<<<<< Updated upstream
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
=======
    rts_disc_list, rts_surp_list = [], []
    for env in all_envs:
        state, _ = env.reset()
        initial_surp = float(state[SURPRISAL_DIM])
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rts_disc_list.append(env.get_cumulative_discourse_improvement()[:5])
        rts_surp_list.append(initial_surp - float(state[SURPRISAL_DIM]))
    rts_disc = np.stack(rts_disc_list)

    # Save all improvement arrays
    np.save(str(output_dir / "improvements_DAPTA.npy"),         dapta_disc)
    np.save(str(output_dir / "improvements_G_DDQN.npy"),        g_ddqn_disc)
    np.save(str(output_dir / "improvements_NO_GRU_DAPTA.npy"),  no_gru_dapta_disc)
    np.save(str(output_dir / "improvements_NO_GRU_G_DDQN.npy"), no_gru_g_ddqn_disc)
    np.save(str(output_dir / "improvements_PPO_GEN.npy"),        ppo_gen_disc)
    np.save(str(output_dir / "improvements_PPO_PERS.npy"),       ppo_pers_disc)
    np.save(str(output_dir / "improvements_RBDE.npy"),           rbde_disc)
    np.save(str(output_dir / "improvements_RTS.npy"),            rts_disc)

    # Save per-patient CIU improvement for RQ3 Part 2
    import pandas as pd
    with open("outputs/dae/patient_profiles.json") as f:
        all_profiles = json.load(f)
    with open("outputs/dae/splits.json") as f:
        splits = json.load(f)
    test_pids = set(splits.get("test", []))
    test_profiles = [p for p in all_profiles if p["participant_id"] in test_pids]
    n_save = min(len(test_profiles), len(dapta_disc))
    rl_results_df = pd.DataFrame({
        "participant_id": [p["participant_id"] for p in test_profiles[:n_save]],
        "ciu_improvement": dapta_disc[:n_save, 0],
        "mc_improvement":  dapta_disc[:n_save, 1],
        "mlu_improvement": dapta_disc[:n_save, 2],
    })
    rl_results_df.to_csv(str(output_dir / "dapta_test_results.csv"), index=False)
    logger.info("  Saved dapta_test_results.csv for RQ3 Part 2.")

    # ── 6. Compute all RQ statistics ─────────────────────────────
    logger.info("\n[6/6] Computing RQ2 / RQ3 / RQ4 / Ablation statistics...")

    # RQ2 — DAPTA vs baselines
    rq2 = answer_rq2(dapta_disc, rbde_disc, rts_disc)
    with open(output_dir / "rq2_results.json", "w") as f:
        json.dump(rq2, f, indent=2)

    # RQ3 — cross-task transfer (replaces surprisal)
    rq3 = answer_rq3_cross_task(output_dir)

    # RQ4 — personalisation vs generalised
    min_n = min(len(dapta_disc), len(g_ddqn_disc))
    rq4   = answer_rq4(
        dapta_disc=dapta_disc[:min_n],
        g_ddqn_disc=g_ddqn_disc[:min_n],
        dapta_surp=dapta_surp[:min_n],
        g_ddqn_surp=g_ddqn_surp[:min_n],
        cluster_labels=dapta_labels[:min_n],
        profiles_data=profiles_data[:min_n],
    )
    with open(output_dir / "rq4_personalisation.json", "w") as f:
        json.dump(rq4, f, indent=2)

    # Ablation — all six agents
    all_agent_results = {"DAPTA": dapta_disc, "G_DDQN": g_ddqn_disc}
    if len(no_gru_dapta_disc)  > 0: all_agent_results["NO_GRU_DAPTA"]  = no_gru_dapta_disc
    if len(no_gru_g_ddqn_disc) > 0: all_agent_results["NO_GRU_G_DDQN"] = no_gru_g_ddqn_disc
    if len(ppo_gen_disc)        > 0: all_agent_results["PPO_GENERALISED"]  = ppo_gen_disc
    if len(ppo_pers_disc)       > 0: all_agent_results["PPO_PERSONALISED"] = ppo_pers_disc

    ablation = answer_ablation(all_agent_results, rbde_disc)
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # Cluster performance for evaluation
    Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)
    with open("outputs/evaluation/cluster_performance.json", "w") as f:
        json.dump(rq4["per_cluster"], f, indent=2)

    # Full results table
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
        if isinstance(obj, list):  return [convert(x) for x in obj]
        if isinstance(obj, dict):  return {k: convert(v) for k, v in obj.items()}
        return obj

    results_table = {
        "per_agent_means": {
            name: {
                metric: round(float(np.mean(arr[:, i])), 4)
                for i, metric in enumerate(METRIC_NAMES)
            }
            for name, arr in all_agent_results.items()
            if len(arr) > 0
        },
        "baselines": {
            "RBDE": {metric: round(float(np.mean(rbde_disc[:, i])), 4) for i, metric in enumerate(METRIC_NAMES)},
            "RTS":  {metric: round(float(np.mean(rts_disc[:, i])),  4) for i, metric in enumerate(METRIC_NAMES)},
        },
        "ablation_summary": ablation,
        "rq2_summary": rq2["summary"],
        "rq3_summary": {
            "method":              rq3.get("method", "cross_task"),
            "answered_positively": rq3.get("rq3_answered_positively", False),
            "n_matched":           rq3.get("n_matched", 0),
        },
        "rq4_summary": {
            "answered_positively":        rq4["personalisation_beneficial"],
            "ciu_variance_reduction_pct": rq4["ciu_variance_reduction_pct"],
        },
        "recommended_config": ablation.get("recommended_configuration", {}),
>>>>>>> Stashed changes
        "config": {
            "total_steps": args.total_steps,
            "episode_horizon": 20,
<<<<<<< Updated upstream
            "n_clusters": 8,
            "n_environments": len(all_envs),
        }
=======
            "n_clusters":      n_clusters,
            "n_environments":  len(all_envs),
        },
>>>>>>> Stashed changes
    }
    with open(output_dir / "results_table.json", "w") as f:
        json.dump(convert(results_table), f, indent=2, default=str)

<<<<<<< Updated upstream
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
=======
    with open(output_dir / "training_logs.json", "w") as f:
        json.dump(convert(training_logs), f, indent=2)

    # ── Summary print ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DAPTA — FULL ABLATION RESULTS")
    print("=" * 70)

    print("\nAll agents — CIU rate mean improvement:")
    for name, arr in all_agent_results.items():
        if len(arr) > 0:
            d = cohens_d_paired(rbde_disc[:, 0], arr[:, 0])
            print(f"  {name:22} mean={np.mean(arr[:,0]):.4f}  d_vs_RBDE={d:.3f}")
    print(f"  {'RBDE':22} mean={np.mean(rbde_disc[:,0]):.4f}  (baseline)")
    print(f"  {'RTS':22} mean={np.mean(rts_disc[:,0]):.4f}  (baseline)")

    print(f"\nRecommended config: {ablation.get('recommended_configuration', {}).get('agent', 'N/A')}")

    print("\nRQ2:")
    for comp in ["DAPTA_vs_RBDE", "DAPTA_vs_RTS"]:
        s = rq2["summary"][comp]
        print(f"  {comp}: {s['n_clinically_meaningful']}/5 clinically meaningful — "
              f"{'ANSWERED ✓' if s['rq2_answered_positively'] else 'NOT ANSWERED'}")

    print(f"\nRQ3 (cross-task transfer): answered={rq3.get('rq3_answered_positively', False)}, "
          f"n={rq3.get('n_matched', 0)}")

    print(f"\nRQ4: Personalisation beneficial={rq4['personalisation_beneficial']}, "
          f"variance reduction={rq4['ciu_variance_reduction_pct']:.1f}%")

    print("\nGRU ablation:")
    if "gru_benefit_personalised" in ablation:
        g = ablation["gru_benefit_personalised"]
        print(f"  Personalised: d={g['cohens_d']}, p={g['wilcoxon_p']} — {g['interpretation']}")
    if "gru_benefit_generalised" in ablation:
        g = ablation["gru_benefit_generalised"]
        print(f"  Generalised:  d={g['cohens_d']}, p={g['wilcoxon_p']}")

    if "algorithm_effect_generalised" in ablation:
        a = ablation["algorithm_effect_generalised"]
        print(f"\nAlgorithm effect (PPO vs DDQN): d={a['ppo_vs_gddqn_cohens_d']}, "
              f"p={a['wilcoxon_p']} — {a['interpretation']}")

    print("\n" + "=" * 70)
    print(f"Outputs saved to: {output_dir}/")
    print("  rq2_results.json")
    print("  rq3_transfer.json")
    print("  rq4_personalisation.json")
    print("  ablation_results.json")
    print("  results_table.json")
    print("  dapta_test_results.csv")
    print("=" * 70)
>>>>>>> Stashed changes
    logger.info("Next step: python experiments/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPTA Phase 2b: RL Agent Training — Full Ablation"
    )
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
