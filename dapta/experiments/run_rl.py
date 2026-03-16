"""
Phase 2b: RL Agent Training.

Trains and evaluates all agents to answer:
  RQ2: Does RL outperform rule-based/random sequencing?
  RQ3: Do discourse gains transfer to naturalistic speech (surprisal)?
  RQ4: Does patient-specific RL beat generalised RL, and what patient
       factors drive the difference?

Requires:
  outputs/pes/  (from run_pes.py)

Outputs:
  outputs/rl/ddqn_cluster_{i}.pt
  outputs/rl/ddqn_generalised.pt
  outputs/rl/ppo_agent/
  outputs/rl/training_logs.json
  outputs/rl/results_table.json
  outputs/rl/rq2_results.json
  outputs/rl/rq3_transfer.json
  outputs/rl/rq4_personalisation.json
  outputs/rl/cluster_performance.json

Usage:
  python experiments/run_rl.py
  python experiments/run_rl.py --total_steps 5000  # quick test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats

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

# Metric names matching the first 5 dims of the discourse block
METRIC_NAMES = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]

# Surprisal is dim 38 in the full state vector (SLICE_STATIC.start + 2)
SURPRISAL_DIM = 38

# iTalkBetter clinical benchmark (Upton et al., 2024)
CLINICAL_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
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
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------

def load_environments(
    transition_model: TransitionModel,
    initial_states:   np.ndarray,
    cluster_labels:   np.ndarray,
    episode_horizon:  int = 20,
) -> Dict[int, List[TherapyEnv]]:
    n_clusters   = int(cluster_labels.max()) + 1
    cluster_envs = {}

    for cluster_id in range(n_clusters):
        mask           = cluster_labels == cluster_id
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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_ddqn_agent(checkpoint_path: str = None) -> DDQNAgent:
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


# ---------------------------------------------------------------------------
# Full episode runner — returns per-patient improvement arrays
# ---------------------------------------------------------------------------

def run_episodes(
    agent,
    envs: List[TherapyEnv],
    is_ddqn: bool = True,
    greedy: bool  = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one episode per environment.

    Returns
    -------
    discourse_improvements : (N, 5)  — first 5 discourse dims
    surprisal_improvements : (N,)    — surprisal delta per patient
    total_rewards          : (N,)
    """
    disc_imps  = []
    surp_imps  = []
    rewards    = []

    for env in envs:
        state, _ = env.reset()
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

        # Discourse improvement — first 5 dims of discourse block
        disc_imp = env.get_cumulative_discourse_improvement()[:5]
        final_surp = float(state[SURPRISAL_DIM])

        disc_imps.append(disc_imp)
        surp_imps.append(initial_surp - final_surp)  # positive = improvement
        rewards.append(total_reward)

    return (
        np.stack(disc_imps),
        np.array(surp_imps, dtype=np.float32),
        np.array(rewards,   dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# RQ2: RL vs rule-based and random baselines
# ---------------------------------------------------------------------------

def answer_rq2(
    dapta_disc:  np.ndarray,
    rbde_disc:   np.ndarray,
    rts_disc:    np.ndarray,
    alpha:       float = 0.05,
) -> dict:
    """
    RQ2: Does RL outperform rule-based/random sequencing?

    For each metric, compute Cohen's d and Wilcoxon test
    (DAPTA vs RBDE, DAPTA vs RTS) with Bonferroni correction.
    """
    out = {
        "DAPTA_vs_RBDE": {},
        "DAPTA_vs_RTS":  {},
        "summary": {},
    }

    for comparison, baseline_disc in [
        ("DAPTA_vs_RBDE", rbde_disc),
        ("DAPTA_vs_RTS",  rts_disc),
    ]:
        p_values = []
        entries  = {}

        for i, metric in enumerate(METRIC_NAMES):
            d          = cohens_d_paired(baseline_disc[:, i], dapta_disc[:, i])
            stat, p    = wilcoxon_test(dapta_disc[:, i], baseline_disc[:, i])
            ci_lo, ci_hi = bootstrap_ci(dapta_disc[:, i] - baseline_disc[:, i])
            p_values.append(p)
            entries[metric] = {
                "dapta_mean":  round(float(np.mean(dapta_disc[:, i])),    4),
                "baseline_mean": round(float(np.mean(baseline_disc[:, i])), 4),
                "cohens_d":    round(d, 3),
                "effect_size": effect_size_label(d),
                "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
                "wilcoxon_stat": round(stat, 4),
                "p_raw":       round(float(p), 4),
                "ci_95":       [round(ci_lo, 4), round(ci_hi, 4)],
            }

        p_corr, sig = bonferroni(p_values, alpha)
        for i, metric in enumerate(METRIC_NAMES):
            entries[metric]["p_corrected"] = round(p_corr[i], 4)
            entries[metric]["significant"] = sig[i]

        out[comparison] = entries

    # Summary: how many metrics are clinically meaningful?
    for comp in ["DAPTA_vs_RBDE", "DAPTA_vs_RTS"]:
        n_sig  = sum(1 for v in out[comp].values() if v["significant"])
        n_clin = sum(1 for v in out[comp].values() if v["clinically_meaningful"])
        out["summary"][comp] = {
            "n_significant":           n_sig,
            "n_clinically_meaningful": n_clin,
            "rq2_answered_positively": n_clin >= 2,
        }

    return out


# ---------------------------------------------------------------------------
# RQ3: Transfer to naturalistic speech (surprisal)
# ---------------------------------------------------------------------------

def answer_rq3(
    dapta_disc:  np.ndarray,
    dapta_surp:  np.ndarray,
    rbde_disc:   np.ndarray,
    rbde_surp:   np.ndarray,
) -> dict:
    """
    RQ3: Do discourse gains transfer to naturalistic connected speech?

    Two tests:
    1. Does DAPTA produce significantly greater surprisal improvement
       than RBDE? (direct comparison)
    2. Within DAPTA patients, does discourse improvement (CIU) correlate
       with surprisal improvement? (transfer correlation)
    """
    out = {}

    # Test 1: DAPTA vs RBDE on surprisal improvement
    d_surp   = cohens_d_paired(rbde_surp, dapta_surp)
    stat, p  = wilcoxon_test(dapta_surp, rbde_surp)
    ci_lo, ci_hi = bootstrap_ci(dapta_surp - rbde_surp)

    out["surprisal_improvement"] = {
        "dapta_mean_surp_improvement":  round(float(np.mean(dapta_surp)), 4),
        "rbde_mean_surp_improvement":   round(float(np.mean(rbde_surp)),  4),
        "cohens_d":                     round(d_surp, 3),
        "effect_size":                  effect_size_label(d_surp),
        "clinically_meaningful":        abs(d_surp) >= CLINICAL_THRESHOLD,
        "wilcoxon_p":                   round(float(p), 4),
        "significant":                  bool(p < 0.05),
        "ci_95":                        [round(ci_lo, 4), round(ci_hi, 4)],
    }

    # Test 2: Correlation between CIU improvement and surprisal improvement
    ciu_imp = dapta_disc[:, 0]  # CIU rate is index 0
    if len(ciu_imp) >= 5:
        r, p_corr = scipy_stats.pearsonr(ciu_imp, dapta_surp)
        out["discourse_to_surprisal_correlation"] = {
            "pearson_r":   round(float(r),      3),
            "p_value":     round(float(p_corr), 4),
            "significant": bool(p_corr < 0.05),
            "n":           int(len(ciu_imp)),
            "interpretation": (
                "discourse gains transfer to naturalistic speech"
                if r > 0.3 and p_corr < 0.05
                else "weak or no transfer detected"
            ),
        }
    else:
        out["discourse_to_surprisal_correlation"] = {
            "note": "insufficient data for correlation"
        }

    # RQ3 answered positively if either test shows significant transfer
    surp_sig  = out["surprisal_improvement"]["significant"]
    corr_sig  = out.get("discourse_to_surprisal_correlation", {}).get("significant", False)
    out["rq3_answered_positively"] = bool(surp_sig or corr_sig)

    return out


# ---------------------------------------------------------------------------
# RQ4: Patient-specific vs generalised RL
# ---------------------------------------------------------------------------

def answer_rq4(
    dapta_disc:    np.ndarray,
    g_ddqn_disc:   np.ndarray,
    dapta_surp:    np.ndarray,
    g_ddqn_surp:   np.ndarray,
    cluster_labels: np.ndarray,
    profiles_data:  list,
    alpha:          float = 0.05,
) -> dict:
    """
    RQ4: Does patient-specific RL beat generalised RL?
    And what patient factors drive the difference?
    """
    out = {}

    # --- Pooled comparison: DAPTA vs G-DDQN ---
    p_values = []
    entries  = {}

    for i, metric in enumerate(METRIC_NAMES):
        d        = cohens_d_paired(g_ddqn_disc[:, i], dapta_disc[:, i])
        stat, p  = wilcoxon_test(dapta_disc[:, i], g_ddqn_disc[:, i])
        ci_lo, ci_hi = bootstrap_ci(dapta_disc[:, i] - g_ddqn_disc[:, i])
        p_values.append(p)
        entries[metric] = {
            "dapta_mean":  round(float(np.mean(dapta_disc[:, i])),   4),
            "gddqn_mean":  round(float(np.mean(g_ddqn_disc[:, i])),  4),
            "cohens_d":    round(d, 3),
            "effect_size": effect_size_label(d),
            "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
            "wilcoxon_stat": round(stat, 4),
            "p_raw":       round(float(p), 4),
            "ci_95":       [round(ci_lo, 4), round(ci_hi, 4)],
        }

    p_corr, sig = bonferroni(p_values, alpha)
    for i, metric in enumerate(METRIC_NAMES):
        entries[metric]["p_corrected"] = round(p_corr[i], 4)
        entries[metric]["significant"] = sig[i]

    # Variance reduction: does DAPTA reduce patient-level variability?
    ciu_dapta = dapta_disc[:, 0]
    ciu_gddqn = g_ddqn_disc[:, 0]
    var_reduction = (
        (np.var(ciu_gddqn, ddof=1) - np.var(ciu_dapta, ddof=1))
        / max(np.var(ciu_gddqn, ddof=1), 1e-8) * 100
    )

    out["pooled_dapta_vs_gddqn"] = entries
    out["ciu_variance_reduction_pct"] = round(float(var_reduction), 2)
    out["personalisation_beneficial"] = bool(
        np.mean(ciu_dapta) > np.mean(ciu_gddqn)
        and any(sig)
    )

    # --- Per-cluster breakdown ---
    unique_clusters = np.unique(cluster_labels)
    per_cluster     = {}

    for c in unique_clusters:
        mask    = cluster_labels == c
        n       = int(mask.sum())
        if n == 0:
            continue

        d_ciu = dapta_disc[mask, 0]
        g_ciu = g_ddqn_disc[mask, 0]
        d_val = cohens_d_paired(g_ciu, d_ciu)

        # Dominant subtype in this cluster
        cluster_profiles = [
            p for p, m in zip(profiles_data, mask) if m
        ]
        subtypes = [p.get("aphasia_subtype", "Other") for p in cluster_profiles]
        dominant = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
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

    out["per_cluster"] = per_cluster

    # --- Patient factor analysis: WAB-AQ moderates personalisation benefit ---
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

        # Split by severity: low (<50) vs high (>=50) WAB-AQ
        low_mask  = wab_aqs < 50
        high_mask = wab_aqs >= 50

        out["severity_subgroup_analysis"] = {
            "low_wabaq_n":           int(low_mask.sum()),
            "high_wabaq_n":          int(high_mask.sum()),
            "low_wabaq_dapta_mean":  round(float(np.mean(ciu_dapta[low_mask])),  4) if low_mask.sum() > 0 else None,
            "high_wabaq_dapta_mean": round(float(np.mean(ciu_dapta[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_gddqn_mean":  round(float(np.mean(ciu_gddqn[low_mask])),  4) if low_mask.sum() > 0 else None,
            "high_wabaq_gddqn_mean": round(float(np.mean(ciu_gddqn[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_d":           round(cohens_d_paired(ciu_gddqn[low_mask],  ciu_dapta[low_mask]),  3) if low_mask.sum() > 1 else None,
            "high_wabaq_d":          round(cohens_d_paired(ciu_gddqn[high_mask], ciu_dapta[high_mask]), 3) if high_mask.sum() > 1 else None,
        }

    out["rq4_answered_positively"] = out["personalisation_beneficial"]

    return out
def run_eval_only(args) -> None:
    """
    Load saved agents and compute RQ2-4 results on training environments.
    Skips all training — agents must already be saved in outputs/rl/.
    """
    from scipy import stats as scipy_stats

    output_dir = Path("outputs/rl")
    pes_dir    = Path("outputs/pes")

    # Load environments
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

    cluster_envs = load_environments(
        transition_model=transition_model,
        initial_states=initial_states,
        cluster_labels=cluster_labels,
        episode_horizon=20,
    )
    all_envs   = [env for envs in cluster_envs.values() for env in envs]
    n_clusters = len(cluster_envs)

    # Load saved agents
    from dapta.dae.state_builder import STATE_DIM
    from dapta.prta.action_space import N_ACTIONS

    cluster_agents = {}
    for cluster_id in range(n_clusters):
        ckpt = output_dir / f"ddqn_cluster_{cluster_id}.pt"
        if not ckpt.exists():
            logger.warning(f"No saved agent for cluster {cluster_id}, skipping.")
            continue
        agent = make_ddqn_agent(checkpoint_path=str(ckpt))
        agent.load()
        agent.eps = 0.0
        cluster_agents[cluster_id] = agent
        logger.info(f"  Loaded cluster {cluster_id} agent from {ckpt}")

    g_ddqn_ckpt = output_dir / "ddqn_generalised.pt"
    g_ddqn = make_ddqn_agent(checkpoint_path=str(g_ddqn_ckpt))
    g_ddqn.load()
    g_ddqn.eps = 0.0
    logger.info(f"  Loaded G-DDQN from {g_ddqn_ckpt}")

    # Collect per-patient improvements
    SURPRISAL_DIM  = 38
    METRIC_NAMES   = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
    CLINICAL_THRESHOLD = 0.40

    def run_episode(agent, env, is_ddqn=True):
        state, _       = env.reset()
        state_history  = []
        action_history = []
        initial_surp   = float(state[SURPRISAL_DIM])
        for _ in range(env.episode_horizon):
            if is_ddqn:
                history_tensor = agent.build_history_tensor(state_history, action_history)
                action         = agent.select_action(state, history_tensor, greedy=True)
            else:
                action = agent.select_action(state)
            state_history.append(state.copy())
            action_history.append(action)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        disc_imp = env.get_cumulative_discourse_improvement()[:5]
        surp_imp = initial_surp - float(state[SURPRISAL_DIM])
        return disc_imp, surp_imp

    # DAPTA
    dapta_disc_list, dapta_surp_list, dapta_labels = [], [], []
    for cluster_id, agent in cluster_agents.items():
        for env in cluster_envs[cluster_id]:
            d, s = run_episode(agent, env)
            dapta_disc_list.append(d)
            dapta_surp_list.append(s)
            dapta_labels.append(cluster_id)

    dapta_disc   = np.stack(dapta_disc_list)
    dapta_surp   = np.array(dapta_surp_list, dtype=np.float32)
    dapta_labels = np.array(dapta_labels, dtype=int)

    # G-DDQN
    g_disc_list, g_surp_list = [], []
    for env in all_envs:
        d, s = run_episode(g_ddqn, env)
        g_disc_list.append(d)
        g_surp_list.append(s)
    g_ddqn_disc = np.stack(g_disc_list)
    g_ddqn_surp = np.array(g_surp_list, dtype=np.float32)

    # RBDE
    rbde = RuleBasedBaseline()
    rbde_disc_list, rbde_surp_list = [], []
    for env in all_envs:
        state, _ = env.reset()
        rbde.reset()
        initial_surp = float(state[SURPRISAL_DIM])
        for _ in range(env.episode_horizon):
            action = rbde.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rbde_disc_list.append(env.get_cumulative_discourse_improvement()[:5])
        rbde_surp_list.append(initial_surp - float(state[SURPRISAL_DIM]))
    rbde_disc = np.stack(rbde_disc_list)
    rbde_surp = np.array(rbde_surp_list, dtype=np.float32)

    # RTS
    rts = RandomBaseline()
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
    rts_surp = np.array(rts_surp_list, dtype=np.float32)

    # Save improvement arrays
    np.save(str(output_dir / "improvements_DAPTA.npy"),  dapta_disc)
    np.save(str(output_dir / "improvements_G_DDQN.npy"), g_ddqn_disc)
    np.save(str(output_dir / "improvements_RBDE.npy"),   rbde_disc)
    np.save(str(output_dir / "improvements_RTS.npy"),    rts_disc)
    

    # RQ2
    rq2 = {"DAPTA_vs_RBDE": {}, "DAPTA_vs_RTS": {}, "summary": {}}
    for comp, baseline_disc in [("DAPTA_vs_RBDE", rbde_disc), ("DAPTA_vs_RTS", rts_disc)]:
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
                "p_raw":                 round(float(p), 4),
                "ci_95":                 [round(ci_lo, 4), round(ci_hi, 4)],
            }
        p_corr, sig = bonferroni(p_values)
        for i, metric in enumerate(METRIC_NAMES):
            entries[metric]["p_corrected"] = round(p_corr[i], 4)
            entries[metric]["significant"] = sig[i]
        rq2[comp] = entries
        n_clin = sum(1 for v in entries.values() if v["clinically_meaningful"])
        rq2["summary"][comp] = {
            "n_clinically_meaningful": n_clin,
            "rq2_answered_positively": n_clin >= 2,
        }

    with open(output_dir / "rq2_results.json", "w") as f:
        json.dump(rq2, f, indent=2)

    # RQ3
    d_surp       = cohens_d_paired(rbde_surp, dapta_surp)
    stat, p      = wilcoxon_test(dapta_surp, rbde_surp)
    ci_lo, ci_hi = bootstrap_ci(dapta_surp - rbde_surp)
    ciu_imp      = dapta_disc[:, 0]
    r_corr, p_corr_val = scipy_stats.pearsonr(ciu_imp, dapta_surp) if len(ciu_imp) >= 5 else (0, 1)

    rq3 = {
        "surprisal_improvement": {
            "dapta_mean": round(float(np.mean(dapta_surp)), 4),
            "rbde_mean":  round(float(np.mean(rbde_surp)),  4),
            "cohens_d":   round(d_surp, 3),
            "wilcoxon_p": round(float(p), 4),
            "significant": bool(p < 0.05),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "discourse_to_surprisal_correlation": {
            "pearson_r": round(float(r_corr), 3),
            "p_value":   round(float(p_corr_val), 4),
            "significant": bool(p_corr_val < 0.05),
        },
        "rq3_answered_positively": bool(p < 0.05 or p_corr_val < 0.05),
    }
    with open(output_dir / "rq3_transfer.json", "w") as f:
        json.dump(rq3, f, indent=2)

    # RQ4
    ciu_dapta = dapta_disc[:, 0]
    ciu_gddqn = g_ddqn_disc[:, 0]
    min_n     = min(len(ciu_dapta), len(ciu_gddqn))
    ciu_dapta = ciu_dapta[:min_n]
    ciu_gddqn = ciu_gddqn[:min_n]

    d_ciu        = cohens_d_paired(ciu_gddqn, ciu_dapta)
    stat, p      = wilcoxon_test(ciu_dapta, ciu_gddqn)
    ci_lo, ci_hi = bootstrap_ci(ciu_dapta - ciu_gddqn)
    var_red      = (np.var(ciu_gddqn, ddof=1) - np.var(ciu_dapta, ddof=1)) / max(np.var(ciu_gddqn, ddof=1), 1e-8) * 100

    # Per-cluster
    per_cluster = {}
    for c in np.unique(dapta_labels):
        mask      = dapta_labels == c
        d_arr     = dapta_disc[mask, 0]
        g_arr     = g_ddqn_disc[mask[:min_n], 0] if mask.sum() <= min_n else g_ddqn_disc[:mask.sum(), 0]
        g_arr     = g_ddqn_disc[mask, 0] if len(g_ddqn_disc) > mask.sum() else g_ddqn_disc[:mask.sum(), 0]
        subtypes  = [profiles_data[i].get("aphasia_subtype", "Other") for i, m in enumerate(mask) if m]
        dominant  = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
        sc        = {s: subtypes.count(s) for s in set(subtypes)}
        per_cluster[str(c)] = {
            "n":                int(mask.sum()),
            "dominant_subtype": dominant,
            "subtype_counts":   sc,
            "dapta_ciu_mean":   round(float(np.mean(d_arr)), 4),
            "gddqn_ciu_mean":   round(float(np.mean(g_arr)), 4),
            "cohens_d_ciu":     round(cohens_d_paired(g_arr, d_arr), 3),
            "dapta_wins":       bool(np.mean(d_arr) > np.mean(g_arr)),
        }

    wab_aqs = np.array([
        float(p.get("wab_aq") or 55.0)
        for p in profiles_data[:min_n]
    ])
    r_wab, p_wab = scipy_stats.pearsonr(wab_aqs, ciu_dapta - ciu_gddqn) if len(wab_aqs) >= 5 else (0, 1)

    rq4 = {
        "pooled": {
            "dapta_ciu_mean":  round(float(np.mean(ciu_dapta)), 4),
            "gddqn_ciu_mean":  round(float(np.mean(ciu_gddqn)), 4),
            "cohens_d":        round(d_ciu, 3),
            "wilcoxon_p":      round(float(p), 4),
            "significant":     bool(p < 0.05),
            "ci_95":           [round(ci_lo, 4), round(ci_hi, 4)],
            "variance_reduction_pct": round(float(var_red), 2),
        },
        "per_cluster": per_cluster,
        "wabaq_moderates_personalisation": {
            "pearson_r": round(float(r_wab), 3),
            "p_value":   round(float(p_wab), 4),
            "significant": bool(p_wab < 0.05),
        },
        "personalisation_beneficial": bool(np.mean(ciu_dapta) > np.mean(ciu_gddqn) and p < 0.05),
        "rq4_answered_positively": bool(np.mean(ciu_dapta) > np.mean(ciu_gddqn) and p < 0.05),
    }
    with open(output_dir / "rq4_personalisation.json", "w") as f:
        json.dump(rq4, f, indent=2)

    # Save cluster_performance.json for rq4_aphasia_only.py
    Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)
    with open("outputs/evaluation/cluster_performance.json", "w") as f:
        json.dump(per_cluster, f, indent=2)

    logger.info("RQ2-4 results saved to outputs/rl/")
    logger.info("  rq2_results.json")
    logger.info("  rq3_transfer.json")
    logger.info("  rq4_personalisation.json")
    logger.info("  cluster_performance.json -> outputs/evaluation/")

    print("\n" + "=" * 60)
    print("TRAINING SET RQ RESULTS")
    print("=" * 60)
    print(f"\nRQ2: DAPTA vs RBDE — {rq2['summary']['DAPTA_vs_RBDE']['n_clinically_meaningful']}/5 metrics clinically meaningful")
    print(f"RQ3: Transfer to naturalistic speech — answered: {rq3['rq3_answered_positively']}")
    print(f"RQ4: Personalisation beneficial — {rq4['personalisation_beneficial']}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    cfg        = Config.load()
    output_dir = Path("outputs/rl")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    if args.eval_only:
        run_eval_only(args)
        return
    logger.info("=" * 60)
    logger.info("DAPTA Phase 2b: RL Agent Training")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load PES outputs
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading PES outputs...")
    pes_dir = Path("outputs/pes")

    if not (pes_dir / "transition_model.pt").exists():
        logger.error("PES outputs not found. Run experiments/run_pes.py first.")
        return

    pes_data       = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    initial_states = pes_data["initial_states"]
    cluster_labels = pes_data["cluster_labels"]

    with open(pes_dir / "cluster_assignments.json") as f:
        cluster_info = json.load(f)

    # Load patient profiles for RQ4 factor analysis
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

    # ------------------------------------------------------------------
    # 2. Build environments
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Building environments per cluster...")
    cluster_envs = load_environments(
        transition_model=transition_model,
        initial_states=initial_states,
        cluster_labels=cluster_labels,
        episode_horizon=20,
    )
    all_envs   = [env for envs in cluster_envs.values() for env in envs]
    n_clusters = len(cluster_envs)
    logger.info(f"Total environments: {len(all_envs)}, clusters: {n_clusters}")

    # ------------------------------------------------------------------
    # 3. Train patient-specific DDQN per cluster
    # ------------------------------------------------------------------
    logger.info(f"\n[3/5] Training patient-specific DDQN agents ({args.total_steps} steps each)...")
    training_logs  = {}
    cluster_agents: Dict[int, DDQNAgent] = {}

    for cluster_id, envs in cluster_envs.items():
        if not envs:
            logger.warning(f"Skipping cluster {cluster_id} — no environments.")
            continue

        logger.info(f"\n  Training DDQN for cluster {cluster_id} ({len(envs)} patients)...")
        ckpt  = str(output_dir / f"ddqn_cluster_{cluster_id}.pt")
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
    # 4. Train G-DDQN (generalised)
    # ------------------------------------------------------------------
    logger.info(f"\n  Training G-DDQN on all {len(all_envs)} environments...")
    g_ddqn_ckpt = str(output_dir / "ddqn_generalised.pt")
    g_ddqn      = make_ddqn_agent(checkpoint_path=g_ddqn_ckpt)

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
    # 5. Train PPO
    # ------------------------------------------------------------------
    if all_envs and not args.skip_ppo:
        logger.info(f"\n  Training PPO...")
        ppo_save  = str(output_dir / "ppo_agent")
        ppo_model = train_ppo(
            env=all_envs[0],
            total_steps=args.total_steps,
            save_path=ppo_save,
        )
        training_logs["ppo"] = {
            "status": "trained" if ppo_model else "skipped_no_sb3",
            "steps": args.total_steps,
        }
    else:
        logger.info("  Skipping PPO.")

    # ------------------------------------------------------------------
    # 6. Collect per-patient improvement arrays for all agents
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Collecting per-patient improvement arrays...")

    # DAPTA: run each cluster's specialist agent on its own patients
    dapta_disc_list = []
    dapta_surp_list = []
    dapta_label_list = []

    for cluster_id, agent in cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0  # greedy
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        dapta_disc_list.append(disc)
        dapta_surp_list.append(surp)
        dapta_label_list.extend([cluster_id] * len(envs))

    dapta_disc   = np.concatenate(dapta_disc_list,  axis=0) if dapta_disc_list  else np.zeros((0, 5))
    dapta_surp   = np.concatenate(dapta_surp_list,  axis=0) if dapta_surp_list  else np.zeros(0)
    dapta_labels = np.array(dapta_label_list, dtype=int)

    # G-DDQN: run on all environments
    g_ddqn.eps = 0.0  # greedy — fixed attribute name
    g_ddqn_disc, g_ddqn_surp, _ = run_episodes(g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # RBDE baseline
    rbde = RuleBasedBaseline()
    rbde_disc_list, rbde_surp_list = [], []
    for env in all_envs:
        state, _ = env.reset()
        rbde.reset()
        initial_surp = float(state[SURPRISAL_DIM])
        total_reward = 0.0
        for _ in range(env.episode_horizon):
            action = rbde.select_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        disc_imp = env.get_cumulative_discourse_improvement()[:5]
        rbde_disc_list.append(disc_imp)
        rbde_surp_list.append(initial_surp - float(state[SURPRISAL_DIM]))

    rbde_disc = np.stack(rbde_disc_list)
    rbde_surp = np.array(rbde_surp_list, dtype=np.float32)

    # RTS baseline
    rts = RandomBaseline()
    rts_disc_list, rts_surp_list = [], []
    for env in all_envs:
        state, _ = env.reset()
        initial_surp = float(state[SURPRISAL_DIM])
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        disc_imp = env.get_cumulative_discourse_improvement()[:5]
        rts_disc_list.append(disc_imp)
        rts_surp_list.append(initial_surp - float(state[SURPRISAL_DIM]))

    rts_disc = np.stack(rts_disc_list)
    rts_surp = np.array(rts_surp_list, dtype=np.float32)

    # Save improvement arrays for run_evaluation.py and rq4_aphasia_only.py
    np.save(str(output_dir / "improvements_DAPTA.npy"),  dapta_disc)
    np.save(str(output_dir / "improvements_G_DDQN.npy"), g_ddqn_disc)
    np.save(str(output_dir / "improvements_RBDE.npy"),   rbde_disc)
    np.save(str(output_dir / "improvements_RTS.npy"),    rts_disc)

    # ------------------------------------------------------------------
    # 7. Answer RQ2, RQ3, RQ4
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Computing RQ2 / RQ3 / RQ4 statistics...")

    # RQ2
    rq2 = answer_rq2(dapta_disc, rbde_disc, rts_disc)
    with open(output_dir / "rq2_results.json", "w") as f:
        json.dump(rq2, f, indent=2)
    logger.info(
        f"  RQ2: DAPTA vs RBDE — "
        f"{rq2['summary']['DAPTA_vs_RBDE']['n_clinically_meaningful']} "
        f"metrics clinically meaningful. "
        f"Answered positively: {rq2['summary']['DAPTA_vs_RBDE']['rq2_answered_positively']}"
    )

    # RQ3
    rq3 = answer_rq3(dapta_disc, dapta_surp, rbde_disc, rbde_surp)
    with open(output_dir / "rq3_transfer.json", "w") as f:
        json.dump(rq3, f, indent=2)
    logger.info(
        f"  RQ3: Transfer to naturalistic speech — "
        f"answered positively: {rq3['rq3_answered_positively']}"
    )

    # RQ4 — align arrays: DAPTA and G-DDQN must cover same patients
    # G-DDQN covers all_envs; DAPTA covers only cluster agents' patients
    # For RQ4 we use only patients covered by both
    n_dapta   = len(dapta_disc)
    n_all     = len(all_envs)
    min_n     = min(n_dapta, n_all)

    rq4 = answer_rq4(
        dapta_disc=dapta_disc[:min_n],
        g_ddqn_disc=g_ddqn_disc[:min_n],
        dapta_surp=dapta_surp[:min_n],
        g_ddqn_surp=g_ddqn_surp[:min_n],
        cluster_labels=dapta_labels[:min_n],
        profiles_data=profiles_data[:min_n],
    )
    with open(output_dir / "rq4_personalisation.json", "w") as f:
        json.dump(rq4, f, indent=2)

    # Save cluster_performance.json for rq4_aphasia_only.py
    cluster_perf = {}
    for c_str, c_data in rq4["per_cluster"].items():
        cluster_perf[c_str] = {
            "n":                c_data["n"],
            "dominant_subtype": c_data["dominant_subtype"],
            "subtype_counts":   c_data["subtype_counts"],
            "dapta_ciu_mean":   c_data["dapta_ciu_mean"],
            "gddqn_ciu_mean":   c_data["gddqn_ciu_mean"],
            "cohens_d_ciu":     c_data["cohens_d_ciu"],
        }
    (output_dir.parent / "evaluation").mkdir(parents=True, exist_ok=True)
    with open(output_dir.parent / "evaluation" / "cluster_performance.json", "w") as f:
        json.dump(cluster_perf, f, indent=2)

    logger.info(
        f"  RQ4: Personalisation beneficial: {rq4['personalisation_beneficial']} | "
        f"CIU variance reduction: {rq4['ciu_variance_reduction_pct']:.1f}%"
    )

    # ------------------------------------------------------------------
    # Save training logs and results table
    # ------------------------------------------------------------------
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
        if isinstance(obj, list):  return [convert(x) for x in obj]
        if isinstance(obj, dict):  return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(output_dir / "training_logs.json", "w") as f:
        json.dump(convert(training_logs), f, indent=2)

    results_table = {
        "per_agent_means": {
            agent: {
                metric: round(float(np.mean(arr[:, i])), 4)
                for i, metric in enumerate(METRIC_NAMES)
            }
            for agent, arr in [
                ("DAPTA",  dapta_disc),
                ("G_DDQN", g_ddqn_disc),
                ("RBDE",   rbde_disc),
                ("RTS",    rts_disc),
            ]
        },
        "per_cluster": rq4["per_cluster"],
        "config": {
            "total_steps":     args.total_steps,
            "episode_horizon": 20,
            "n_clusters":      n_clusters,
            "n_environments":  len(all_envs),
        },
        "rq2_summary": rq2["summary"],
        "rq3_summary": {
            "answered_positively": rq3["rq3_answered_positively"],
            "surprisal_cohens_d":  rq3["surprisal_improvement"]["cohens_d"],
        },
        "rq4_summary": {
            "answered_positively":       rq4["personalisation_beneficial"],
            "ciu_variance_reduction_pct": rq4["ciu_variance_reduction_pct"],
        },
    }

    with open(output_dir / "results_table.json", "w") as f:
        json.dump(results_table, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DAPTA PHASE 2b — RQ SUMMARY")
    print("=" * 70)

    print("\nRQ2: Does RL outperform rule-based sequencing?")
    for comp in ["DAPTA_vs_RBDE", "DAPTA_vs_RTS"]:
        s = rq2["summary"][comp]
        print(
            f"  {comp}: {s['n_clinically_meaningful']}/5 metrics clinically "
            f"meaningful — {'ANSWERED ✓' if s['rq2_answered_positively'] else 'NOT ANSWERED'}"
        )

    print("\nRQ3: Do discourse gains transfer to naturalistic speech?")
    si = rq3["surprisal_improvement"]
    print(f"  DAPTA surp improvement: {si['dapta_mean_surp_improvement']:.4f}")
    print(f"  RBDE  surp improvement: {si['rbde_mean_surp_improvement']:.4f}")
    print(f"  Cohen's d: {si['cohens_d']:.3f}  p={si['wilcoxon_p']:.4f}")
    if "discourse_to_surprisal_correlation" in rq3:
        corr = rq3["discourse_to_surprisal_correlation"]
        print(f"  CIU-Surprisal correlation: r={corr.get('pearson_r', 'N/A')} p={corr.get('p_value', 'N/A')}")
    print(f"  {'ANSWERED ✓' if rq3['rq3_answered_positively'] else 'NOT ANSWERED'}")

    print("\nRQ4: Does patient-specific RL outperform generalised RL?")
    print(f"  Personalisation beneficial: {rq4['personalisation_beneficial']}")
    print(f"  CIU variance reduction: {rq4['ciu_variance_reduction_pct']:.1f}%")
    if "wabaq_moderates_personalisation" in rq4:
        wab = rq4["wabaq_moderates_personalisation"]
        print(f"  WAB-AQ moderates benefit: r={wab['pearson_r']} p={wab['p_value']}")
        print(f"  {wab['interpretation']}")
    print(f"  {'ANSWERED ✓' if rq4['rq4_answered_positively'] else 'NOT ANSWERED'}")

    print("\n" + "=" * 70)
    print(f"Outputs saved to: {output_dir}/")
    print("=" * 70)
    logger.info("Next step: python experiments/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 2b: RL Agent Training")
    parser.add_argument(
        "--total_steps", type=int, default=50_000,
        help="Training steps per agent (default: 50000). Use 5000 for a quick test."
    )
    parser.add_argument(
        "--skip_ppo", action="store_true",
        help="Skip PPO training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training, load saved agents and compute RQ2-4 results only"
    )
    args = parser.parse_args()
    main(args)