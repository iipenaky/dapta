"""
run_evaluation.py
-----------------
DAPTA Phase 3: Held-out test-set evaluation.

Loads all trained agent checkpoints and evaluates them on the
held-out TEST patients only — patients never seen during training
or clustering.

For personalised agents (DAPTA, No-GRU DAPTA, PPO-Pers):
    Each test patient is assigned to the nearest cluster centroid
    using PatientClusterer.predict(), then evaluated by that
    cluster's trained agent.

For generalised agents (G-DDQN, No-GRU G-DDQN, PPO-Gen):
    A single pooled agent evaluates all test patients.

Baselines (RBDE, RTS) run on all test patients.

Outputs (all in outputs/evaluation/):
    test_rq2_results.json          — main RQ2 results on test set
    test_rq4_personalisation.json  — RQ4 personalisation results
    test_rq3_transfer.json         — cross-task transfer (RQ3)
    test_ablation_results.json     — ablation results on test set
    test_all_agents_vs_rbde.json   — per-agent RQ2 detail
    dapta_test_results.csv         — per-patient gains for RQ3
    cluster_performance.json       — per-cluster breakdown
    test_per_agent_arrays/         — raw .npy improvement arrays

Run after:
    python experiments/run_pes.py
    python experiments/run_rl.py
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from dapta.pes.transition_model import TransitionModel
from dapta.pes.cluster import PatientClusterer
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import RuleBasedBaseline, RandomBaseline
from dapta.dae.state_builder import PatientProfile, STATE_DIM
from dapta.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/run_evaluation.log")

METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
SURPRISAL_DIM      = 38
CLINICAL_THRESHOLD = 0.40

# TASKS order in state vector:
# cookie_theft=0, cinderella=1, sandwich=2, stroke_narrative=3, conversation=4
# Each task block is 5 metrics wide, CIU is index 0 within each block
_STRUCTURED_TASK_INDICES  = [0, 1, 2]
_CONVERSATION_TASK_INDEX  = 4
_N_METRICS_PER_TASK       = 5


# ──────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# PPO wrapper
# ──────────────────────────────────────────────────────────────────────

class PPOWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def select_action(self, state: np.ndarray) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# FIX: returns full discourse improvement array (25 dims), not just [:5]
# This is required for RQ3 task-level gain extraction.
# ──────────────────────────────────────────────────────────────────────

def run_episode(
    agent,
    env:     TherapyEnv,
    is_ddqn: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Run a single greedy episode.
    Returns (disc_improvement — full 25-dim discourse block, surp_imp).
    """
    state, _       = env.reset()
    state_history  = []
    action_history = []
    initial_surp   = float(state[SURPRISAL_DIM])

    for _ in range(env.episode_horizon):
        if is_ddqn:
            history_tensor = agent.build_history_tensor(
                state_history, action_history
            )
            action = agent.select_action(state, history_tensor, greedy=True)
        else:
            action = agent.select_action(state)

        state_history.append(state.copy())
        action_history.append(action)

        next_state, _, done, _, _ = env.step(action)
        state = next_state
        if done:
            break

    # FIX: full 25-dim discourse block instead of [:5]
    disc_imp   = env.get_cumulative_discourse_improvement()
    final_surp = float(state[SURPRISAL_DIM])
    return disc_imp, initial_surp - final_surp


# ──────────────────────────────────────────────────────────────────────
# Agent loaders
# ──────────────────────────────────────────────────────────────────────

def load_ddqn(checkpoint_path: str, use_gru: bool) -> Optional[DDQNAgent]:
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    agent = DDQNAgent(
        state_dim       = STATE_DIM,
        gru_hidden      = 128,
        gru_layers      = 2,
        use_gru         = use_gru,
        checkpoint_path = checkpoint_path,
    )
    agent.load()
    agent.eps = 0.0   # fully greedy at evaluation time
    return agent


def load_ppo(save_path: str) -> Optional[PPOWrapper]:
    try:
        from stable_baselines3 import PPO
        path = Path(save_path + ".zip")
        if not path.exists():
            path = Path(save_path)
        if not path.exists():
            logger.warning(f"PPO checkpoint not found: {save_path}")
            return None
        model = PPO.load(str(save_path))
        return PPOWrapper(model)
    except ImportError:
        logger.warning("stable-baselines3 not installed — skipping PPO.")
        return None
    except Exception as e:
        logger.warning(f"PPO load failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────
# RQ helpers
# ──────────────────────────────────────────────────────────────────────

def compute_rq2(
    agent_disc:  np.ndarray,
    rbde_disc:   np.ndarray,
    rts_disc:    np.ndarray,
    agent_name:  str = "BEST_RL",
) -> dict:
    # agent_disc may be 25-dim; we only score the first 5 (cookie_theft block)
    # for the summary metrics — consistent with how METRIC_NAMES is defined.
    a = agent_disc[:, :5]
    r = rbde_disc[:, :5]
    t = rts_disc[:,  :5]

    out = {f"{agent_name}_vs_RBDE": {}, f"{agent_name}_vs_RTS": {}, "summary": {}}

    for comparison, baseline in [
        (f"{agent_name}_vs_RBDE", r),
        (f"{agent_name}_vs_RTS",  t),
    ]:
        p_values = []
        entries  = {}
        for i, metric in enumerate(METRIC_NAMES):
            d            = cohens_d_paired(baseline[:, i], a[:, i])
            stat, p      = wilcoxon_test(a[:, i], baseline[:, i])
            ci_lo, ci_hi = bootstrap_ci(a[:, i] - baseline[:, i])
            p_values.append(p)
            entries[metric] = {
                "agent_mean":            round(float(np.mean(a[:, i])),        4),
                "baseline_mean":         round(float(np.mean(baseline[:, i])), 4),
                "cohens_d":              round(d, 3),
                "effect_size":           effect_size_label(d),
                "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
                "wilcoxon_stat":         round(stat, 4),
                "p_raw":                 round(float(p), 4),
                "ci_95":                 [round(ci_lo, 4), round(ci_hi, 4)],
            }
        p_corr, sig = bonferroni(p_values)
        for i, metric in enumerate(METRIC_NAMES):
            entries[metric]["p_corrected"] = round(p_corr[i], 4)
            entries[metric]["significant"] = sig[i]
        out[comparison] = entries
        n_clin = sum(1 for v in entries.values() if v["clinically_meaningful"])
        n_sig  = sum(1 for v in entries.values() if v["significant"])
        out["summary"][comparison] = {
            "n_significant":           n_sig,
            "n_clinically_meaningful": n_clin,
            "rq2_answered_positively": n_clin >= 2,
        }
    return out


def compute_rq4(
    pers_disc:      np.ndarray,
    gen_disc:       np.ndarray,
    cluster_labels: np.ndarray,
    profiles_data:  list,
    pers_name:      str = "BEST_PERS",
    gen_name:       str = "BEST_GEN",
) -> dict:
    # Use first 5 dims (cookie_theft block) for summary metrics
    pd_ = pers_disc[:, :5]
    gd_ = gen_disc[:,  :5]

    out      = {"best_personalised": pers_name, "best_generalised": gen_name}
    p_values = []
    entries  = {}

    for i, metric in enumerate(METRIC_NAMES):
        d            = cohens_d_paired(gd_[:, i], pd_[:, i])
        stat, p      = wilcoxon_test(pd_[:, i], gd_[:, i])
        ci_lo, ci_hi = bootstrap_ci(pd_[:, i] - gd_[:, i])
        p_values.append(p)
        entries[metric] = {
            "personalised_mean":     round(float(np.mean(pd_[:, i])), 4),
            "generalised_mean":      round(float(np.mean(gd_[:, i])), 4),
            "cohens_d":              round(d, 3),
            "effect_size":           effect_size_label(d),
            "clinically_meaningful": abs(d) >= CLINICAL_THRESHOLD,
            "wilcoxon_stat":         round(stat, 4),
            "p_raw":                 round(float(p), 4),
            "ci_95":                 [round(ci_lo, 4), round(ci_hi, 4)],
        }

    p_corr, sig = bonferroni(p_values)
    for i, metric in enumerate(METRIC_NAMES):
        entries[metric]["p_corrected"] = round(p_corr[i], 4)
        entries[metric]["significant"] = sig[i]

    ciu_pers = pd_[:, 0]
    ciu_gen  = gd_[:, 0]
    var_reduction = (
        (np.var(ciu_gen, ddof=1) - np.var(ciu_pers, ddof=1))
        / max(np.var(ciu_gen, ddof=1), 1e-8) * 100
    )

    out["pooled_pers_vs_gen"]         = entries
    out["ciu_variance_reduction_pct"] = round(float(var_reduction), 2)
    out["personalisation_beneficial"] = bool(
        np.mean(ciu_pers) > np.mean(ciu_gen) and any(sig)
    )

    # Per-cluster breakdown
    per_cluster = {}
    for c in np.unique(cluster_labels):
        mask  = cluster_labels == c
        n     = int(mask.sum())
        if n == 0:
            continue
        d_ciu = ciu_pers[mask]
        g_ciu = ciu_gen[mask]
        subtypes = [
            p.get("aphasia_subtype", "Other")
            for p, m in zip(profiles_data, mask) if m
        ]
        dominant = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
        per_cluster[str(c)] = {
            "n":                     int(n),
            "dominant_subtype":      dominant,
            "subtype_counts":        {s: subtypes.count(s) for s in set(subtypes)},
            "personalised_ciu_mean": round(float(np.mean(d_ciu)), 4),
            "generalised_ciu_mean":  round(float(np.mean(g_ciu)), 4),
            "cohens_d_ciu":          round(cohens_d_paired(g_ciu, d_ciu), 3),
            "personalised_wins":     bool(np.mean(d_ciu) > np.mean(g_ciu)),
        }

    # WAB-AQ moderation
    wab_aqs = np.array([
        float(p.get("wab_aq") or 55.0) for p in profiles_data
    ], dtype=np.float32)
    if len(wab_aqs) >= 5:
        r_wab, p_wab = scipy_stats.pearsonr(wab_aqs, ciu_pers - ciu_gen)
        out["wabaq_moderates_personalisation"] = {
            "pearson_r":     round(float(r_wab), 3),
            "p_value":       round(float(p_wab), 4),
            "significant":   bool(p_wab < 0.05),
            "interpretation": (
                "higher WAB-AQ patients benefit more from personalisation"
                if r_wab > 0.2 and p_wab < 0.05
                else "lower WAB-AQ patients benefit more from personalisation"
                if r_wab < -0.2 and p_wab < 0.05
                else "WAB-AQ does not significantly moderate personalisation benefit"
            ),
        }

    out["per_cluster"]             = per_cluster
    out["rq4_answered_positively"] = out["personalisation_beneficial"]
    return out


def compute_ablation(
    all_agent_results: Dict[str, np.ndarray],
    rbde_disc:         np.ndarray,
) -> dict:
    out       = {}
    per_agent = {}

    for name, disc in all_agent_results.items():
        if len(disc) == 0:
            continue
        d5 = disc[:, :5]
        r5 = rbde_disc[:, :5]
        per_agent[name] = {
            metric: {
                "mean":             round(float(np.mean(d5[:, i])), 4),
                "cohens_d_vs_rbde": round(cohens_d_paired(r5[:, i], d5[:, i]), 3),
                "effect_size":      effect_size_label(
                    cohens_d_paired(r5[:, i], d5[:, i])
                ),
            }
            for i, metric in enumerate(METRIC_NAMES)
        }
        per_agent[name]["ciu_cohens_d_vs_rbde"] = round(
            cohens_d_paired(r5[:, 0], d5[:, 0]), 3
        )
    out["per_agent_summary"] = per_agent

    def _d_and_p(name_a, name_b):
        a = all_agent_results[name_a][:, 0]
        b = all_agent_results[name_b][:, 0]
        d = cohens_d_paired(b, a)
        _, p = wilcoxon_test(a, b)
        return d, p

    if "DAPTA" in all_agent_results and "NO_GRU_DAPTA" in all_agent_results:
        d, p = _d_and_p("DAPTA", "NO_GRU_DAPTA")
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

    if "G_DDQN" in all_agent_results and "NO_GRU_G_DDQN" in all_agent_results:
        d, p = _d_and_p("G_DDQN", "NO_GRU_G_DDQN")
        out["gru_benefit_generalised"] = {
            "cohens_d":    round(d, 3),
            "wilcoxon_p":  round(float(p), 4),
            "significant": bool(p < 0.05),
            "gru_helps":   bool(d > 0.1 and p < 0.05),
        }

    if "PPO_GENERALISED" in all_agent_results and "G_DDQN" in all_agent_results:
        d, p = _d_and_p("PPO_GENERALISED", "G_DDQN")
        out["algorithm_effect_generalised"] = {
            "ppo_vs_gddqn_cohens_d": round(d, 3),
            "wilcoxon_p":            round(float(p), 4),
            "significant":           bool(p < 0.05),
            "ppo_wins": bool(
                np.mean(all_agent_results["PPO_GENERALISED"][:, 0]) >
                np.mean(all_agent_results["G_DDQN"][:, 0])
            ),
            "interpretation": (
                "PPO outperforms DDQN — algorithm choice matters"
                if d > 0.1 and p < 0.05
                else "No significant difference between PPO and DDQN"
            ),
        }

    if "PPO_PERSONALISED" in all_agent_results and "DAPTA" in all_agent_results:
        d, _ = _d_and_p("PPO_PERSONALISED", "DAPTA")
        out["algorithm_effect_personalised"] = {
            "ppo_vs_dapta_cohens_d": round(d, 3),
            "ppo_wins": bool(
                np.mean(all_agent_results["PPO_PERSONALISED"][:, 0]) >
                np.mean(all_agent_results["DAPTA"][:, 0])
            ),
        }

    ciu_means = {
        name: float(np.mean(disc[:, 0]))
        for name, disc in all_agent_results.items()
        if len(disc) > 0
    }
    if ciu_means:
        best = max(ciu_means, key=ciu_means.get)
        out["recommended_configuration"] = {
            "agent":         best,
            "ciu_rate_mean": round(ciu_means[best], 4),
            "all_ciu_means": {k: round(v, 4) for k, v in ciu_means.items()},
        }

    return out


def compute_rq3_transfer(output_dir: Path) -> dict:
    """
    RQ3 — cross-task transfer using RL episode gains on test patients.

    Correlates structured task CIU gain (mean of cookie_theft, cinderella,
    sandwich) against conversation CIU gain across test patients.
    Both gains come from the same DAPTA episode, so this is a genuine
    transfer question rather than a baseline correlation.
    """
    csv_path = output_dir / "dapta_test_results.csv"
    if not csv_path.exists():
        logger.warning("dapta_test_results.csv not found — skipping RQ3.")
        return {
            "rq3_answered_positively": False,
            "note": "dapta_test_results.csv not found — run after DAPTA evaluation",
        }

    try:
        df = pd.read_csv(str(csv_path))
        df = df.dropna(subset=["structured_ciu_gain", "conversation_ciu_gain"])
        n  = len(df)

        if n < 5:
            return {
                "rq3_answered_positively": False,
                "note": f"insufficient test patients (n={n})",
            }

        r, p = scipy_stats.spearmanr(
            df["structured_ciu_gain"],
            df["conversation_ciu_gain"],
        )
        sig = bool(p < 0.05)

        ci_lo, ci_hi = bootstrap_ci(
            df["structured_ciu_gain"].values - df["conversation_ciu_gain"].values
        )

        return {
            "method":                  "rl_episode_gain_transfer_spearman",
            "n_test_patients":         int(n),
            "spearman_r":              round(float(r), 3),
            "p_value":                 round(float(p), 4),
            "significant":             sig,
            "ci_95_gain_diff":         [round(ci_lo, 4), round(ci_hi, 4)],
            "rq3_answered_positively": bool(sig and r > 0.2),
            "interpretation": (
                "structured task RL gains transfer to naturalistic conversation"
                if sig and r > 0.2
                else "no significant cross-task transfer detected"
            ),
            "note": (
                "Transfer measured as Spearman correlation between mean "
                "structured-task CIU gain (cookie_theft, cinderella, sandwich) "
                "and conversation CIU gain across test patients within the "
                "simulated RL environment."
            ),
        }
    except Exception as e:
        logger.warning(f"RQ3 analysis failed: {e}")
        return {"rq3_answered_positively": False, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(args) -> None:
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "test_per_agent_arrays"
    arrays_dir.mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 3: Held-out Test Evaluation")
    logger.info("=" * 60)

    # ── 1. Load PES outputs ──────────────────────────────────────────
    logger.info("\n[1/6] Loading PES outputs...")
    pes_dir = Path("outputs/pes")
    rl_dir  = Path("outputs/rl")

    pes_data        = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    all_states      = pes_data["initial_states"]
    all_session_ids = pes_data["session_ids"]
    cluster_labels  = pes_data["cluster_labels"]
    split_labels    = pes_data["split_labels"]

    with open("outputs/dae/patient_profiles.json") as f:
        all_profiles_data = json.load(f)

    transition_model = TransitionModel(
        hidden_sizes    = (128, 64),
        dropout         = 0.2,
        checkpoint_path = str(pes_dir / "transition_model.pt"),
        device          = args.device,
    )
    transition_model.load()
    logger.info("Transition model loaded.")

    # ── 2. Filter to TEST patients only ─────────────────────────────
    logger.info("\n[2/6] Filtering to test-split patients...")

    test_mask        = split_labels == "test"
    test_states      = all_states[test_mask]
    test_session_ids = all_session_ids[test_mask]

    session_to_profile = {p["session_id"]: p for p in all_profiles_data}
    test_profiles_data = [
        session_to_profile.get(str(sid), {"aphasia_subtype": "Other", "wab_aq": None})
        for sid in test_session_ids
    ]

    n_test = int(test_mask.sum())
    logger.info(f"  Test patients: {n_test}")

    if n_test == 0:
        logger.error("No test patients found. Check split_labels in env_initial_states.npz.")
        return

    test_envs = build_env_population(
        initial_states   = list(test_states),
        transition_model = transition_model,
        episode_horizon  = 20,
    )
    logger.info(f"  Built {len(test_envs)} test environments.")

    # ── 3. Load all trained agents ───────────────────────────────────
    logger.info("\n[3/6] Loading trained agent checkpoints...")

    n_clusters = int(cluster_labels[cluster_labels >= 0].max()) + 1

    dapta_cluster_agents: Dict[int, DDQNAgent] = {}
    for c in range(n_clusters):
        agent = load_ddqn(str(rl_dir / f"ddqn_cluster_{c}.pt"), use_gru=True)
        if agent:
            dapta_cluster_agents[c] = agent
            logger.info(f"  DAPTA cluster {c} loaded.")

    g_ddqn = load_ddqn(str(rl_dir / "ddqn_generalised.pt"), use_gru=True)
    if g_ddqn:
        logger.info("  G-DDQN loaded.")

    no_gru_cluster_agents: Dict[int, DDQNAgent] = {}
    for c in range(n_clusters):
        agent = load_ddqn(str(rl_dir / f"no_gru_ddqn_cluster_{c}.pt"), use_gru=False)
        if agent:
            no_gru_cluster_agents[c] = agent
            logger.info(f"  No-GRU DAPTA cluster {c} loaded.")

    no_gru_g_ddqn = load_ddqn(str(rl_dir / "no_gru_ddqn_generalised.pt"), use_gru=False)
    if no_gru_g_ddqn:
        logger.info("  No-GRU G-DDQN loaded.")

    ppo_gen = load_ppo(str(rl_dir / "ppo_generalised"))

    ppo_pers_agents: Dict[int, PPOWrapper] = {}
    for c in range(n_clusters):
        ppo = load_ppo(str(rl_dir / f"ppo_cluster_{c}"))
        if ppo:
            ppo_pers_agents[c] = ppo

    # ── 4. Assign test patients to clusters ─────────────────────────
    logger.info("\n[4/6] Assigning test patients to clusters...")

    test_profiles_obj = [
        PatientProfile(
            participant_id  = p.get("participant_id", str(sid)),
            aphasia_subtype = p.get("aphasia_subtype", "Other"),
            wab_aq          = p.get("wab_aq", 50.0),
        )
        for p, sid in zip(test_profiles_data, test_session_ids)
    ]

    with open(pes_dir / "clusterer.pkl", "rb") as f:
        clusterer = pickle.load(f)

    test_raw_subtypes  = [p.get("aphasia_subtype", "Other") for p in test_profiles_data]
    predicted_clusters = clusterer.predict(
        test_profiles_obj,
        state_vectors = test_states,
        raw_subtypes  = test_raw_subtypes,
    )
    predicted_clusters = np.where(predicted_clusters < 0, 0, predicted_clusters)
    logger.info(
        f"  Cluster assignments for {n_test} test patients: "
        f"{np.bincount(predicted_clusters)}"
    )

    # ── 5. Run all agents on test environments ───────────────────────
    logger.info("\n[5/6] Evaluating all agents on test patients...")

    def run_personalised(
        cluster_agents: Dict[int, object],
        is_ddqn:        bool,
    ) -> np.ndarray:
        disc_list = []
        for env, cluster_id in zip(test_envs, predicted_clusters):
            agent = cluster_agents.get(int(cluster_id)) or cluster_agents.get(0)
            if agent is None:
                disc_list.append(np.zeros(25, dtype=np.float32))
                continue
            disc, _ = run_episode(agent, env, is_ddqn=is_ddqn)
            disc_list.append(disc)
        return np.stack(disc_list)

    def run_generalised(agent, is_ddqn: bool) -> np.ndarray:
        disc_list = []
        for env in test_envs:
            disc, _ = run_episode(agent, env, is_ddqn=is_ddqn)
            disc_list.append(disc)
        return np.stack(disc_list)

    all_agent_results: Dict[str, np.ndarray] = {}

    if dapta_cluster_agents:
        logger.info("  Evaluating DAPTA (GRU personalised)...")
        all_agent_results["DAPTA"] = run_personalised(dapta_cluster_agents, is_ddqn=True)

    if g_ddqn:
        logger.info("  Evaluating G-DDQN (GRU generalised)...")
        all_agent_results["G_DDQN"] = run_generalised(g_ddqn, is_ddqn=True)

    if no_gru_cluster_agents:
        logger.info("  Evaluating No-GRU DAPTA (personalised)...")
        all_agent_results["NO_GRU_DAPTA"] = run_personalised(no_gru_cluster_agents, is_ddqn=True)

    if no_gru_g_ddqn:
        logger.info("  Evaluating No-GRU G-DDQN (generalised)...")
        all_agent_results["NO_GRU_G_DDQN"] = run_generalised(no_gru_g_ddqn, is_ddqn=True)

    if ppo_gen:
        logger.info("  Evaluating PPO generalised...")
        all_agent_results["PPO_GENERALISED"] = run_generalised(ppo_gen, is_ddqn=False)

    if ppo_pers_agents:
        logger.info("  Evaluating PPO personalised...")
        all_agent_results["PPO_PERSONALISED"] = run_personalised(ppo_pers_agents, is_ddqn=False)

    # RBDE baseline
    logger.info("  Evaluating RBDE baseline...")
    rbde = RuleBasedBaseline()
    rbde_disc_list = []
    for env in test_envs:
        state, _ = env.reset()
        rbde.reset()
        for _ in range(env.episode_horizon):
            action = rbde.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rbde_disc_list.append(env.get_cumulative_discourse_improvement())
    rbde_disc = np.stack(rbde_disc_list)

    # RTS baseline
    logger.info("  Evaluating RTS baseline...")
    rts = RandomBaseline()
    rts_disc_list = []
    for env in test_envs:
        state, _ = env.reset()
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rts_disc_list.append(env.get_cumulative_discourse_improvement())
    rts_disc = np.stack(rts_disc_list)

    # Save raw arrays
    for name, arr in all_agent_results.items():
        np.save(str(arrays_dir / f"test_{name}.npy"), arr)
    np.save(str(arrays_dir / "test_RBDE.npy"), rbde_disc)
    np.save(str(arrays_dir / "test_RTS.npy"),  rts_disc)
    logger.info(f"  Raw arrays saved to {arrays_dir}/")

    # ── 6. Compute all RQ statistics ────────────────────────────────
    logger.info("\n[6/6] Computing test-set RQ statistics...")

    PERSONALISED = {"DAPTA", "NO_GRU_DAPTA", "PPO_PERSONALISED"}
    GENERALISED  = {"G_DDQN", "NO_GRU_G_DDQN", "PPO_GENERALISED"}

    available = {k: v for k, v in all_agent_results.items() if len(v) > 0}

    # RQ2
    rq2_all_agents = {}
    for agent_name, agent_disc in available.items():
        rq2_all_agents[agent_name] = compute_rq2(
            agent_disc, rbde_disc, rts_disc, agent_name=agent_name
        )

    rq2_summary_table = {}
    for agent_name, agent_disc in available.items():
        a5 = agent_disc[:, :5]
        r5 = rbde_disc[:, :5]
        t5 = rts_disc[:,  :5]
        d_ciu_rbde    = cohens_d_paired(r5[:, 0], a5[:, 0])
        _, p_ciu_rbde = wilcoxon_test(a5[:, 0], r5[:, 0])
        d_ciu_rts     = cohens_d_paired(t5[:, 0], a5[:, 0])
        n_clin_rbde   = sum(
            1 for i in range(5)
            if abs(cohens_d_paired(r5[:, i], a5[:, i])) >= CLINICAL_THRESHOLD
        )
        rq2_summary_table[agent_name] = {
            "type":                    "personalised" if agent_name in PERSONALISED else "generalised",
            "ciu_mean":                round(float(np.mean(a5[:, 0])), 4),
            "ciu_cohens_d_vs_rbde":    round(d_ciu_rbde, 3),
            "ciu_effect_size_vs_rbde": effect_size_label(d_ciu_rbde),
            "ciu_p_vs_rbde":           round(float(p_ciu_rbde), 4),
            "ciu_cohens_d_vs_rts":     round(d_ciu_rts, 3),
            "n_metrics_clin_vs_rbde":  n_clin_rbde,
            "rq2_answered_positively": n_clin_rbde >= 2,
            "beats_rbde":              bool(np.mean(a5[:, 0]) > np.mean(r5[:, 0])),
            "beats_rts":               bool(np.mean(a5[:, 0]) > np.mean(t5[:, 0])),
        }

    rq2_summary_table["RBDE"] = {
        "type":     "baseline",
        "ciu_mean": round(float(np.mean(rbde_disc[:, 0])), 4),
    }
    rq2_summary_table["RTS"] = {
        "type":     "baseline",
        "ciu_mean": round(float(np.mean(rts_disc[:, 0])), 4),
    }

    rq2_out = {
        "per_agent":               rq2_all_agents,
        "summary_table":           rq2_summary_table,
        "n_agents_beating_rbde":   sum(
            1 for v in rq2_summary_table.values() if v.get("beats_rbde", False)
        ),
        "n_agents_rq2_positive":   sum(
            1 for v in rq2_summary_table.values() if v.get("rq2_answered_positively", False)
        ),
        "rq2_answered_positively": any(
            v.get("rq2_answered_positively", False)
            for v in rq2_summary_table.values()
        ),
    }

    with open(output_dir / "test_rq2_results.json", "w") as f:
        json.dump(rq2_out, f, indent=2)

    with open(output_dir / "test_all_agents_vs_rbde.json", "w") as f:
        json.dump(rq2_out["per_agent"], f, indent=2)

    # Best agent overall
    best_name = max(available, key=lambda k: float(np.mean(available[k][:, 0])))
    best_disc = available[best_name]

    # RQ4
    pers_available = {k: v for k, v in available.items() if k in PERSONALISED}
    gen_available  = {k: v for k, v in available.items() if k in GENERALISED}

    rq4_out = {"note": "no personalised or generalised agents available"}
    if pers_available and gen_available:
        best_pers_name = max(pers_available, key=lambda k: float(np.mean(pers_available[k][:, 0])))
        best_gen_name  = max(gen_available,  key=lambda k: float(np.mean(gen_available[k][:, 0])))
        best_pers_disc = pers_available[best_pers_name]
        best_gen_disc  = gen_available[best_gen_name]

        min_n   = min(len(best_pers_disc), len(best_gen_disc))
        rq4_out = compute_rq4(
            pers_disc      = best_pers_disc[:min_n],
            gen_disc       = best_gen_disc[:min_n],
            cluster_labels = predicted_clusters[:min_n],
            profiles_data  = test_profiles_data[:min_n],
            pers_name      = best_pers_name,
            gen_name       = best_gen_name,
        )

    with open(output_dir / "test_rq4_personalisation.json", "w") as f:
        json.dump(rq4_out, f, indent=2)

    # Ablation
    ablation = compute_ablation(all_agent_results, rbde_disc)
    with open(output_dir / "test_ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # Per-patient CSV — saves task-level gains for RQ3
    if "DAPTA" in all_agent_results:
        dapta_disc = all_agent_results["DAPTA"]

        # Mean CIU gain across structured tasks (cookie_theft, cinderella, sandwich)
        structured_ciu_gain = np.mean(
            [dapta_disc[:, t * _N_METRICS_PER_TASK + 0]
             for t in _STRUCTURED_TASK_INDICES],
            axis=0,
        )
        # CIU gain on conversation task
        conversation_ciu_gain = dapta_disc[
            :, _CONVERSATION_TASK_INDEX * _N_METRICS_PER_TASK + 0
        ]

        rl_df = pd.DataFrame({
            "session_id":             [str(sid) for sid in test_session_ids],
            "participant_id":         [
                p.get("participant_id", str(sid))
                for p, sid in zip(test_profiles_data, test_session_ids)
            ],
            "cluster_id":             predicted_clusters.tolist(),
            "ciu_improvement":        dapta_disc[:, 0].tolist(),
            "mc_improvement":         dapta_disc[:, 1].tolist(),
            "mlu_improvement":        dapta_disc[:, 2].tolist(),
            "structured_ciu_gain":    structured_ciu_gain.tolist(),
            "conversation_ciu_gain":  conversation_ciu_gain.tolist(),
        })
        rl_df.to_csv(str(output_dir / "dapta_test_results.csv"), index=False)
        logger.info("  Saved dapta_test_results.csv")

    # RQ3 — uses episode gains from dapta_test_results.csv
    rq3 = compute_rq3_transfer(output_dir)
    with open(output_dir / "test_rq3_transfer.json", "w") as f:
        json.dump(rq3, f, indent=2)

    # Cluster performance summary
    if "DAPTA" in all_agent_results and "G_DDQN" in all_agent_results:
        cluster_perf = {}
        dapta_disc  = all_agent_results["DAPTA"]
        g_ddqn_disc = all_agent_results["G_DDQN"]
        for c in np.unique(predicted_clusters):
            mask  = predicted_clusters == c
            d_ciu = dapta_disc[mask, 0]
            g_ciu = g_ddqn_disc[mask, 0]
            cluster_perf[str(c)] = {
                "n":                 int(mask.sum()),
                "dapta_ciu_mean":    round(float(np.mean(d_ciu)), 4),
                "gddqn_ciu_mean":    round(float(np.mean(g_ciu)), 4),
                "cohens_d_ciu":      round(cohens_d_paired(g_ciu, d_ciu), 3),
                "personalised_wins": bool(np.mean(d_ciu) > np.mean(g_ciu)),
            }
        with open(output_dir / "cluster_performance.json", "w") as f:
            json.dump(cluster_perf, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DAPTA — HELD-OUT TEST SET RESULTS  (n={n_test} patients)")
    print("=" * 70)

    print(f"\nAll agents — CIU rate on TEST patients:")
    for name, arr in sorted(
        all_agent_results.items(),
        key=lambda x: -float(np.mean(x[1][:, 0]))
    ):
        if len(arr) > 0:
            d     = cohens_d_paired(rbde_disc[:, 0], arr[:, 0])
            label = "[pers]" if name in PERSONALISED else "[gen] "
            print(
                f"  {label} {name:22} CIU={np.mean(arr[:,0]):.4f}  "
                f"d_vs_RBDE={d:.3f}  ({effect_size_label(d)})"
            )
    print(f"         {'RBDE':22} CIU={np.mean(rbde_disc[:,0]):.4f}  (clinical baseline)")
    print(f"         {'RTS':22} CIU={np.mean(rts_disc[:,0]):.4f}  (random baseline)")

    print(f"\nBest overall agent: {best_name}  (CIU={np.mean(best_disc[:,0]):.4f})")

    if pers_available and gen_available:
        print(f"\nRQ4 — Personalisation:")
        print(f"  Best personalised: {best_pers_name}  (CIU={np.mean(best_pers_disc[:,0]):.4f})")
        print(f"  Best generalised:  {best_gen_name}   (CIU={np.mean(best_gen_disc[:,0]):.4f})")
        print(f"  Personalisation beneficial: {rq4_out.get('personalisation_beneficial', 'N/A')}")

    print(f"\nRQ2 — all agents vs RBDE:")
    for name, row in rq2_out["summary_table"].items():
        if row.get("type") == "baseline":
            print(f"  {name:26} CIU={row['ciu_mean']:.4f}  (baseline)")
        else:
            tag = "[pers]" if row["type"] == "personalised" else "[gen] "
            ans = "ANSWERED ✓" if row["rq2_answered_positively"] else "not answered"
            print(
                f"  {tag} {name:22} CIU={row['ciu_mean']:.4f}  "
                f"d_vs_RBDE={row['ciu_cohens_d_vs_rbde']:.3f}  "
                f"{row['n_metrics_clin_vs_rbde']}/5 clin  {ans}"
            )

    print(
        f"\nRQ3 — cross-task transfer: "
        f"r={rq3.get('spearman_r', 'N/A')}, "
        f"p={rq3.get('p_value', 'N/A')}, "
        f"answered={rq3.get('rq3_answered_positively', False)}, "
        f"n={rq3.get('n_test_patients', 0)}"
    )

    if "gru_benefit_personalised" in ablation:
        g = ablation["gru_benefit_personalised"]
        print(f"\nGRU benefit: d={g['cohens_d']}, p={g['wilcoxon_p']} — {g['interpretation']}")

    if "algorithm_effect_generalised" in ablation:
        a = ablation["algorithm_effect_generalised"]
        print(f"Algorithm effect (PPO vs DDQN): d={a['ppo_vs_gddqn_cohens_d']}, p={a['wilcoxon_p']}")

    print("\n" + "=" * 70)
    print(f"All results saved to: {output_dir}/")
    print("  test_rq2_results.json")
    print("  test_rq4_personalisation.json")
    print("  test_rq3_transfer.json")
    print("  test_ablation_results.json")
    print("  test_all_agents_vs_rbde.json")
    print("  dapta_test_results.csv")
    print("  cluster_performance.json")
    print("  test_per_agent_arrays/   (raw .npy per agent)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPTA Phase 3: Held-out Test Evaluation"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()
    main(args)