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

METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
SURPRISAL_DIM      = 38
CLINICAL_THRESHOLD = 0.40


# ─────────────────────────────────────────────
# Statistical helpers
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


# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────

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
            initial_states   = list(cluster_states),
            transition_model = transition_model,
            episode_horizon  = episode_horizon,
        )
        cluster_envs[cluster_id] = envs
        logger.info(f"  Cluster {cluster_id}: {len(envs)} environments")
    return cluster_envs


# ─────────────────────────────────────────────
# Agent factories
# ─────────────────────────────────────────────

def make_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    return DDQNAgent(
        state_dim      = STATE_DIM,
        gru_hidden     = 128,
        gru_layers     = 2,
        learning_rate  = 1e-4,
        gamma          = 0.95,
        tau            = 0.005,
        buffer_size    = 10_000,
        batch_size     = 64,
        eps_start      = 1.0,
        eps_end        = 0.05,
        eps_fraction   = 0.3,
        history_len    = 10,
        use_gru        = True,
        checkpoint_path = checkpoint_path,
    )


def make_no_gru_ddqn(checkpoint_path: str = None) -> DDQNAgent:
    return DDQNAgent(
        state_dim      = STATE_DIM,
        gru_hidden     = 128,
        gru_layers     = 2,
        learning_rate  = 1e-4,
        gamma          = 0.95,
        tau            = 0.005,
        buffer_size    = 10_000,
        batch_size     = 64,
        eps_start      = 1.0,
        eps_end        = 0.05,
        eps_fraction   = 0.3,
        history_len    = 10,
        use_gru        = False,
        checkpoint_path = checkpoint_path,
    )


class PPOWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def select_action(self, state: np.ndarray) -> int:
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episodes(
    agent,
    envs:    List[TherapyEnv],
    is_ddqn: bool = True,
    greedy:  bool = True,
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


def answer_rq4(
    dapta_disc:     np.ndarray,
    g_ddqn_disc:    np.ndarray,
    dapta_surp:     np.ndarray,
    g_ddqn_surp:    np.ndarray,
    cluster_labels: np.ndarray,
    profiles_data:  list,
    alpha:          float = 0.05,
    best_pers_name: str   = "DAPTA",
    best_gen_name:  str   = "G_DDQN",
) -> dict:
    out      = {}
    out["best_personalised_agent"] = best_pers_name
    out["best_generalised_agent"]  = best_gen_name
    p_values = []
    entries  = {}

    for i, metric in enumerate(METRIC_NAMES):
        d            = cohens_d_paired(g_ddqn_disc[:, i], dapta_disc[:, i])
        stat, p      = wilcoxon_test(dapta_disc[:, i], g_ddqn_disc[:, i])
        ci_lo, ci_hi = bootstrap_ci(dapta_disc[:, i] - g_ddqn_disc[:, i])
        p_values.append(p)
        entries[metric] = {
            "personalised_mean":     round(float(np.mean(dapta_disc[:, i])),   4),
            "generalised_mean":      round(float(np.mean(g_ddqn_disc[:, i])), 4),
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

    out["pooled_personalised_vs_generalised"] = entries
    out["ciu_variance_reduction_pct"]         = round(float(var_reduction), 2)
    out["personalisation_beneficial"]         = bool(
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
        dominant       = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
        subtype_counts = {s: subtypes.count(s) for s in set(subtypes)}
        per_cluster[str(c)] = {
            "n":                     int(n),
            "dominant_subtype":      dominant,
            "subtype_counts":        subtype_counts,
            "personalised_ciu_mean": round(float(np.mean(d_ciu)), 4),
            "generalised_ciu_mean":  round(float(np.mean(g_ciu)), 4),
            "cohens_d_ciu":          round(d_val, 3),
            "personalised_wins":     bool(np.mean(d_ciu) > np.mean(g_ciu)),
        }

    wab_aqs = np.array([
        float(p.get("wab_aq") or 55.0) for p in profiles_data
    ], dtype=np.float32)

    personalisation_benefit = ciu_dapta - ciu_gddqn
    if len(wab_aqs) >= 5:
        r_wab, p_wab = scipy_stats.pearsonr(wab_aqs, personalisation_benefit)
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
        low_mask  = wab_aqs < 50
        high_mask = wab_aqs >= 50
        out["severity_subgroup_analysis"] = {
            "low_wabaq_n":           int(low_mask.sum()),
            "high_wabaq_n":          int(high_mask.sum()),
            "low_wabaq_dapta_mean":  round(float(np.mean(ciu_dapta[low_mask])),  4) if low_mask.sum()  > 0 else None,
            "high_wabaq_dapta_mean": round(float(np.mean(ciu_dapta[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_gddqn_mean":  round(float(np.mean(ciu_gddqn[low_mask])),  4) if low_mask.sum()  > 0 else None,
            "high_wabaq_gddqn_mean": round(float(np.mean(ciu_gddqn[high_mask])), 4) if high_mask.sum() > 0 else None,
            "low_wabaq_d":  round(cohens_d_paired(ciu_gddqn[low_mask],  ciu_dapta[low_mask]),  3) if low_mask.sum()  > 1 else None,
            "high_wabaq_d": round(cohens_d_paired(ciu_gddqn[high_mask], ciu_dapta[high_mask]), 3) if high_mask.sum() > 1 else None,
        }

    out["per_cluster"]             = per_cluster
    out["rq4_answered_positively"] = out["personalisation_beneficial"]
    return out


def answer_ablation(
    all_agent_results: Dict[str, np.ndarray],
    rbde_disc:         np.ndarray,
) -> dict:
    out = {}

    per_agent = {}
    for agent_name, disc in all_agent_results.items():
        d_ciu = cohens_d_paired(rbde_disc[:, 0], disc[:, 0])
        per_agent[agent_name] = {
            metric: {
                "mean":              round(float(np.mean(disc[:, i])), 4),
                "cohens_d_vs_rbde":  round(cohens_d_paired(rbde_disc[:, i], disc[:, i]), 3),
                "effect_size":       effect_size_label(cohens_d_paired(rbde_disc[:, i], disc[:, i])),
            }
            for i, metric in enumerate(METRIC_NAMES)
        }
        per_agent[agent_name]["ciu_cohens_d_vs_rbde"] = round(d_ciu, 3)

    out["per_agent_summary"] = per_agent

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
            "ppo_wins":              bool(
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
# Main
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
    split_labels   = pes_data["split_labels"]   # "train" / "val" / "test"

    with open("outputs/dae/patient_profiles.json") as f:
        profiles_data = json.load(f)

    transition_model = TransitionModel(
        hidden_sizes    = (128, 64),
        dropout         = 0.2,
        checkpoint_path = str(pes_dir / "transition_model.pt"),
        device          = args.device,
    )
    transition_model.load()
    logger.info("Transition model loaded.")

    # ── 2. Filter to TRAIN patients only ────────────────────────
    # This is the data leakage fix.
    # Agents are trained exclusively on train environments.
    # Val and test patients are held out and evaluated in run_evaluation.py.
    logger.info("\n[2/6] Filtering to train-only environments...")

    train_mask            = split_labels == "train"
    train_initial_states  = initial_states[train_mask]
    train_cluster_labels  = cluster_labels[train_mask]
    train_profiles        = [
        p for p, m in zip(profiles_data, train_mask) if m
    ]

    logger.info(
        f"  Train: {train_mask.sum()}  "
        f"Val: {(split_labels == 'val').sum()}  "
        f"Test: {(split_labels == 'test').sum()}  "
        f"(val and test excluded from RL training)"
    )

    cluster_envs = load_environments(
        transition_model = transition_model,
        initial_states   = train_initial_states,
        cluster_labels   = train_cluster_labels,
        episode_horizon  = 20,
    )
    all_envs   = [env for envs in cluster_envs.values() for env in envs]
    n_clusters = len(cluster_envs)
    logger.info(f"  Train environments: {len(all_envs)}, clusters: {n_clusters}")

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
            agent           = agent,
            envs            = envs,
            total_steps     = args.total_steps,
            eval_freq       = max(100, args.total_steps // 50),
            n_eval_episodes = min(10, len(envs)),
        )
        agent.save()
        cluster_agents[cluster_id] = agent
        training_logs[f"dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    Cluster {cluster_id} saved.")

    # 3b — G-DDQN: GRU-DDQN generalised
    logger.info("\n  3b. G-DDQN — GRU-DDQN generalised")
    g_ddqn = make_gru_ddqn(checkpoint_path=str(output_dir / "ddqn_generalised.pt"))
    training_logs["g_ddqn"] = train_ddqn(
        agent           = g_ddqn,
        envs            = all_envs,
        total_steps     = args.total_steps,
        eval_freq       = max(100, args.total_steps // 50),
        n_eval_episodes = min(20, len(all_envs)),
    )
    g_ddqn.save()
    logger.info("    G-DDQN saved.")

    # 3c — No-GRU DAPTA: No-GRU DDQN cluster-specific
    logger.info("\n  3c. No-GRU DAPTA — No-GRU DDQN cluster-specific")
    no_gru_cluster_agents: Dict[int, DDQNAgent] = {}
    for cluster_id, envs in cluster_envs.items():
        if not envs:
            continue
        ckpt  = str(output_dir / f"no_gru_ddqn_cluster_{cluster_id}.pt")
        agent = make_no_gru_ddqn(checkpoint_path=ckpt)
        logs  = train_ddqn(
            agent           = agent,
            envs            = envs,
            total_steps     = args.total_steps,
            eval_freq       = max(100, args.total_steps // 50),
            n_eval_episodes = min(10, len(envs)),
        )
        agent.save()
        no_gru_cluster_agents[cluster_id] = agent
        training_logs[f"no_gru_dapta_cluster_{cluster_id}"] = logs
        logger.info(f"    No-GRU cluster {cluster_id} saved.")

    # 3d — No-GRU G-DDQN: No-GRU DDQN generalised
    logger.info("\n  3d. No-GRU G-DDQN — No-GRU DDQN generalised")
    no_gru_g_ddqn = make_no_gru_ddqn(
        checkpoint_path = str(output_dir / "no_gru_ddqn_generalised.pt")
    )
    training_logs["no_gru_g_ddqn"] = train_ddqn(
        agent           = no_gru_g_ddqn,
        envs            = all_envs,
        total_steps     = args.total_steps,
        eval_freq       = max(100, args.total_steps // 50),
        n_eval_episodes = min(20, len(all_envs)),
    )
    no_gru_g_ddqn.save()
    logger.info("    No-GRU G-DDQN saved.")

    # ── 4. Train PPO agents ──────────────────────────────────────
    logger.info(f"\n[4/6] Training PPO agents...")

    ppo_generalised  = None
    ppo_cluster_models = {}

    if not args.skip_ppo:
        logger.info("\n  4a. PPO generalised — pooled train environments")
        ppo_save       = str(output_dir / "ppo_generalised")
        ppo_generalised = train_ppo(
            env         = all_envs[0],
            total_steps = args.total_steps,
            save_path   = ppo_save,
        )
        training_logs["ppo_generalised"] = {
            "status": "trained" if ppo_generalised else "skipped_no_sb3",
            "steps":  args.total_steps,
        }

        logger.info("\n  4b. PPO personalised — cluster-specific train environments")
        for cluster_id, envs in cluster_envs.items():
            if not envs:
                continue
            ppo_c_save  = str(output_dir / f"ppo_cluster_{cluster_id}")
            ppo_cluster = train_ppo(
                env         = envs[0],
                total_steps = args.total_steps,
                save_path   = ppo_c_save,
            )
            if ppo_cluster:
                ppo_cluster_models[cluster_id] = ppo_cluster
                training_logs[f"ppo_cluster_{cluster_id}"] = {
                    "status": "trained",
                    "steps":  args.total_steps,
                }
    else:
        logger.info("  Skipping PPO (--skip_ppo flag set).")

    # ── 5. Collect improvement arrays (training environments only) ──
    # Note: these results are on the TRAINING set and are used only to
    # compute interim training statistics and save checkpoints.
    # The authoritative evaluation on the held-out test set is performed
    # by run_evaluation.py using split_labels to load test environments.
    logger.info("\n[5/6] Collecting train-set improvement arrays...")

    # DAPTA
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

    # G-DDQN
    g_ddqn.eps = 0.0
    g_ddqn_disc, g_ddqn_surp, _ = run_episodes(g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # No-GRU DAPTA
    no_gru_dapta_disc_list, no_gru_dapta_surp_list = [], []
    for cluster_id, agent in no_gru_cluster_agents.items():
        envs = cluster_envs[cluster_id]
        if not envs:
            continue
        agent.eps = 0.0
        disc, surp, _ = run_episodes(agent, envs, is_ddqn=True, greedy=True)
        no_gru_dapta_disc_list.append(disc)
        no_gru_dapta_surp_list.append(surp)
    no_gru_dapta_disc = (
        np.concatenate(no_gru_dapta_disc_list, axis=0)
        if no_gru_dapta_disc_list else np.zeros((0, 5))
    )

    # No-GRU G-DDQN
    no_gru_g_ddqn.eps = 0.0
    no_gru_g_ddqn_disc, _, _ = run_episodes(no_gru_g_ddqn, all_envs, is_ddqn=True, greedy=True)

    # PPO generalised
    ppo_gen_disc = np.zeros((0, 5))
    if ppo_generalised:
        ppo_gen_wrapper   = PPOWrapper(ppo_generalised)
        ppo_gen_disc_list = []
        for env in all_envs:
            disc, _, _ = run_episodes(ppo_gen_wrapper, [env], is_ddqn=False, greedy=True)
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
            ppo_pers_wrapper = PPOWrapper(ppo_model)
            for env in envs:
                disc, _, _ = run_episodes(ppo_pers_wrapper, [env], is_ddqn=False, greedy=True)
                ppo_pers_disc_list.append(disc)
        if ppo_pers_disc_list:
            ppo_pers_disc = np.concatenate(ppo_pers_disc_list, axis=0)

    # Baselines (on train environments)
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

    rts = RandomBaseline()
    rts_disc_list = []
    for env in all_envs:
        state, _ = env.reset()
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rts_disc_list.append(env.get_cumulative_discourse_improvement()[:5])
    rts_disc = np.stack(rts_disc_list)

    # Save training-set improvement arrays
    np.save(str(output_dir / "train_improvements_DAPTA.npy"),         dapta_disc)
    np.save(str(output_dir / "train_improvements_G_DDQN.npy"),        g_ddqn_disc)
    np.save(str(output_dir / "train_improvements_NO_GRU_DAPTA.npy"),  no_gru_dapta_disc)
    np.save(str(output_dir / "train_improvements_NO_GRU_G_DDQN.npy"), no_gru_g_ddqn_disc)
    np.save(str(output_dir / "train_improvements_PPO_GEN.npy"),       ppo_gen_disc)
    np.save(str(output_dir / "train_improvements_PPO_PERS.npy"),      ppo_pers_disc)
    np.save(str(output_dir / "train_improvements_RBDE.npy"),          rbde_disc)
    np.save(str(output_dir / "train_improvements_RTS.npy"),           rts_disc)

    # ── 6. Training-set RQ statistics (interim monitoring only) ──
    # These are computed on the TRAINING SET.
    # The final reported results come from run_evaluation.py (test set).
    logger.info("\n[6/6] Computing interim training-set statistics...")

    all_agent_results = {"DAPTA": dapta_disc, "G_DDQN": g_ddqn_disc}
    if len(no_gru_dapta_disc)  > 0: all_agent_results["NO_GRU_DAPTA"]  = no_gru_dapta_disc
    if len(no_gru_g_ddqn_disc) > 0: all_agent_results["NO_GRU_G_DDQN"] = no_gru_g_ddqn_disc
    if len(ppo_gen_disc)        > 0: all_agent_results["PPO_GENERALISED"]  = ppo_gen_disc
    if len(ppo_pers_disc)       > 0: all_agent_results["PPO_PERSONALISED"] = ppo_pers_disc

    rq2_train = answer_rq2(dapta_disc, rbde_disc, rts_disc)
    with open(output_dir / "train_rq2_interim.json", "w") as f:
        json.dump(rq2_train, f, indent=2)

    ablation = answer_ablation(all_agent_results, rbde_disc)
    with open(output_dir / "train_ablation_interim.json", "w") as f:
        json.dump(ablation, f, indent=2)

    PERSONALISED_AGENTS = {"DAPTA", "NO_GRU_DAPTA", "PPO_PERSONALISED"}
    GENERALISED_AGENTS  = {"G_DDQN", "NO_GRU_G_DDQN", "PPO_GENERALISED"}

    pers_candidates = {k: v for k, v in all_agent_results.items() if k in PERSONALISED_AGENTS and len(v) > 0}
    gen_candidates  = {k: v for k, v in all_agent_results.items() if k in GENERALISED_AGENTS  and len(v) > 0}

    best_pers_name = max(pers_candidates, key=lambda k: float(np.mean(pers_candidates[k][:, 0])))
    best_gen_name  = max(gen_candidates,  key=lambda k: float(np.mean(gen_candidates[k][:, 0])))
    best_pers_disc = pers_candidates[best_pers_name]
    best_gen_disc  = gen_candidates[best_gen_name]

    min_n     = min(len(best_pers_disc), len(best_gen_disc))
    rq4_train = answer_rq4(
        dapta_disc     = best_pers_disc[:min_n],
        g_ddqn_disc    = best_gen_disc[:min_n],
        dapta_surp     = dapta_surp[:min_n],
        g_ddqn_surp    = g_ddqn_surp[:min_n],
        cluster_labels = dapta_labels[:min_n],
        profiles_data  = train_profiles[:min_n],
        best_pers_name = best_pers_name,
        best_gen_name  = best_gen_name,
    )
    with open(output_dir / "train_rq4_interim.json", "w") as f:
        json.dump(rq4_train, f, indent=2)

    with open(output_dir / "training_logs.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
            if isinstance(obj, list):  return [convert(x) for x in obj]
            if isinstance(obj, dict):  return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(training_logs), f, indent=2)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DAPTA RL TRAINING COMPLETE (train-set interim results)")
    print("=" * 70)
    print("\n⚠️  These results are on the TRAINING SET.")
    print("   Run experiments/run_evaluation.py for held-out test results.\n")

    print("All agents — CIU rate mean improvement (train set):")
    for name, arr in all_agent_results.items():
        if len(arr) > 0:
            d = cohens_d_paired(rbde_disc[:, 0], arr[:, 0])
            print(f"  {name:22} mean={np.mean(arr[:,0]):.4f}  d_vs_RBDE={d:.3f}")
    print(f"  {'RBDE':22} mean={np.mean(rbde_disc[:,0]):.4f}  (baseline)")
    print(f"  {'RTS':22} mean={np.mean(rts_disc[:,0]):.4f}  (baseline)")

    print(f"\nRecommended config: {ablation.get('recommended_configuration', {}).get('agent', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"Checkpoints saved to: {output_dir}/")
    print("  ddqn_cluster_*.pt, ddqn_generalised.pt")
    print("  no_gru_ddqn_cluster_*.pt, no_gru_ddqn_generalised.pt")
    print("  ppo_generalised.zip (if trained)")
    print("  train_rq2_interim.json  (training set only — not final results)")
    print("  train_rq4_interim.json  (training set only — not final results)")
    print("=" * 70)
    logger.info("Next step: python experiments/run_evaluation.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPTA Phase 2b: RL Agent Training"
    )
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
    args = parser.parse_args()
    main(args)