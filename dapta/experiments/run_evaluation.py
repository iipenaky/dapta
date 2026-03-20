import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats

from dapta.utils.logger import get_logger
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.pes.transition_model import TransitionModel
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import RuleBasedBaseline, RandomBaseline
from dapta.dae.state_builder import STATE_DIM

logger = get_logger(__name__, log_file="logs/evaluation.log")

METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
CLINICAL_THRESHOLD = 0.40  
SURPRISAL_DIM      = 38     

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


def bonferroni(p_values: List[float], alpha: float = 0.05):
    n         = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    sig       = [p <= alpha for p in corrected]
    return corrected, sig


def bootstrap_ci(diff: np.ndarray, n_resamples: int = 1000, seed: int = 42):
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

def run_ddqn_episode(
    agent: DDQNAgent,
    env:   TherapyEnv,
) -> Tuple[np.ndarray, float]:
    """Returns (discourse_improvement (5,), surprisal_improvement)."""
    state, _       = env.reset()
    state_history  = []
    action_history = []
    initial_surp   = float(state[SURPRISAL_DIM])

    for _ in range(env.episode_horizon):
        history_tensor = agent.build_history_tensor(state_history, action_history)
        action         = agent.select_action(state, history_tensor, greedy=True)
        state_history.append(state.copy())
        action_history.append(action)
        state, _, done, _, _ = env.step(action)
        if done:
            break

    disc_imp = env.get_cumulative_discourse_improvement()[:5]
    surp_imp = initial_surp - float(state[SURPRISAL_DIM])
    return disc_imp, surp_imp


def run_ppo_episode(
    agent,
    env: TherapyEnv,
) -> Tuple[np.ndarray, float]:
    state, _     = env.reset()
    initial_surp = float(state[SURPRISAL_DIM])

    for _ in range(env.episode_horizon):
        action, _ = agent.predict(state.reshape(1, -1), deterministic=True)
        state, _, done, _, _ = env.step(int(action))
        if done:
            break

    disc_imp = env.get_cumulative_discourse_improvement()[:5]
    surp_imp = initial_surp - float(state[SURPRISAL_DIM])
    return disc_imp, surp_imp


def run_baseline_episode(
    baseline,
    env: TherapyEnv,
) -> Tuple[np.ndarray, float]:
    state, _ = env.reset()
    initial_surp = float(state[SURPRISAL_DIM])

    if hasattr(baseline, "reset"):
        baseline.reset()

    for _ in range(env.episode_horizon):
        action = baseline.select_action(state)
        state, _, done, _, _ = env.step(action)
        if done:
            break

    disc_imp = env.get_cumulative_discourse_improvement()[:5]
    surp_imp = initial_surp - float(state[SURPRISAL_DIM])
    return disc_imp, surp_imp


# ---------------------------------------------------------------------------
# Load agents
# ---------------------------------------------------------------------------

def load_ddqn_agents(
    rl_dir:         Path,
    cluster_labels: np.ndarray,
    state_dim:      int,
    n_actions:      int,
    device:         str,
) -> Tuple[Dict[int, DDQNAgent], DDQNAgent]:
    n_clusters     = int(cluster_labels.max()) + 1
    cluster_agents = {}

    for cluster_id in range(n_clusters):
        agent_path = rl_dir / f"ddqn_cluster_{cluster_id}.pt"
        if not agent_path.exists():
            logger.warning(f"No saved agent for cluster {cluster_id}, skipping.")
            continue
        agent = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device=device)
        agent.load(agent_path)
        agent.eps = 0.0   # greedy at evaluation — fixed from epsilon bug
        cluster_agents[cluster_id] = agent
        logger.info(f"  Loaded DDQN cluster {cluster_id} from {agent_path}")

    g_ddqn_path = rl_dir / "ddqn_generalised.pt"
    g_ddqn      = DDQNAgent(state_dim=state_dim, n_actions=n_actions, device=device)
    g_ddqn.load(g_ddqn_path)
    g_ddqn.eps = 0.0   # greedy at evaluation — fixed from epsilon bug
    logger.info(f"  Loaded G-DDQN from {g_ddqn_path}")

    return cluster_agents, g_ddqn


# ---------------------------------------------------------------------------
# Run all agents on test environments
# ---------------------------------------------------------------------------

def evaluate_all_agents(
    test_envs:           List[TherapyEnv],
    test_cluster_labels: np.ndarray,
    cluster_agents:      Dict[int, DDQNAgent],
    g_ddqn:              DDQNAgent,
    rbde:                RuleBasedBaseline,
    rts:                 RandomBaseline,
    ppo_agent=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    agent_names = ["DAPTA", "G_DDQN", "RBDE", "RTS"]
    if ppo_agent is not None:
        agent_names.append("PPO")

    disc_results = {k: [] for k in agent_names}
    surp_results = {k: [] for k in agent_names}

    for i, env in enumerate(test_envs):
        cluster_id = int(test_cluster_labels[i])

        # DAPTA — patient-specific agent for this cluster
        if cluster_id in cluster_agents:
            d_imp, s_imp = run_ddqn_episode(cluster_agents[cluster_id], env)
        else:
            logger.warning(f"  No cluster agent for cluster {cluster_id}, falling back to G-DDQN.")
            d_imp, s_imp = run_ddqn_episode(g_ddqn, env)
        disc_results["DAPTA"].append(d_imp)
        surp_results["DAPTA"].append(s_imp)

        # G-DDQN
        d_imp, s_imp = run_ddqn_episode(g_ddqn, env)
        disc_results["G_DDQN"].append(d_imp)
        surp_results["G_DDQN"].append(s_imp)

        # RBDE
        d_imp, s_imp = run_baseline_episode(rbde, env)
        disc_results["RBDE"].append(d_imp)
        surp_results["RBDE"].append(s_imp)

        # RTS
        d_imp, s_imp = run_baseline_episode(rts, env)
        disc_results["RTS"].append(d_imp)
        surp_results["RTS"].append(s_imp)

        # PPO
        if ppo_agent is not None:
            d_imp, s_imp = run_ppo_episode(ppo_agent, env)
            disc_results["PPO"].append(d_imp)
            surp_results["PPO"].append(s_imp)

    return (
        {k: np.stack(v) for k, v in disc_results.items()},
        {k: np.array(v, dtype=np.float32) for k, v in surp_results.items()},
    )


# ---------------------------------------------------------------------------
# RQ2: RL vs baselines
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RQ3: Transfer to naturalistic speech
# ---------------------------------------------------------------------------

def answer_rq3(
    dapta_disc: np.ndarray,
    dapta_surp: np.ndarray,
    rbde_disc:  np.ndarray,
    rbde_surp:  np.ndarray,
) -> dict:
    out = {}

    d_surp       = cohens_d_paired(rbde_surp, dapta_surp)
    stat, p      = wilcoxon_test(dapta_surp, rbde_surp)
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

    # CIU improvement vs surprisal improvement correlation
    ciu_imp = dapta_disc[:, 0]
    if len(ciu_imp) >= 5:
        r, p_corr = scipy_stats.pearsonr(ciu_imp, dapta_surp)
        out["discourse_to_surprisal_correlation"] = {
            "pearson_r":     round(float(r),      3),
            "p_value":       round(float(p_corr), 4),
            "significant":   bool(p_corr < 0.05),
            "n":             int(len(ciu_imp)),
            "interpretation": (
                "discourse gains transfer to naturalistic speech"
                if r > 0.3 and p_corr < 0.05
                else "weak or no transfer detected"
            ),
        }

    surp_sig = out["surprisal_improvement"]["significant"]
    corr_sig = out.get("discourse_to_surprisal_correlation", {}).get("significant", False)
    out["rq3_answered_positively"] = bool(surp_sig or corr_sig)

    return out


# ---------------------------------------------------------------------------
# RQ4: Patient-specific vs generalised RL
# ---------------------------------------------------------------------------

def answer_rq4(
    dapta_disc:      np.ndarray,
    g_ddqn_disc:     np.ndarray,
    dapta_surp:      np.ndarray,
    g_ddqn_surp:     np.ndarray,
    cluster_labels:  np.ndarray,
    profiles_data:   list,
    alpha:           float = 0.05,
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
            "gddqn_mean":            round(float(np.mean(g_ddqn_disc[:, i])),  4),
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

    out["pooled_dapta_vs_gddqn"]     = entries
    out["ciu_variance_reduction_pct"] = round(float(var_reduction), 2)
    out["personalisation_beneficial"] = bool(
        np.mean(ciu_dapta) > np.mean(ciu_gddqn) and any(sig)
    )

    # Per-cluster breakdown
    unique_clusters = np.unique(cluster_labels)
    per_cluster     = {}

    for c in unique_clusters:
        mask = cluster_labels == c
        n    = int(mask.sum())
        if n == 0:
            continue

        d_ciu = dapta_disc[mask, 0]
        g_ciu = g_ddqn_disc[mask, 0]
        d_val = cohens_d_paired(g_ciu, d_ciu)

        cluster_profiles = [p for p, m in zip(profiles_data, mask) if m]
        subtypes         = [p.get("aphasia_subtype", "Other") for p in cluster_profiles]
        dominant         = max(set(subtypes), key=subtypes.count) if subtypes else "Unknown"
        subtype_counts   = {s: subtypes.count(s) for s in set(subtypes)}

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

    # WAB-AQ moderates personalisation benefit
    wab_aqs = np.array([float(p.get("wab_aq") or 55.0) for p in profiles_data], dtype=np.float32)
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
            "low_wabaq_d":           round(cohens_d_paired(ciu_gddqn[low_mask],  ciu_dapta[low_mask]),  3) if low_mask.sum()  > 1 else None,
            "high_wabaq_d":          round(cohens_d_paired(ciu_gddqn[high_mask], ciu_dapta[high_mask]), 3) if high_mask.sum() > 1 else None,
        }

    out["rq4_answered_positively"] = out["personalisation_beneficial"]
    return out


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_results(rq2: dict, rq3: dict, rq4: dict) -> None:
    print("\n" + "=" * 70)
    print("DAPTA HELD-OUT TEST SET EVALUATION — Chapter 4 Results")
    print("=" * 70)

    print("\nTable 1: Mean Discourse Improvement (test set)")
    print("-" * 70)
    print(f"{'Metric':<28} {'DAPTA':>8} {'G_DDQN':>8} {'RBDE':>8} {'RTS':>8}")
    print("-" * 70)
    for metric in METRIC_NAMES:
        dapta_m  = rq2["DAPTA_vs_RBDE"][metric]["dapta_mean"]
        rbde_m   = rq2["DAPTA_vs_RBDE"][metric]["baseline_mean"]
        rts_m    = rq2["DAPTA_vs_RTS"][metric]["baseline_mean"]
        gddqn_m  = rq4["pooled_dapta_vs_gddqn"][metric]["gddqn_mean"]
        print(f"  {metric:<26} {dapta_m:>8.4f} {gddqn_m:>8.4f} {rbde_m:>8.4f} {rts_m:>8.4f}")
    print("-" * 70)

    print("\nRQ2: Does RL outperform rule-based sequencing?")
    for comp in ["DAPTA_vs_RBDE", "DAPTA_vs_RTS"]:
        s = rq2["summary"][comp]
        print(
            f"  {comp}: {s['n_clinically_meaningful']}/5 metrics clinically meaningful "
            f"— {'ANSWERED ✓' if s['rq2_answered_positively'] else 'NOT ANSWERED'}"
        )

    print("\nRQ3: Do discourse gains transfer to naturalistic speech?")
    si = rq3["surprisal_improvement"]
    print(f"  DAPTA surprisal improvement : {si['dapta_mean_surp_improvement']:.4f}")
    print(f"  RBDE  surprisal improvement : {si['rbde_mean_surp_improvement']:.4f}")
    print(f"  Cohen's d = {si['cohens_d']:.3f}  p = {si['wilcoxon_p']:.4f}")
    if "discourse_to_surprisal_correlation" in rq3:
        c = rq3["discourse_to_surprisal_correlation"]
        print(f"  CIU–Surprisal r = {c['pearson_r']}  p = {c['p_value']}")
    print(f"  {'ANSWERED ✓' if rq3['rq3_answered_positively'] else 'NOT ANSWERED'}")

    print("\nRQ4: Does patient-specific RL outperform generalised RL?")
    print(f"  Personalisation beneficial  : {rq4['personalisation_beneficial']}")
    print(f"  CIU variance reduction      : {rq4['ciu_variance_reduction_pct']:.1f}%")
    if "wabaq_moderates_personalisation" in rq4:
        w = rq4["wabaq_moderates_personalisation"]
        print(f"  WAB-AQ moderates benefit    : r={w['pearson_r']}  p={w['p_value']}")
        print(f"  {w['interpretation']}")
    print(f"  {'ANSWERED ✓' if rq4['rq4_answered_positively'] else 'NOT ANSWERED'}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DAPTA Phase 3: Held-out Test Evaluation")
    p.add_argument("--dae_dir",  default="outputs/dae")
    p.add_argument("--pes_dir",  default="outputs/pes")
    p.add_argument("--rl_dir",   default="outputs/rl")
    p.add_argument("--out_dir",  default="outputs/evaluation")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--n_actions", type=int, default=12)
    p.add_argument("--skip_ppo", action="store_true")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dae_dir = Path(args.dae_dir)
    pes_dir = Path(args.pes_dir)
    rl_dir  = Path(args.rl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 3: Held-out Test Set Evaluation")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load test set from splits.json
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Loading held-out test set...")
    dae_data        = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    state_vectors   = dae_data["state_vectors"]
    all_session_ids = list(dae_data["session_ids"])
    state_dim       = state_vectors.shape[1]

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    with open(dae_dir / "patient_profiles.json") as f:
        profiles_data = json.load(f)

    sid_to_idx   = {sid: i for i, sid in enumerate(all_session_ids)}
    test_ids     = splits["test"]
    test_indices = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    test_states  = state_vectors[test_indices]
    test_profiles = [profiles_data[i] for i in test_indices]

    logger.info(f"  Test patients: {len(test_indices)}  state_dim: {state_dim}")

    # ------------------------------------------------------------------
    # 2. Load cluster labels for test patients
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Loading cluster assignments...")
    with open(pes_dir / "cluster_assignments.json") as f:
        cluster_info = json.load(f)

    assignments         = cluster_info["assignments"]
    test_cluster_labels = np.array([
        assignments.get(sid, 0) for sid in test_ids if sid in sid_to_idx
    ], dtype=int)

    # ------------------------------------------------------------------
    # 3. Build test environments
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Building test environments...")
    transition_model = TransitionModel(
            hidden_sizes=(128, 64),
            dropout=0.2,
        checkpoint_path=pes_dir / "transition_model.pt"
    )
    transition_model.load()

    test_envs = build_env_population(
        transition_model=transition_model,
        initial_states=list(test_states),
        episode_horizon=20,
    )
    logger.info(f"  Built {len(test_envs)} test environments.")

    # ------------------------------------------------------------------
    # 4. Load trained agents
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Loading trained agents...")
    cluster_agents, g_ddqn = load_ddqn_agents(
        rl_dir=rl_dir,
        cluster_labels=test_cluster_labels,
        state_dim=state_dim,
        n_actions=args.n_actions,
        device=args.device,
    )
    rbde = RuleBasedBaseline()
    rts  = RandomBaseline()

    ppo_agent = None
    if not args.skip_ppo:
        try:
            from stable_baselines3 import PPO as SB3PPO
            ppo_path = rl_dir / "ppo_agent.zip"
            if ppo_path.exists():
                ppo_agent = SB3PPO.load(str(ppo_path))
                logger.info(f"  PPO loaded from {ppo_path}")
        except Exception as e:
            logger.warning(f"  Could not load PPO: {e}. Skipping.")

    # ------------------------------------------------------------------
    # 5. Run evaluation
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Running evaluation on test set...")
    disc_improvements, surp_improvements = evaluate_all_agents(
        test_envs=test_envs,
        test_cluster_labels=test_cluster_labels,
        cluster_agents=cluster_agents,
        g_ddqn=g_ddqn,
        rbde=rbde,
        rts=rts,
        ppo_agent=ppo_agent,
    )
    logger.info(f"  Evaluated {len(test_envs)} patients.")

    # Save improvement arrays
    for agent, matrix in disc_improvements.items():
        np.save(out_dir / f"improvements_{agent}.npy", matrix)
    for agent, arr in surp_improvements.items():
        np.save(out_dir / f"surp_improvements_{agent}.npy", arr)

    # ------------------------------------------------------------------
    # Answer RQ2, RQ3, RQ4 on test set
    # ------------------------------------------------------------------
    rq2 = answer_rq2(
        disc_improvements["DAPTA"],
        disc_improvements["RBDE"],
        disc_improvements["RTS"],
    )
    rq3 = answer_rq3(
        disc_improvements["DAPTA"],
        surp_improvements["DAPTA"],
        disc_improvements["RBDE"],
        surp_improvements["RBDE"],
    )
    rq4 = answer_rq4(
        dapta_disc=disc_improvements["DAPTA"],
        g_ddqn_disc=disc_improvements["G_DDQN"],
        dapta_surp=surp_improvements["DAPTA"],
        g_ddqn_surp=surp_improvements["G_DDQN"],
        cluster_labels=test_cluster_labels,
        profiles_data=test_profiles,
    )

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
    with open(out_dir / "cluster_performance.json", "w") as f:
        json.dump(cluster_perf, f, indent=2)

    # Save full results
    full_results = {
        "rq2": rq2,
        "rq3": rq3,
        "rq4": rq4,
        "n_test_patients": len(test_envs),
        "note": "Results computed on held-out test set only. Patients not seen during RL training."
    }
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    logger.info(f"  Results saved to {out_dir}/evaluation_results.json")

    print_results(rq2, rq3, rq4)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 Complete.")
    logger.info(f"  Outputs saved to: {out_dir}/")
    logger.info("=" * 60)
    logger.info("Optional: python experiments/rq4_aphasia_only.py --data_dir outputs/evaluation --dae_dir outputs/dae")


if __name__ == "__main__":
    main()