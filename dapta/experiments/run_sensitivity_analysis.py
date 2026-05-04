"""
DAPTA External Validity: Sensitivity Analysis.

Tests whether the core conclusions from run_evaluation.py hold up
across different transition model noise levels. If DAPTA consistently
beats RBDE and personalisation consistently beats generalisation
across all noise assumptions, that strengthens the external validity
argument.

Three noise levels are tested:
    low    : noise_std = 0.001  (near-deterministic transitions)
    medium : noise_std = 0.005  (default — matches run_evaluation.py)
    high   : noise_std = 0.010  (high stochasticity)

Outputs (all in outputs/sensitivity/):
    sensitivity_results.json   — full results across all noise levels
    sensitivity_summary.csv    — one row per noise level, easy to table
    sensitivity_report.txt     — human-readable conclusion

Run after:
    python experiments/run_evaluation.py
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from dapta.pes.transition_model import TransitionModel
from dapta.pes.environment import TherapyEnv, build_env_population
from dapta.prta.ddqn_agent import DDQNAgent
from dapta.prta.trainer import RuleBasedBaseline, RandomBaseline
from dapta.dae.state_builder import PatientProfile, STATE_DIM
from dapta.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/run_sensitivity.log")

NOISE_LEVELS = {
    "low":    0.001,
    "medium": 0.005,
    "high":   0.010,
}

METRIC_NAMES       = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
CLINICAL_THRESHOLD = 0.40
SURPRISAL_DIM      = 38
_N_METRICS_PER_TASK = 5
_STRUCTURED_TASK_INDICES = [0, 1, 2]
_CONVERSATION_TASK_INDEX = 4


# 
# Helpers
# 

def cohens_d_paired(before: np.ndarray, after: np.ndarray) -> float:
    diff = after - before
    std  = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 0 else 0.0


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return 0.0, 1.0
    try:
        from scipy import stats
        stat, p = stats.wilcoxon(diff, alternative="greater")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def load_ddqn(checkpoint_path: str, use_gru: bool) -> DDQNAgent:
    agent = DDQNAgent(
        state_dim       = STATE_DIM,
        gru_hidden      = 128,
        gru_layers      = 2,
        use_gru         = use_gru,
        checkpoint_path = checkpoint_path,
    )
    agent.load()
    agent.eps = 0.0
    return agent


def run_episode(
    agent,
    env:     TherapyEnv,
    is_ddqn: bool = True,
) -> np.ndarray:
    """Run a greedy episode. Returns full 25-dim discourse improvement."""
    state, _       = env.reset()
    state_history  = []
    action_history = []

    for _ in range(env.episode_horizon):
        if is_ddqn:
            history_tensor = agent.build_history_tensor(state_history, action_history)
            action = agent.select_action(state, history_tensor, greedy=True)
        else:
            action = agent.select_action(state)

        state_history.append(state.copy())
        action_history.append(action)
        next_state, _, done, _, _ = env.step(action)
        state = next_state
        if done:
            break

    return env.get_cumulative_discourse_improvement()


def evaluate_at_noise(
    noise_std:             float,
    test_states:           np.ndarray,
    transition_model:      TransitionModel,
    dapta_cluster_agents:  Dict[int, DDQNAgent],
    g_ddqn:                DDQNAgent,
    predicted_clusters:    np.ndarray,
    n_clusters:            int,
) -> dict:
    """
    Rebuild test environments at a given noise level and run all agents.
    Returns a dict of per-agent discourse improvement arrays.
    """
    test_envs = build_env_population(
        initial_states   = list(test_states),
        transition_model = transition_model,
        episode_horizon  = 20,
        noise_std        = noise_std,
    )

    results = {}

    # DAPTA — personalised
    if dapta_cluster_agents:
        disc_list = []
        for env, cluster_id in zip(test_envs, predicted_clusters):
            agent = dapta_cluster_agents.get(int(cluster_id)) or dapta_cluster_agents.get(0)
            if agent is None:
                disc_list.append(np.zeros(25, dtype=np.float32))
                continue
            disc_list.append(run_episode(agent, env, is_ddqn=True))
        results["DAPTA"] = np.stack(disc_list)

    # G-DDQN — generalised
    if g_ddqn:
        disc_list = [run_episode(g_ddqn, env, is_ddqn=True) for env in test_envs]
        results["G_DDQN"] = np.stack(disc_list)

    # RBDE baseline
    rbde = RuleBasedBaseline()
    rbde_list = []
    for env in test_envs:
        state, _ = env.reset()
        rbde.reset()
        for _ in range(env.episode_horizon):
            action = rbde.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rbde_list.append(env.get_cumulative_discourse_improvement())
    results["RBDE"] = np.stack(rbde_list)

    # RTS baseline
    rts = RandomBaseline()
    rts_list = []
    for env in test_envs:
        state, _ = env.reset()
        for _ in range(env.episode_horizon):
            action = rts.select_action(state)
            state, _, done, _, _ = env.step(action)
            if done:
                break
        rts_list.append(env.get_cumulative_discourse_improvement())
    results["RTS"] = np.stack(rts_list)

    return results


def summarise_noise_level(
    noise_label: str,
    noise_std:   float,
    agent_results: Dict[str, np.ndarray],
) -> dict:
    """
    Compute the key statistics for one noise level.
    Returns a dict that feeds into both the JSON report and CSV summary.
    """
    dapta = agent_results.get("DAPTA")
    gddqn = agent_results.get("G_DDQN")
    rbde  = agent_results["RBDE"]
    rts   = agent_results["RTS"]

    out = {
        "noise_label": noise_label,
        "noise_std":   noise_std,
        "n_patients":  int(len(rbde)),
    }

    # RQ2 — DAPTA vs RBDE (CIU rate)
    if dapta is not None:
        d_rq2        = cohens_d_paired(rbde[:, 0], dapta[:, 0])
        _, p_rq2     = wilcoxon_test(dapta[:, 0], rbde[:, 0])
        n_clin       = sum(
            1 for i in range(5)
            if abs(cohens_d_paired(rbde[:, i], dapta[:, i])) >= CLINICAL_THRESHOLD
        )
        out["rq2"] = {
            "dapta_ciu_mean":          round(float(np.mean(dapta[:, 0])), 4),
            "rbde_ciu_mean":           round(float(np.mean(rbde[:,  0])), 4),
            "cohens_d_dapta_vs_rbde":  round(d_rq2, 3),
            "effect_size":             effect_size_label(d_rq2),
            "wilcoxon_p":              round(float(p_rq2), 4),
            "significant":             bool(p_rq2 < 0.05),
            "n_metrics_clin":          n_clin,
            "rq2_answered_positively": n_clin >= 2,
            "dapta_beats_rbde":        bool(np.mean(dapta[:, 0]) > np.mean(rbde[:, 0])),
            "dapta_beats_rts":         bool(np.mean(dapta[:, 0]) > np.mean(rts[:,  0])),
        }

    # RQ4 — DAPTA vs G-DDQN (personalised vs generalised)
    if dapta is not None and gddqn is not None:
        d_rq4    = cohens_d_paired(gddqn[:, 0], dapta[:, 0])
        _, p_rq4 = wilcoxon_test(dapta[:, 0], gddqn[:, 0])
        out["rq4"] = {
            "dapta_ciu_mean":               round(float(np.mean(dapta[:, 0])), 4),
            "gddqn_ciu_mean":               round(float(np.mean(gddqn[:, 0])), 4),
            "cohens_d_dapta_vs_gddqn":      round(d_rq4, 3),
            "effect_size":                  effect_size_label(d_rq4),
            "wilcoxon_p":                   round(float(p_rq4), 4),
            "significant":                  bool(p_rq4 < 0.05),
            "personalisation_beneficial":   bool(
                np.mean(dapta[:, 0]) > np.mean(gddqn[:, 0]) and p_rq4 < 0.05
            ),
        }

    # RQ3 — structured vs conversation CIU gain correlation
    if dapta is not None:
        structured_gain   = np.mean(
            [dapta[:, t * _N_METRICS_PER_TASK + 0]
             for t in _STRUCTURED_TASK_INDICES],
            axis=0,
        )
        conversation_gain = dapta[:, _CONVERSATION_TASK_INDEX * _N_METRICS_PER_TASK + 0]

        if len(structured_gain) >= 5:
            r, p = scipy_stats.spearmanr(structured_gain, conversation_gain)
            out["rq3"] = {
                "spearman_r":              round(float(r), 3),
                "p_value":                 round(float(p), 4),
                "significant":             bool(p < 0.05),
                "rq3_answered_positively": bool(p < 0.05 and r > 0.2),
            }

    return out


def build_csv_row(summary: dict) -> dict:
    """Flatten one noise-level summary into a single CSV row."""
    row = {
        "noise_label": summary["noise_label"],
        "noise_std":   summary["noise_std"],
        "n_patients":  summary["n_patients"],
    }
    rq2 = summary.get("rq2", {})
    row["rq2_dapta_ciu"]       = rq2.get("dapta_ciu_mean",          "")
    row["rq2_rbde_ciu"]        = rq2.get("rbde_ciu_mean",           "")
    row["rq2_cohens_d"]        = rq2.get("cohens_d_dapta_vs_rbde",  "")
    row["rq2_effect"]          = rq2.get("effect_size",             "")
    row["rq2_p"]               = rq2.get("wilcoxon_p",              "")
    row["rq2_significant"]     = rq2.get("significant",             "")
    row["rq2_n_clin"]          = rq2.get("n_metrics_clin",          "")
    row["rq2_answered"]        = rq2.get("rq2_answered_positively", "")
    row["rq2_beats_rbde"]      = rq2.get("dapta_beats_rbde",        "")

    rq4 = summary.get("rq4", {})
    row["rq4_dapta_ciu"]       = rq4.get("dapta_ciu_mean",              "")
    row["rq4_gddqn_ciu"]       = rq4.get("gddqn_ciu_mean",             "")
    row["rq4_cohens_d"]        = rq4.get("cohens_d_dapta_vs_gddqn",    "")
    row["rq4_effect"]          = rq4.get("effect_size",                "")
    row["rq4_p"]               = rq4.get("wilcoxon_p",                 "")
    row["rq4_significant"]     = rq4.get("significant",                "")
    row["rq4_pers_beneficial"] = rq4.get("personalisation_beneficial", "")

    rq3 = summary.get("rq3", {})
    row["rq3_spearman_r"]  = rq3.get("spearman_r",              "")
    row["rq3_p"]           = rq3.get("p_value",                 "")
    row["rq3_significant"] = rq3.get("significant",             "")
    row["rq3_answered"]    = rq3.get("rq3_answered_positively", "")

    return row


def print_report(summaries: List[dict], output_dir: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("DAPTA SENSITIVITY ANALYSIS — TRANSITION MODEL NOISE")
    lines.append("=" * 72)
    lines.append(
        "Conclusion stability: do RQ2/RQ3/RQ4 conclusions hold across "
        "noise levels?"
    )
    lines.append("")

    header = f"  {'Noise':8} {'std':7} | {'RQ2 d':7} {'RQ2 sig':8} {'RQ2 ans':8} | "
    header += f"{'RQ4 d':7} {'RQ4 sig':8} {'Pers?':6} | "
    header += f"{'RQ3 r':7} {'RQ3 sig':8}"
    lines.append(header)
    lines.append("  " + "-" * 68)

    rq2_consistent = True
    rq4_consistent = True
    rq3_consistent = True
    prev_rq2_ans   = None
    prev_rq4_pers  = None
    prev_rq3_ans   = None

    for s in summaries:
        rq2 = s.get("rq2", {})
        rq4 = s.get("rq4", {})
        rq3 = s.get("rq3", {})

        rq2_ans  = rq2.get("rq2_answered_positively", "N/A")
        rq4_pers = rq4.get("personalisation_beneficial", "N/A")
        rq3_ans  = rq3.get("rq3_answered_positively", "N/A")

        if prev_rq2_ans  is not None and rq2_ans  != prev_rq2_ans:  rq2_consistent = False
        if prev_rq4_pers is not None and rq4_pers != prev_rq4_pers: rq4_consistent = False
        if prev_rq3_ans  is not None and rq3_ans  != prev_rq3_ans:  rq3_consistent = False

        prev_rq2_ans  = rq2_ans
        prev_rq4_pers = rq4_pers
        prev_rq3_ans  = rq3_ans

        row = (
            f"  {s['noise_label']:8} {s['noise_std']:<7.3f} | "
            f"{rq2.get('cohens_d_dapta_vs_rbde', 'N/A'):>7}  "
            f"{'yes' if rq2.get('significant') else 'no':>7}  "
            f"{'yes' if rq2_ans else 'no':>7}  | "
            f"{rq4.get('cohens_d_dapta_vs_gddqn', 'N/A'):>7}  "
            f"{'yes' if rq4.get('significant') else 'no':>7}  "
            f"{'yes' if rq4_pers else 'no':>5}  | "
            f"{rq3.get('spearman_r', 'N/A'):>7}  "
            f"{'yes' if rq3.get('significant') else 'no':>7}"
        )
        lines.append(row)

    lines.append("")
    lines.append("Consistency across noise levels:")
    lines.append(f"  RQ2 (DAPTA beats RBDE):          {'STABLE ✓' if rq2_consistent else 'UNSTABLE ✗'}")
    lines.append(f"  RQ4 (personalisation beneficial): {'STABLE ✓' if rq4_consistent else 'UNSTABLE ✗'}")
    lines.append(f"  RQ3 (cross-task transfer):        {'STABLE ✓' if rq3_consistent else 'UNSTABLE ✗'}")
    lines.append("")

    all_stable = rq2_consistent and rq4_consistent and rq3_consistent
    if all_stable:
        lines.append(
            "OVERALL: Conclusions are robust to transition model noise assumptions.\n"
            "This supports external validity — results are not an artefact of a\n"
            "specific noise parameter choice."
        )
    else:
        lines.append(
            "OVERALL: Some conclusions are noise-sensitive. Report which RQs are\n"
            "affected and discuss as a limitation. Stable conclusions can still\n"
            "be reported with confidence."
        )

    lines.append("")
    lines.append(f"Full results: {output_dir}/sensitivity_results.json")
    lines.append(f"CSV table:    {output_dir}/sensitivity_summary.csv")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report)

    with open(output_dir / "sensitivity_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    logger.info(f"Report saved to {output_dir}/sensitivity_report.txt")


# 
# Main
# 

def main() -> None:
    output_dir = Path("outputs/sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Sensitivity Analysis — Transition Model Noise")
    logger.info("=" * 60)

    #  Load shared inputs 
    pes_dir = Path("outputs/pes")
    rl_dir  = Path("outputs/rl")

    pes_data        = np.load(str(pes_dir / "env_initial_states.npz"), allow_pickle=True)
    all_states      = pes_data["initial_states"]
    cluster_labels  = pes_data["cluster_labels"]
    split_labels    = pes_data["split_labels"]

    test_mask       = split_labels == "test"
    test_states     = all_states[test_mask]
    test_clusters   = cluster_labels[test_mask]

    n_test = int(test_mask.sum())
    logger.info(f"Test patients: {n_test}")

    if n_test == 0:
        logger.error("No test patients found.")
        return

    # Load transition model
    transition_model = TransitionModel(
        hidden_sizes    = (128, 64),
        dropout         = 0.2,
        checkpoint_path = str(pes_dir / "transition_model.pt"),
    )
    transition_model.load()

    # Load clusterer and predict test clusters
    with open(pes_dir / "clusterer.pkl", "rb") as f:
        clusterer = pickle.load(f)

    # We need PatientProfile objects for clusterer.predict
    import json
    with open("outputs/dae/patient_profiles.json") as f:
        all_profiles_data = json.load(f)

    session_ids     = pes_data["session_ids"]
    test_session_ids = session_ids[test_mask]
    session_to_profile = {p["session_id"]: p for p in all_profiles_data}
    test_profiles_data = [
        session_to_profile.get(str(sid), {"aphasia_subtype": "Other", "wab_aq": None})
        for sid in test_session_ids
    ]
    test_profiles_obj = [
        PatientProfile(
            participant_id  = p.get("participant_id", str(sid)),
            aphasia_subtype = p.get("aphasia_subtype", "Other"),
            wab_aq          = p.get("wab_aq", 50.0),
        )
        for p, sid in zip(test_profiles_data, test_session_ids)
    ]
    test_raw_subtypes  = [p.get("aphasia_subtype", "Other") for p in test_profiles_data]
    predicted_clusters = clusterer.predict(
        test_profiles_obj,
        state_vectors = test_states,
        raw_subtypes  = test_raw_subtypes,
    )
    predicted_clusters = np.where(predicted_clusters < 0, 0, predicted_clusters)

    # Load trained agents
    n_clusters = int(cluster_labels[cluster_labels >= 0].max()) + 1

    dapta_cluster_agents: Dict[int, DDQNAgent] = {}
    for c in range(n_clusters):
        ckpt = str(rl_dir / f"ddqn_cluster_{c}.pt")
        if Path(ckpt).exists():
            dapta_cluster_agents[c] = load_ddqn(ckpt, use_gru=True)
            logger.info(f"  DAPTA cluster {c} loaded.")

    g_ddqn = None
    ckpt_g = str(rl_dir / "ddqn_generalised.pt")
    if Path(ckpt_g).exists():
        g_ddqn = load_ddqn(ckpt_g, use_gru=True)
        logger.info("  G-DDQN loaded.")

    #  Run across noise levels 
    all_summaries = []
    all_raw       = {}

    for noise_label, noise_std in NOISE_LEVELS.items():
        logger.info(f"\nNoise level: {noise_label} (std={noise_std})")

        agent_results = evaluate_at_noise(
            noise_std            = noise_std,
            test_states          = test_states,
            transition_model     = transition_model,
            dapta_cluster_agents = dapta_cluster_agents,
            g_ddqn               = g_ddqn,
            predicted_clusters   = predicted_clusters,
            n_clusters           = n_clusters,
        )

        summary = summarise_noise_level(noise_label, noise_std, agent_results)
        all_summaries.append(summary)
        all_raw[noise_label] = {
            k: v.tolist() for k, v in agent_results.items()
        }

        logger.info(
            f"  RQ2: d={summary.get('rq2', {}).get('cohens_d_dapta_vs_rbde', 'N/A')}  "
            f"answered={summary.get('rq2', {}).get('rq2_answered_positively', 'N/A')}"
        )
        logger.info(
            f"  RQ4: beneficial="
            f"{summary.get('rq4', {}).get('personalisation_beneficial', 'N/A')}"
        )
        logger.info(
            f"  RQ3: r={summary.get('rq3', {}).get('spearman_r', 'N/A')}  "
            f"answered={summary.get('rq3', {}).get('rq3_answered_positively', 'N/A')}"
        )

    #  Save outputs 
    with open(output_dir / "sensitivity_results.json", "w") as f:
        json.dump({"summaries": all_summaries}, f, indent=2)

    csv_rows = [build_csv_row(s) for s in all_summaries]
    pd.DataFrame(csv_rows).to_csv(
        str(output_dir / "sensitivity_summary.csv"), index=False
    )

    print_report(all_summaries, output_dir)


if __name__ == "__main__":
    main()