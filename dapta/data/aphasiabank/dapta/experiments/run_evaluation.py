"""
experiments/run_evaluation.py
------------------------------
Phase 3: Full evaluation and generalisation test.

Tests:
  - RQ3: Do discourse-level gains transfer to naturalised speech?
  - RQ4: Does patient-specific adaptation outperform G-DDQN?

Compares DAPTA (DDQN patient-specific) vs RBDE, RTS, G-DDQN
on held-out AphasiaBank spontaneous speech samples.

Statistical analysis:
  - Paired Wilcoxon signed-rank test (Bonferroni corrected)
  - Cohen's d effect size (clinically meaningful threshold: d ≥ 0.40)
  - 95% Bootstrap confidence intervals

References
----------
Upton et al. (2024). iTalkBetter (d = 0.42 benchmark). Aphasiology.
Gorshkov et al. (2025). Brain Sciences, 15(9), 1007.
Quique et al. (2024). JSLHR, 67(9), 3203–3228.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dapta.utils.config import Config
from dapta.utils.logger import get_logger
from dapta.utils.metrics_eval import (
    run_pairwise_wilcoxon,
    bootstrap_ci,
    format_evaluation_report,
    cohens_d_paired,
)
from dapta.prta.trainer import RuleBasedBaseline, RandomBaseline

logger = get_logger(__name__, log_file="logs/evaluation.log")

METRIC_NAMES = ["ciu_rate", "mc_score", "mlu_morphemes", "ttr", "syntactic_complexity"]
N_DISCOURSE_METRICS = len(METRIC_NAMES)


def run_agent_on_env(agent, env) -> np.ndarray:
    """Run one episode. Returns discourse improvement vector (6,)."""
    state, _ = env.reset()
    for _ in range(env.episode_horizon):
        if hasattr(agent, "select_action"):
            action = agent.select_action(state)
        else:
            action = int(agent.predict(state.reshape(1, -1))[0])
        state, _, done, _, _ = env.step(action)
        if done:
            break
    return env.get_cumulative_discourse_improvement()


def evaluate_all_agents(
    test_envs: List,
    dapta_agent,
    g_ddqn_agent,
    rbde: RuleBasedBaseline,
    rts: RandomBaseline,
) -> Dict[str, np.ndarray]:
    """
    Run all agents on all test environments.

    Returns
    -------
    {agent_name: improvements_matrix (N_patients, 6)}
    """
    results = {
        "DAPTA": [],
        "G_DDQN": [],
        "RBDE": [],
        "RTS": [],
    }

    for env in test_envs:
        results["DAPTA"].append(run_agent_on_env(dapta_agent, env))
        results["G_DDQN"].append(run_agent_on_env(g_ddqn_agent, env))
        results["RBDE"].append(rbde.run_episode(env)[1])
        results["RTS"].append(rts.run_episode(env)[1])

    return {k: np.stack(v) for k, v in results.items()}


def compute_generalisation_test(
    improvements: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """
    RQ3 generalisation test: Does DAPTA significantly outperform RBDE
    on naturalised discourse metrics?

    Null hypothesis: DAPTA produces no greater improvement than RBDE.

    Returns
    -------
    test_results: per-agent per-metric statistical results
    """
    all_results = {}

    for agent_name, agent_improvements in improvements.items():
        if agent_name == "DAPTA":
            continue

        # Compare DAPTA vs each baseline per metric
        comparison_key = f"DAPTA_vs_{agent_name}"
        metric_data = {}

        for i, metric in enumerate(METRIC_NAMES):
            dapta_vals = improvements["DAPTA"][:, i]
            baseline_vals = agent_improvements[:, i]

            # Treat it as paired: DAPTA improvement - baseline improvement
            diff = dapta_vals - baseline_vals
            metric_data[metric] = (
                np.zeros_like(diff),  # before = 0
                diff,                  # after = delta
            )

        all_results[comparison_key] = run_pairwise_wilcoxon(metric_data, alpha=alpha)

    return all_results


def compute_personalisation_test(
    dapta_improvements: np.ndarray,
    g_ddqn_improvements: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    RQ4: Does patient-specific DAPTA outperform generalised G-DDQN?
    Metric: within-patient variance reduction and mean CIU improvement.
    """
    dapta_ciu = dapta_improvements[:, 0]
    g_ddqn_ciu = g_ddqn_improvements[:, 0]

    d = cohens_d_paired(g_ddqn_ciu, dapta_ciu)
    lo, hi = bootstrap_ci(dapta_ciu - g_ddqn_ciu)

    dapta_var = float(np.var(dapta_ciu, ddof=1))
    g_ddqn_var = float(np.var(g_ddqn_ciu, ddof=1))

    return {
        "mean_dapta_ciu": float(np.mean(dapta_ciu)),
        "mean_gddqn_ciu": float(np.mean(g_ddqn_ciu)),
        "cohens_d_ciu": d,
        "bootstrap_ci_95": (lo, hi),
        "dapta_variance": dapta_var,
        "g_ddqn_variance": g_ddqn_var,
        "variance_reduction_pct": 100 * (g_ddqn_var - dapta_var) / max(g_ddqn_var, 1e-8),
    }


def save_results(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


def main() -> None:
    cfg = Config.load()
    logger.info("=" * 60)
    logger.info("DAPTA Phase 3: Generalisation & Personalisation Evaluation")
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # NOTE: In the actual experiment, load trained agents and test envs
    # from the outputs of run_rl.py and run_pes.py.
    # This script shows the full evaluation pipeline structure.
    # ---------------------------------------------------------------
    logger.info(
        "Load trained DAPTA, G-DDQN agents and test environments "
        "from Phase 2 outputs, then call evaluate_all_agents() → "
        "compute_generalisation_test() → compute_personalisation_test()."
    )
    logger.info("See individual function docstrings for usage.")


if __name__ == "__main__":
    main()
