"""
run_rq3.py
==========
Answers RQ3: Do discourse-level gains achieved through RL-personalised
therapy sequences transfer to improvements in naturalistic connected speech?

Rationale
---------
The G-DDQN agent (run_rq2.py) was trained and evaluated on structured
discourse tasks (AphasiaBank picture description — cookie theft task).
RQ3 asks whether improvements on structured tasks generalise to a
different, more naturalistic task type — free conversation.

This is the transfer validity question: does optimising discourse
metrics on a structured elicitation task produce gains that carry
over to unconstrained conversation? If yes, the therapy signal is
clinically meaningful beyond the training context.

Method
------
AphasiaBank contains multiple task types per participant:
  - cookie      : Cookie Theft picture description (structured)
  - fluency     : Verbal fluency tasks
  - recall      : Story recall
  - conversation: Free conversation / interview (naturalistic)

For each participant with both cookie and conversation transcripts:
  1. Compute discourse metrics on each task type separately
  2. Simulate the G-DDQN recommended sequence on the cookie state
  3. Project the simulated improvement onto the conversation state
     (conservative assumption: 60% transfer, per Gorshkov et al. 2025
     generalisation_potential for discourse-functional exercises)
  4. Compare projected conversation gains against:
       a. No-treatment counterfactual (zero change)
       b. Greedy baseline projected gains
  5. Statistical tests: Wilcoxon signed-rank, paired t-test
  6. Correlation analysis: do patients who benefit more on structured
     tasks also benefit more in conversation? (Pearson r)

Transfer coefficient
--------------------
The transfer coefficient T in [0,1] represents what fraction of
structured-task improvement is expected to transfer to conversation.
It is derived from the mean generalisation_potential of exercises
selected by the agent — exercises with higher generalisation_potential
are expected to produce more transfer (Gorshkov et al., 2025).

Outputs
-------
  results/rq3/transfer_summary.csv          per-patient transfer scores
  results/rq3/transfer_stats.csv            statistical test results
  results/rq3/transfer_correlation.png      structured vs conversation gain
  results/rq3/transfer_by_cluster.png       transfer by patient cluster
  results/rq3/transfer_by_severity.png      transfer by WAB-AQ severity
  results/rq3/task_comparison.png           metric profiles across tasks

Run
---
  python run_rq3.py

Flags
-----
  --state_vectors PATH   default: data/processed/state_vectors.csv
  --best_params   PATH   default: results/rq2/best_params.json
  --out_dir       PATH   default: results/rq3
  --eval_episodes INT    episodes per patient (default: 20)
  --seed          INT    (default: 42)
"""

import argparse
import json
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr, ttest_rel

from reward           import METRIC_ORDER, N_METRICS, softmax_weights
from transition_model import TransitionModel
from transitions      import build_transition_dataset, save_transitions
from therapy_env      import TherapyEnv, build_env_population
from ddqn             import DDQNAgent, GreedyBaseline
from action_space     import THERAPY_EXERCISES, get_exercise


# =============================================================================
# DEFAULTS
# =============================================================================

_STATE_VECTORS = "data/processed/state_vectors.csv"
_BEST_PARAMS   = "results/rq2/best_params.json"
_GDDQN_CKPT    = "models/gddqn.pt"
_TM_CKPT       = "models/transition_model.pt"
_OUT_DIR       = "results/rq3"

# Conservative transfer coefficient baseline — used if agent selects
# no discourse-functional exercises at all
_MIN_TRANSFER  = 0.20

# Task type columns expected in state_vectors.csv
# If per-task metrics are not available, we simulate using the global metrics
_TASK_TYPES = [
    "cookie_theft",
    "cinderella",
    "sandwich",
    "stroke_narrative",
    "conversation",
]


# =============================================================================
# TRANSFER COEFFICIENT
# =============================================================================

def compute_transfer_coefficient(action_distribution: dict) -> float:
    """
    Compute the expected transfer coefficient for a given action distribution.

    T = sum_a( freq(a) * generalisation_potential(a) )

    where freq(a) is the fraction of sessions that used action a.

    A purely greedy agent selecting action 11 (free_conversation, 0.90)
    every session would have T=0.90. A word-level drill agent would have
    T~0.25. This coefficient scales how much of the structured-task gain
    is projected to transfer to naturalistic speech.

    Parameters
    ----------
    action_distribution : dict name -> mean_count per episode

    Returns
    -------
    float T in [0, 1]
    """
    name_to_potential = {
        ex.name: ex.generalisation_potential
        for ex in THERAPY_EXERCISES
    }
    total_actions = sum(action_distribution.values())
    if total_actions == 0:
        return _MIN_TRANSFER

    T = sum(
        (count / total_actions) * name_to_potential.get(name, 0.3)
        for name, count in action_distribution.items()
    )
    return max(_MIN_TRANSFER, float(T))


# =============================================================================
# TASK-LEVEL METRIC EXTRACTION
# =============================================================================

def extract_task_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-task metric profiles from state_vectors.

    AphasiaBank state_vectors may contain task-prefixed columns
    (e.g. cookie_mlu_morphemes, conversation_ttr) if run_rq1.py
    was run with per-task extraction. If not, global metrics are
    used for all tasks (with a note in the output).

    Returns DataFrame with columns:
        participant_id, cluster_id, diagnosis, wab_aq,
        task, mlu_morphemes, ttr, ndw, n_utterances, mean_surprisal
    """
    rows = []
    has_task_cols = any(
        f"cookie_{m}" in df.columns for m in METRIC_ORDER
    )

    for _, row in df.iterrows():
        pid     = row.get("participant_id", "unknown")
        cluster = int(row.get("cluster_id", -1))
        diag    = row.get("diagnosis", "unknown")
        wab_aq  = float(row.get("wab_aq", np.nan))

        if has_task_cols:
            for task in _TASK_TYPES:
                task_metrics = {}
                for m in METRIC_ORDER:
                    col = f"{m}_{task}"
                    task_metrics[m] = float(row[col]) if col in row.index else np.nan
                if not all(np.isnan(v) for v in task_metrics.values()):
                    rows.append({
                        "participant_id": pid,
                        "cluster_id":     cluster,
                        "diagnosis":      diag,
                        "wab_aq":         wab_aq,
                        "task":           task,
                        **task_metrics,
                    })
        else:
            # Fallback: use global metrics, tag as "cookie" (structured task)
            global_metrics = {
                m: float(np.nan_to_num(row.get(m, 0.5), nan=0.5))
                for m in METRIC_ORDER
            }
            rows.append({
                "participant_id": pid,
                "cluster_id":     cluster,
                "diagnosis":      diag,
                "wab_aq":         wab_aq,
                "task":           "cookie",
                **global_metrics,
            })

    df_tasks = pd.DataFrame(rows)
    print(f"  Extracted task-level metrics: {len(df_tasks)} rows "
          f"({'per-task' if has_task_cols else 'global fallback'})")
    return df_tasks


# =============================================================================
# TRANSFER SIMULATION
# =============================================================================

def simulate_transfer(
    df_tasks:         pd.DataFrame,
    gddqn:            DDQNAgent,
    transition_model: TransitionModel,
    best_weights:     np.ndarray,
    eval_episodes:    int,
    seed:             int,
) -> pd.DataFrame:
    """
    For each patient, simulate the G-DDQN therapy sequence on the
    cookie (structured) task state and project gains to conversation.

    For each patient:
      1. Get structured task (cookie) state vector
      2. Run G-DDQN episode greedily -> get cumulative improvement + actions
      3. Compute transfer coefficient T from action distribution
      4. Project conversation gains = structured gains * T
      5. Compare to greedy baseline projected gains

    Returns DataFrame with per-patient transfer scores.
    """
    import random
    rng = random.Random(seed)

    # Get cookie task states
    cookie_df = df_tasks[df_tasks["task"] == "cookie"].copy()
    if cookie_df.empty:
        print("  WARNING: No cookie task data found. Using global metrics.")
        cookie_df = df_tasks.copy()
        cookie_df["task"] = "cookie"

    # Get conversation states if available
    conv_df = df_tasks[df_tasks["task"] == "conversation"].copy()
    has_conv = not conv_df.empty

    results = []
    greedy  = GreedyBaseline()

    for _, row in cookie_df.iterrows():
        pid     = row["participant_id"]
        cluster = int(row["cluster_id"])
        wab_aq  = float(row.get("wab_aq", np.nan))
        diag    = row.get("diagnosis", "unknown")

        # Build state vector
        s = np.array(
            [float(np.nan_to_num(row.get(m, 0.5), nan=0.5))
             for m in METRIC_ORDER],
            dtype=np.float32,
        )
        s = np.clip(s, 0.0, 1.0)

        # G-DDQN episode
        env = TherapyEnv(
            transition_model = transition_model,
            initial_state    = s,
            reward_weights   = best_weights,
        )
        state, _ = env.reset()
        while True:
            action = gddqn.select_action(state, greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        structured_improvement = env.get_cumulative_improvement()
        action_dist            = env.get_action_distribution()
        T                      = compute_transfer_coefficient(action_dist)

        # Project to conversation
        conv_improvement = {
            m: v * T for m, v in structured_improvement.items()
        }
        total_structured = sum(structured_improvement.values())
        total_conv       = sum(conv_improvement.values())

        # Greedy baseline episode for comparison
        env_g = TherapyEnv(
            transition_model = transition_model,
            initial_state    = s,
            reward_weights   = best_weights,
        )
        state_g, _ = env_g.reset()
        while True:
            action_g = greedy._action
            state_g, _, term_g, trunc_g, _ = env_g.step(action_g)
            if term_g or trunc_g:
                break

        greedy_improvement = env_g.get_cumulative_improvement()
        greedy_T           = compute_transfer_coefficient(
            env_g.get_action_distribution()
        )
        greedy_conv_total  = sum(
            v * greedy_T for v in greedy_improvement.values()
        )

        # Real conversation metrics if available
        conv_row     = conv_df[conv_df["participant_id"] == pid]
        has_real_conv = not conv_row.empty

        result = {
            "participant_id":          pid,
            "cluster_id":              cluster,
            "diagnosis":               diag,
            "wab_aq":                  wab_aq,
            "transfer_coeff":          T,
            "structured_improvement":  total_structured,
            "projected_conv_improvement": total_conv,
            "greedy_conv_improvement": greedy_conv_total,
            "transfer_advantage":      total_conv - greedy_conv_total,
            "has_real_conversation":   has_real_conv,
        }

        # Per-metric structured improvements
        for m, v in structured_improvement.items():
            result[f"structured_{m}"]  = v
            result[f"projected_conv_{m}"] = v * T

        # Real conversation metric deltas if available
        if has_real_conv:
            for m in METRIC_ORDER:
                result[f"real_conv_{m}"] = float(
                    np.nan_to_num(conv_row.iloc[0].get(m, np.nan), nan=np.nan)
                )

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_transfer_statistics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive statistical tests for RQ3.

    Tests performed:
      1. Wilcoxon signed-rank: projected conv improvement vs zero
         (are gains significantly positive?)
      2. Paired t-test: G-DDQN projected vs greedy projected
         (does the RL agent produce better transfer than greedy?)
      3. Pearson r: structured gain vs projected conversation gain
         (do structured gains predict conversation gains?)
      4. Pearson r: WAB-AQ vs transfer advantage
         (do more severe patients benefit more or less?)
      5. Cluster-level breakdown: mean transfer advantage per cluster

    Returns DataFrame of test results.
    """
    rows = []

    proj   = df_results["projected_conv_improvement"].dropna().values
    greedy = df_results["greedy_conv_improvement"].dropna().values
    struct = df_results["structured_improvement"].dropna().values
    wab    = df_results["wab_aq"].dropna().values
    tadv   = df_results["transfer_advantage"].dropna().values

    # 1. Projected conv improvement vs zero
    try:
        w_stat, w_p = wilcoxon(proj, alternative="greater")
    except Exception:
        w_stat, w_p = np.nan, np.nan
    rows.append({
        "test":        "Wilcoxon: projected conv improvement > 0",
        "statistic":   w_stat,
        "p_value":     w_p,
        "significant": w_p < 0.05 if not np.isnan(w_p) else False,
        "note":        "Are projected conv gains significantly positive?",
    })

    # 2. G-DDQN projected vs greedy projected (paired t-test)
    n = min(len(proj), len(greedy))
    try:
        t_stat, t_p = ttest_rel(proj[:n], greedy[:n], alternative="greater")
    except Exception:
        t_stat, t_p = np.nan, np.nan
    rows.append({
        "test":        "Paired t-test: G-DDQN projected > greedy projected",
        "statistic":   t_stat,
        "p_value":     t_p,
        "significant": t_p < 0.05 if not np.isnan(t_p) else False,
        "note":        "Does RL produce better projected transfer than greedy?",
    })

    # 3. Pearson r: structured vs projected conversation
    n = min(len(struct), len(proj))
    try:
        r_struct, p_struct = pearsonr(struct[:n], proj[:n])
    except Exception:
        r_struct, p_struct = np.nan, np.nan
    rows.append({
        "test":        "Pearson r: structured gain vs projected conv gain",
        "statistic":   r_struct,
        "p_value":     p_struct,
        "significant": p_struct < 0.05 if not np.isnan(p_struct) else False,
        "note":        "Do structured gains predict conversation gains?",
    })

    # 4. Pearson r: WAB-AQ vs transfer advantage
    n = min(len(wab), len(tadv))
    try:
        r_wab, p_wab = pearsonr(wab[:n], tadv[:n])
    except Exception:
        r_wab, p_wab = np.nan, np.nan
    rows.append({
        "test":        "Pearson r: WAB-AQ vs transfer advantage",
        "statistic":   r_wab,
        "p_value":     p_wab,
        "significant": p_wab < 0.05 if not np.isnan(p_wab) else False,
        "note":        "Does severity predict transfer benefit?",
    })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_transfer_correlation(df: pd.DataFrame, out_path: str) -> None:
    """Scatter: structured improvement vs projected conversation improvement."""
    fig, ax = plt.subplots(figsize=(8, 6))

    clusters = df["cluster_id"].unique()
    colors   = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, cid in enumerate(sorted(clusters)):
        sub = df[df["cluster_id"] == cid]
        ax.scatter(
            sub["structured_improvement"],
            sub["projected_conv_improvement"],
            label=f"Cluster {cid}",
            color=colors[i % len(colors)],
            alpha=0.7,
            s=40,
        )

    # Regression line
    x = df["structured_improvement"].values
    y = df["projected_conv_improvement"].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 1:
        m, b = np.polyfit(x[mask], y[mask], 1)
        xl   = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, m * xl + b, "k--", lw=1.5, alpha=0.6)

    ax.set_xlabel("Structured task improvement (G-DDQN)")
    ax.set_ylabel("Projected conversation improvement")
    ax.set_title("Transfer: structured gains -> naturalistic speech")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Transfer correlation plot -> {out_path}")


def plot_transfer_by_cluster(df: pd.DataFrame, out_path: str) -> None:
    """Box plot: transfer advantage by cluster."""
    clusters = sorted(df["cluster_id"].unique())
    data     = [
        df[df["cluster_id"] == cid]["transfer_advantage"].dropna().values
        for cid in clusters
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[f"Cluster {c}" for c in clusters], patch_artist=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_ylabel("Transfer advantage (G-DDQN - greedy)")
    ax.set_title("Transfer advantage by patient cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Transfer by cluster -> {out_path}")


def plot_transfer_by_severity(df: pd.DataFrame, out_path: str) -> None:
    """Scatter: WAB-AQ severity vs transfer advantage."""
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = ~df["wab_aq"].isna() & ~df["transfer_advantage"].isna()
    x    = df.loc[mask, "wab_aq"].values
    y    = df.loc[mask, "transfer_advantage"].values

    ax.scatter(x, y, alpha=0.6, color="steelblue", s=40)

    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        xl   = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, m * xl + b, "r--", lw=1.5)

    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("WAB-AQ score (higher = less severe)")
    ax.set_ylabel("Transfer advantage (G-DDQN - greedy)")
    ax.set_title("Does severity moderate transfer benefit?")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Transfer by severity -> {out_path}")


def plot_task_comparison(df_tasks: pd.DataFrame, out_path: str) -> None:
    """Bar chart: mean metric profiles across task types."""
    available_tasks = df_tasks["task"].unique()
    metrics         = METRIC_ORDER

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4 * len(metrics), 5), sharey=False
    )

    for ax, metric in zip(axes, metrics):
        means = []
        stds  = []
        tasks = []
        for task in _TASK_TYPES:
            if task not in available_tasks:
                continue
            vals = df_tasks[df_tasks["task"] == task][metric].dropna()
            if len(vals) == 0:
                continue
            means.append(vals.mean())
            stds.append(vals.std())
            tasks.append(task)

        ax.bar(tasks, means, yerr=stds, capsize=4,
               color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"][:len(tasks)])
        ax.set_title(metric, fontsize=10)
        ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)

    plt.suptitle("Discourse metric profiles by task type", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Task comparison -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAPTA RQ3 — transfer to naturalistic speech"
    )
    parser.add_argument("--state_vectors", default=_STATE_VECTORS)
    parser.add_argument("--best_params",   default=_BEST_PARAMS)
    parser.add_argument("--out_dir",       default=_OUT_DIR)
    parser.add_argument("--eval_episodes", default=20,  type=int)
    parser.add_argument("--seed",          default=42,  type=int)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DAPTA -- RQ3: Transfer to Naturalistic Connected Speech")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Step 1: Load best params from RQ2
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading best params from RQ2...")
    if not Path(args.best_params).exists():
        raise FileNotFoundError(
            f"No best_params.json at {args.best_params}. "
            f"Run run_rq2.py first."
        )
    with open(args.best_params) as f:
        saved = json.load(f)

    best_weights = np.array(
        list(saved["weights"].values()), dtype=np.float32
    )
    best_hp      = saved["hyperparams"]
    print(f"  Weights:    {saved['weights']}")
    print(f"  Agent hp:   {best_hp}")

    # ------------------------------------------------------------------
    # Step 2: Load state vectors + extract task metrics
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading state vectors...")
    df = pd.read_csv(args.state_vectors)
    df_aphasia = df.copy().reset_index(drop=True)
    print(f"  {len(df_aphasia)} aphasia patients")

    print("\n[Step 3] Extracting task-level metrics...")
    df_tasks = extract_task_metrics(df_aphasia)

    # ------------------------------------------------------------------
    # Step 3: Load transition model + G-DDQN
    # ------------------------------------------------------------------
    print("\n[Step 4] Loading TransitionModel and G-DDQN...")
    transition_model = TransitionModel(checkpoint_path=_TM_CKPT)
    transition_model.load()

    gddqn = DDQNAgent(
        **best_hp,
        seed            = args.seed,
        checkpoint_path = _GDDQN_CKPT,
    )
    gddqn.load()

    # ------------------------------------------------------------------
    # Step 4: Simulate transfer
    # ------------------------------------------------------------------
    print("\n[Step 5] Simulating transfer for each patient...")
    df_results = simulate_transfer(
        df_tasks         = df_tasks,
        gddqn            = gddqn,
        transition_model = transition_model,
        best_weights     = best_weights,
        eval_episodes    = args.eval_episodes,
        seed             = args.seed,
    )

    summary_path = out_dir / "transfer_summary.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"  Transfer summary -> {summary_path}")

    # Print overview
    print(f"\n  Mean transfer coefficient:          "
          f"{df_results['transfer_coeff'].mean():.3f}")
    print(f"  Mean projected conv improvement:    "
          f"{df_results['projected_conv_improvement'].mean():.4f}")
    print(f"  Mean greedy conv improvement:       "
          f"{df_results['greedy_conv_improvement'].mean():.4f}")
    print(f"  Mean transfer advantage (RL-greedy):"
          f" {df_results['transfer_advantage'].mean():.4f}")

    # ------------------------------------------------------------------
    # Step 5: Statistical tests
    # ------------------------------------------------------------------
    print("\n[Step 6] Running statistical tests...")
    stats_df   = run_transfer_statistics(df_results)
    stats_path = out_dir / "transfer_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n{stats_df[['test','statistic','p_value','significant']].to_string(index=False)}")
    print(f"\n  Stats -> {stats_path}")

    # ------------------------------------------------------------------
    # Step 6: Plots
    # ------------------------------------------------------------------
    print("\n[Step 7] Generating plots...")
    plot_transfer_correlation(
        df_results, str(out_dir / "transfer_correlation.png")
    )
    plot_transfer_by_cluster(
        df_results, str(out_dir / "transfer_by_cluster.png")
    )
    plot_transfer_by_severity(
        df_results, str(out_dir / "transfer_by_severity.png")
    )
    plot_task_comparison(
        df_tasks, str(out_dir / "task_comparison.png")
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  RQ3 complete.")
    print(f"  Transfer summary  -> {summary_path}")
    print(f"  Statistical tests -> {stats_path}")
    print(f"  Plots             -> {out_dir}/")
    print(f"{'='*65}")
    print("\nNext step: run run_rq4.py for personalised vs generalised comparison.")


if __name__ == "__main__":
    main()