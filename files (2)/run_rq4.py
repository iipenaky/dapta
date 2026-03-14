"""
run_rq4.py
==========
Answers RQ4: Does patient-specific RL adaptation produce significantly
better discourse outcomes than a generalised model trained across all
patients, and what patient factors influence this difference?

What this script does
---------------------
  1.  Loads best weights + hyperparameters from run_rq2.py (HPO output)
  2.  Loads the trained G-DDQN from run_rq2.py
  3.  Trains one P-DDQN agent per cluster using the same best weights
      and hyperparameters — the only difference is training data scope
      (cluster patients only vs all patients)
  4.  Evaluates G-DDQN and all P-DDQN agents on held-out patients
  5.  Statistical comparison: P-DDQN vs G-DDQN per cluster
  6.  Patient factor analysis — which patient characteristics predict
      who benefits most from personalisation:
        - WAB-AQ score (aphasia severity)
        - Cluster membership (language ability profile)
        - Diagnosis label (Anomic vs Broca vs other)
        - Initial metric levels (baseline mlu, ttr, ndw, n_utt, surprisal)
        - Interaction: severity × cluster
  7.  Feature importance via:
        - Pearson correlations (each factor vs personalisation benefit)
        - Multiple linear regression (all factors simultaneously)
        - Effect size (Cohen's d) for cluster-level comparisons

G-DDQN vs P-DDQN design
------------------------
G-DDQN: one agent, trained on all 456 aphasia patients pooled.
        Learns a general policy that works reasonably well on average.

P-DDQN: one agent per cluster (k=2 from cluster.py).
        Trained only on patients in that cluster.
        Learns a policy tuned to that cluster's language ability profile.

Both use identical reward weights (learned in run_rq2.py) and identical
network architecture / hyperparameters. The only difference is the
training population. This isolates the effect of personalisation.

Outputs
-------
  models/pddqn_cluster_{k}.pt              P-DDQN checkpoint per cluster
  results/rq4/pddqn_vs_gddqn.csv          main comparison table
  results/rq4/patient_factor_analysis.csv  per-patient factor scores
  results/rq4/regression_results.csv       MLR predicting personalisation benefit
  results/rq4/personalisation_benefit.png  G-DDQN vs P-DDQN per cluster
  results/rq4/factor_correlations.png      patient factors vs benefit
  results/rq4/regression_coefficients.png  MLR coefficient plot
  results/rq4/per_metric_comparison.png    per-metric gains comparison
  results/rq4/action_profiles.png          exercise selection by agent

Run
---
  python run_rq4.py

Flags
-----
  --state_vectors  PATH   default: data/processed/state_vectors.csv
  --best_params    PATH   default: results/rq2/best_params.json
  --out_dir        PATH   default: results/rq4
  --final_episodes INT    P-DDQN training episodes per cluster (default: 2000)
  --eval_episodes  INT    evaluation episodes per agent (default: 50)
  --train_split    FLOAT  fraction for training (default: 0.8)
  --seed           INT    (default: 42)
"""

import argparse
import json
from pathlib import Path

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats  import wilcoxon, pearsonr, ttest_rel
from scipy.stats  import f_oneway
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model    import LinearRegression
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import r2_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from reward           import METRIC_ORDER, N_METRICS, softmax_weights
from transition_model import TransitionModel
from transitions      import build_transition_dataset
from therapy_env      import TherapyEnv, build_env_population
from ddqn             import DDQNAgent
from action_space     import THERAPY_EXERCISES


# =============================================================================
# DEFAULTS
# =============================================================================

_STATE_VECTORS = "data/processed/state_vectors.csv"
_BEST_PARAMS   = "results/rq2/best_params.json"
_GDDQN_CKPT    = "models/gddqn.pt"
_TM_CKPT       = "models/transition_model.pt"
_OUT_DIR       = "results/rq4"


# =============================================================================
# DATA HELPERS
# =============================================================================

def load_and_split(
    state_vectors_csv: str,
    train_split:       float,
    seed:              int,
) -> tuple:
    """Stratified train/eval split by cluster. Returns df_train, df_eval, df_full."""
    df         = pd.read_csv(state_vectors_csv)
    df_aphasia = df[df["cluster_id"] >= 0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    train_rows, eval_rows = [], []

    for cid, group in df_aphasia.groupby("cluster_id"):
        idx   = group.index.tolist()
        rng.shuffle(idx)
        split = max(1, int(len(idx) * train_split))
        train_rows.extend(idx[:split])
        eval_rows.extend(idx[split:])

    return (
        df_aphasia.loc[train_rows].reset_index(drop=True),
        df_aphasia.loc[eval_rows ].reset_index(drop=True),
        df_aphasia,
    )


def df_to_envs(
    df:               pd.DataFrame,
    transition_model: TransitionModel,
    weights:          np.ndarray,
) -> list:
    """Build TherapyEnv list from a DataFrame of patients."""
    states = []
    for _, row in df.iterrows():
        s = row[METRIC_ORDER].values.astype(np.float32)
        s = np.nan_to_num(s, nan=0.5)
        s = np.clip(s, 0.0, 1.0)
        states.append(s)
    return build_env_population(
        initial_states   = states,
        transition_model = transition_model,
        reward_weights   = weights,
    )


# =============================================================================
# TRAIN P-DDQN PER CLUSTER
# =============================================================================

def train_pddqn_agents(
    df_train:         pd.DataFrame,
    df_eval:          pd.DataFrame,
    transition_model: TransitionModel,
    best_weights:     np.ndarray,
    best_hp:          dict,
    final_episodes:   int,
    eval_episodes:    int,
    seed:             int,
    out_dir:          Path,
) -> dict:
    """
    Train one P-DDQN per cluster.

    Each agent uses:
      - Same reward weights as G-DDQN (learned in run_rq2.py)
      - Same network architecture and hyperparameters (from HPO)
      - Different training population: only patients in that cluster

    Returns dict: cluster_id -> DDQNAgent
    """
    cluster_ids = sorted(df_train["cluster_id"].unique().tolist())
    agents      = {}

    for cid in cluster_ids:
        print(f"\n  Training P-DDQN for Cluster {cid}...")

        # Cluster-specific train/eval environments
        df_c_train = df_train[df_train["cluster_id"] == cid]
        df_c_eval  = df_eval[ df_eval[ "cluster_id"] == cid]

        # Fallback: if cluster has no eval patients, use train set
        if df_c_eval.empty:
            df_c_eval = df_c_train

        train_envs = df_to_envs(df_c_train, transition_model, best_weights)
        eval_envs  = df_to_envs(df_c_eval,  transition_model, best_weights)

        n_train = len(train_envs)
        print(f"  Cluster {cid}: {n_train} training patients, "
              f"{len(eval_envs)} eval patients")

        ckpt = str(out_dir.parent / f"models/pddqn_cluster_{cid}.pt")
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)

        agent = DDQNAgent(
            **best_hp,
            seed            = seed + cid,
            checkpoint_path = ckpt,
        )
        history = agent.train(
            envs          = train_envs,
            n_episodes    = final_episodes,
            eval_envs     = eval_envs,
            eval_every    = max(10, final_episodes // 20),
            eval_episodes = eval_episodes,
        )

        agents[int(cid)] = agent
        print(f"  P-DDQN Cluster {cid} trained. "
              f"Best eval: {agent._best_reward:.4f}")

    return agents


# =============================================================================
# EVALUATE ALL AGENTS PER PATIENT
# =============================================================================

def evaluate_all_agents(
    df_eval:          pd.DataFrame,
    gddqn:            DDQNAgent,
    pddqn_agents:     dict,
    transition_model: TransitionModel,
    best_weights:     np.ndarray,
    eval_episodes:    int,
    seed:             int,
) -> pd.DataFrame:
    """
    For each eval patient, run both G-DDQN and their cluster's P-DDQN.
    Record per-patient improvement scores for every agent.

    This patient-level comparison is what powers the factor analysis —
    we compute personalisation_benefit = P-DDQN score - G-DDQN score
    for each patient individually.

    Returns DataFrame with one row per patient.
    """
    import random
    rng = random.Random(seed)

    results = []

    for _, row in df_eval.iterrows():
        pid     = row.get("participant_id", "unknown")
        cluster = int(row["cluster_id"])
        diag    = str(row.get("diagnosis", "unknown")).lower().strip()
        wab_aq  = float(row.get("wab_aq", np.nan))

        s = np.array(
            [float(np.nan_to_num(row.get(m, 0.5), nan=0.5))
             for m in METRIC_ORDER],
            dtype=np.float32,
        )
        s = np.clip(s, 0.0, 1.0)

        def run_agent(agent, state):
            env = TherapyEnv(
                transition_model = transition_model,
                initial_state    = state.copy(),
                reward_weights   = best_weights,
            )
            obs, _ = env.reset()
            while True:
                a = agent.select_action(obs, greedy=True)
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            return (
                env.get_cumulative_improvement(),
                env.get_action_distribution(),
            )

        # G-DDQN
        g_improvement, g_actions = run_agent(gddqn, s)
        g_total = sum(g_improvement.values())

        # P-DDQN for this patient's cluster
        p_agent = pddqn_agents.get(cluster)
        if p_agent is not None:
            p_improvement, p_actions = run_agent(p_agent, s)
            p_total = sum(p_improvement.values())
        else:
            p_total      = g_total
            p_improvement = g_improvement
            p_actions    = g_actions

        personalisation_benefit = p_total - g_total

        result = {
            "participant_id":          pid,
            "cluster_id":              cluster,
            "diagnosis":               diag,
            "wab_aq":                  wab_aq,
            "gddqn_total":             g_total,
            "pddqn_total":             p_total,
            "personalisation_benefit": personalisation_benefit,
        }

        # Per-metric
        for m in METRIC_ORDER:
            result[f"gddqn_{m}"]    = g_improvement.get(m, 0.0)
            result[f"pddqn_{m}"]    = p_improvement.get(m, 0.0)
            result[f"initial_{m}"]  = float(s[METRIC_ORDER.index(m)])

        # Action profile counts
        for ex in THERAPY_EXERCISES:
            result[f"gddqn_action_{ex.name}"] = g_actions.get(ex.name, 0)
            result[f"pddqn_action_{ex.name}"] = p_actions.get(ex.name, 0)

        results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def statistical_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive statistical comparison: P-DDQN vs G-DDQN.

    Tests:
      1. Overall: Wilcoxon signed-rank (paired, P-DDQN > G-DDQN)
      2. Overall: Paired t-test
      3. Per cluster: Wilcoxon signed-rank
      4. Per cluster: Cohen's d effect size
      5. One-way ANOVA: personalisation benefit across clusters

    Returns DataFrame of results.
    """
    rows = []

    g = df["gddqn_total"].values
    p = df["pddqn_total"].values

    # 1. Overall Wilcoxon
    try:
        w_stat, w_p = wilcoxon(p, g, alternative="greater")
    except Exception:
        w_stat, w_p = np.nan, np.nan
    rows.append({
        "comparison": "Overall P-DDQN > G-DDQN",
        "test":       "Wilcoxon signed-rank",
        "n":          len(g),
        "statistic":  w_stat,
        "p_value":    w_p,
        "effect_size": float(np.mean(p - g) / (np.std(p - g) + 1e-8)),
        "significant": w_p < 0.05 if not np.isnan(w_p) else False,
    })

    # 2. Overall paired t-test
    try:
        t_stat, t_p = ttest_rel(p, g, alternative="greater")
    except Exception:
        t_stat, t_p = np.nan, np.nan
    rows.append({
        "comparison": "Overall P-DDQN > G-DDQN",
        "test":       "Paired t-test",
        "n":          len(g),
        "statistic":  t_stat,
        "p_value":    t_p,
        "effect_size": float(np.mean(p - g) / (np.std(p - g) + 1e-8)),
        "significant": t_p < 0.05 if not np.isnan(t_p) else False,
    })

    # 3+4. Per-cluster
    for cid, group in df.groupby("cluster_id"):
        gc = group["gddqn_total"].values
        pc = group["pddqn_total"].values
        n  = len(gc)
        diff = pc - gc

        try:
            w_s, w_p = wilcoxon(pc, gc, alternative="greater")
        except Exception:
            w_s, w_p = np.nan, np.nan

        # Cohen's d
        cohen_d = (
            float(np.mean(diff) / (np.std(diff) + 1e-8))
            if len(diff) > 1 else np.nan
        )

        rows.append({
            "comparison":  f"Cluster {cid}: P-DDQN > G-DDQN",
            "test":        "Wilcoxon signed-rank",
            "n":           n,
            "statistic":   w_s,
            "p_value":     w_p,
            "effect_size": cohen_d,
            "significant": w_p < 0.05 if not np.isnan(w_p) else False,
        })

    # 5. ANOVA: does personalisation benefit differ across clusters?
    cluster_benefits = [
        group["personalisation_benefit"].values
        for _, group in df.groupby("cluster_id")
    ]
    if len(cluster_benefits) > 1:
        try:
            f_stat, f_p = f_oneway(*cluster_benefits)
        except Exception:
            f_stat, f_p = np.nan, np.nan
        rows.append({
            "comparison":  "Personalisation benefit differs across clusters",
            "test":        "One-way ANOVA",
            "n":           len(df),
            "statistic":   f_stat,
            "p_value":     f_p,
            "effect_size": np.nan,
            "significant": f_p < 0.05 if not np.isnan(f_p) else False,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PATIENT FACTOR ANALYSIS
# =============================================================================

def patient_factor_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which patient characteristics predict personalisation benefit.

    Factors examined:
      - wab_aq              : aphasia severity (continuous)
      - cluster_id          : language ability profile (categorical)
      - diagnosis_*         : diagnosis type dummies
      - initial_*           : baseline metric levels
      - wab_aq × cluster_id : severity-by-cluster interaction

    For each factor:
      - Pearson r with personalisation_benefit (continuous factors)
      - Point-biserial r for binary/dummy factors
      - p-value and 95% CI

    Returns DataFrame of correlations.
    """
    benefit = df["personalisation_benefit"].values
    rows    = []

    # Continuous factors
    continuous_factors = ["wab_aq"] + [f"initial_{m}" for m in METRIC_ORDER]
    for factor in continuous_factors:
        if factor not in df.columns:
            continue
        vals = df[factor].values
        mask = ~np.isnan(vals) & ~np.isnan(benefit)
        if mask.sum() < 5:
            continue
        try:
            r, p = pearsonr(vals[mask], benefit[mask])
        except Exception:
            r, p = np.nan, np.nan
        rows.append({
            "factor":      factor,
            "type":        "continuous",
            "pearson_r":   r,
            "p_value":     p,
            "significant": p < 0.05 if not np.isnan(p) else False,
            "n":           int(mask.sum()),
        })

    # Cluster (binary dummy if k=2, otherwise ordinal)
    cluster_vals = df["cluster_id"].values.astype(float)
    mask = ~np.isnan(benefit)
    try:
        r, p = pearsonr(cluster_vals[mask], benefit[mask])
    except Exception:
        r, p = np.nan, np.nan
    rows.append({
        "factor":      "cluster_id",
        "type":        "categorical",
        "pearson_r":   r,
        "p_value":     p,
        "significant": p < 0.05 if not np.isnan(p) else False,
        "n":           int(mask.sum()),
    })

    # Diagnosis dummies
    diag_dummies = pd.get_dummies(df["diagnosis"], prefix="diag")
    for col in diag_dummies.columns:
        vals = diag_dummies[col].values.astype(float)
        mask = ~np.isnan(benefit)
        try:
            r, p = pearsonr(vals[mask], benefit[mask])
        except Exception:
            r, p = np.nan, np.nan
        rows.append({
            "factor":      col,
            "type":        "diagnosis_dummy",
            "pearson_r":   r,
            "p_value":     p,
            "significant": p < 0.05 if not np.isnan(p) else False,
            "n":           int(mask.sum()),
        })

    # Interaction: wab_aq × cluster_id
    if "wab_aq" in df.columns:
        interaction = (
            df["wab_aq"].fillna(df["wab_aq"].median()).values *
            cluster_vals
        )
        mask = ~np.isnan(benefit)
        try:
            r, p = pearsonr(interaction[mask], benefit[mask])
        except Exception:
            r, p = np.nan, np.nan
        rows.append({
            "factor":      "wab_aq × cluster_id",
            "type":        "interaction",
            "pearson_r":   r,
            "p_value":     p,
            "significant": p < 0.05 if not np.isnan(p) else False,
            "n":           int(mask.sum()),
        })

    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)


def multiple_linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple linear regression predicting personalisation_benefit.

    Predictors: wab_aq, cluster_id, initial metric levels,
                diagnosis dummies, interaction term.
    Standardised coefficients (beta weights) show relative importance.

    Returns DataFrame of coefficients.
    """
    if not _SKLEARN:
        print("  sklearn not available — skipping MLR.")
        return pd.DataFrame()

    benefit = df["personalisation_benefit"].values

    # Build feature matrix
    feature_cols = []
    X_parts      = []

    # WAB-AQ
    if "wab_aq" in df.columns:
        wab = df["wab_aq"].fillna(df["wab_aq"].median()).values.reshape(-1, 1)
        X_parts.append(wab)
        feature_cols.append("wab_aq")

    # Cluster
    cluster = df["cluster_id"].values.reshape(-1, 1).astype(float)
    X_parts.append(cluster)
    feature_cols.append("cluster_id")

    # Initial metrics
    for m in METRIC_ORDER:
        col = f"initial_{m}"
        if col in df.columns:
            vals = df[col].fillna(0.5).values.reshape(-1, 1)
            X_parts.append(vals)
            feature_cols.append(col)

    # Diagnosis dummies
    diag_dummies = pd.get_dummies(df["diagnosis"], prefix="diag")
    if not diag_dummies.empty:
        X_parts.append(diag_dummies.values.astype(float))
        feature_cols.extend(diag_dummies.columns.tolist())

    # Interaction: wab_aq × cluster
    if "wab_aq" in df.columns:
        interaction = (
            df["wab_aq"].fillna(df["wab_aq"].median()).values *
            df["cluster_id"].values
        ).reshape(-1, 1)
        X_parts.append(interaction)
        feature_cols.append("wab_aq_x_cluster")

    X = np.hstack(X_parts)

    # Remove rows with NaN in benefit
    mask = ~np.isnan(benefit)
    X    = X[mask]
    y    = benefit[mask]

    if len(y) < len(feature_cols) + 1:
        print("  Too few samples for MLR — skipping.")
        return pd.DataFrame()

    scaler  = StandardScaler()
    X_std   = scaler.fit_transform(X)

    reg     = LinearRegression()
    reg.fit(X_std, y)
    r2      = r2_score(y, reg.predict(X_std))

    rows = []
    for name, coef in zip(feature_cols, reg.coef_):
        rows.append({
            "predictor":         name,
            "std_coefficient":   float(coef),
            "abs_coefficient":   abs(float(coef)),
        })

    rows.append({
        "predictor":       "R_squared",
        "std_coefficient": r2,
        "abs_coefficient": r2,
    })

    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_personalisation_benefit(df: pd.DataFrame, out_path: str) -> None:
    """Box plot: G-DDQN vs P-DDQN total improvement per cluster."""
    clusters = sorted(df["cluster_id"].unique())
    n_rows   = 1
    n_cols   = len(clusters) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6))

    # Overall
    ax = axes[0]
    ax.boxplot(
        [df["gddqn_total"].values, df["pddqn_total"].values],
        labels=["G-DDQN", "P-DDQN"],
        patch_artist=True,
    )
    ax.set_title("Overall")
    ax.set_ylabel("Cumulative discourse improvement")

    # Per cluster
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for i, cid in enumerate(clusters):
        ax  = axes[i + 1]
        sub = df[df["cluster_id"] == cid]
        bp  = ax.boxplot(
            [sub["gddqn_total"].values, sub["pddqn_total"].values],
            labels=["G-DDQN", f"P-DDQN\nCluster {cid}"],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], ["#90CAF9", colors[i % len(colors)]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"Cluster {cid}")

    plt.suptitle("G-DDQN vs P-DDQN: discourse improvement", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Personalisation benefit -> {out_path}")


def plot_factor_correlations(
    factor_df: pd.DataFrame,
    out_path:  str,
) -> None:
    """Horizontal bar chart of Pearson r for each patient factor."""
    factor_df = factor_df.dropna(subset=["pearson_r"])
    factor_df = factor_df[factor_df["factor"] != "R_squared"]

    fig, ax = plt.subplots(figsize=(9, max(5, len(factor_df) * 0.4)))
    colors  = [
        "#E53935" if row["significant"] else "#90A4AE"
        for _, row in factor_df.iterrows()
    ]
    ax.barh(
        factor_df["factor"],
        factor_df["pearson_r"],
        color=colors,
        edgecolor="white",
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r with personalisation benefit")
    ax.set_title(
        "Patient factors predicting personalisation benefit\n"
        "(red = significant p<0.05)"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Factor correlations -> {out_path}")


def plot_regression_coefficients(
    reg_df:   pd.DataFrame,
    out_path: str,
) -> None:
    """Bar chart of standardised regression coefficients."""
    if reg_df.empty:
        return
    reg_df = reg_df[reg_df["predictor"] != "R_squared"].copy()
    reg_df = reg_df.sort_values("std_coefficient")

    fig, ax = plt.subplots(figsize=(9, max(5, len(reg_df) * 0.4)))
    colors  = [
        "#E53935" if v > 0 else "#1565C0"
        for v in reg_df["std_coefficient"]
    ]
    ax.barh(reg_df["predictor"], reg_df["std_coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised regression coefficient (beta)")
    ax.set_title("MLR: predictors of personalisation benefit")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Regression coefficients -> {out_path}")


def plot_per_metric_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart: per-metric improvement for G-DDQN vs P-DDQN."""
    metrics  = METRIC_ORDER
    x        = np.arange(len(metrics))
    width    = 0.35

    g_means = [df[f"gddqn_{m}"].mean() for m in metrics]
    p_means = [df[f"pddqn_{m}"].mean() for m in metrics]
    g_stds  = [df[f"gddqn_{m}"].std()  for m in metrics]
    p_stds  = [df[f"pddqn_{m}"].std()  for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, g_means, width, yerr=g_stds, label="G-DDQN",
           color="#90CAF9", capsize=4)
    ax.bar(x + width/2, p_means, width, yerr=p_stds, label="P-DDQN",
           color="#EF9A9A", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Mean improvement (normalised)")
    ax.set_title("Per-metric improvement: G-DDQN vs P-DDQN")
    ax.legend()
    ax.axhline(0, color="black", lw=0.8, ls="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Per-metric comparison -> {out_path}")


def plot_action_profiles(df: pd.DataFrame, out_path: str) -> None:
    """Compare exercise selection between G-DDQN and P-DDQN per cluster."""
    exercise_names = [ex.name for ex in THERAPY_EXERCISES]
    clusters       = sorted(df["cluster_id"].unique())
    n_cols         = len(clusters)
    fig, axes      = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, cid in zip(axes, clusters):
        sub  = df[df["cluster_id"] == cid]
        g_counts = [sub[f"gddqn_action_{n}"].mean() for n in exercise_names]
        p_counts = [sub[f"pddqn_action_{n}"].mean() for n in exercise_names]
        y        = np.arange(len(exercise_names))
        w        = 0.35
        ax.barh(y - w/2, g_counts, w, label="G-DDQN", color="#90CAF9")
        ax.barh(y + w/2, p_counts, w, label=f"P-DDQN C{cid}", color="#EF9A9A")
        ax.set_yticks(y)
        ax.set_yticklabels(exercise_names, fontsize=8)
        ax.set_xlabel("Mean selections per episode")
        ax.set_title(f"Cluster {cid} action profile")
        ax.legend()

    plt.suptitle("Exercise selection: G-DDQN vs P-DDQN per cluster", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Action profiles -> {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DAPTA RQ4 — P-DDQN vs G-DDQN + patient factor analysis"
    )
    parser.add_argument("--state_vectors",  default=_STATE_VECTORS)
    parser.add_argument("--best_params",    default=_BEST_PARAMS)
    parser.add_argument("--out_dir",        default=_OUT_DIR)
    parser.add_argument("--final_episodes", default=2000, type=int)
    parser.add_argument("--eval_episodes",  default=50,   type=int)
    parser.add_argument("--train_split",    default=0.8,  type=float)
    parser.add_argument("--seed",           default=42,   type=int)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DAPTA -- RQ4: Personalised vs Generalised RL")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Step 1: Load best params
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
    print(f"  Weights: {saved['weights']}")

    # ------------------------------------------------------------------
    # Step 2: Load data
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading and splitting data...")
    df_train, df_eval, df_full = load_and_split(
        args.state_vectors, args.train_split, args.seed
    )
    print(f"  Train: {len(df_train)} | Eval: {len(df_eval)}")

    # ------------------------------------------------------------------
    # Step 3: Load transition model + G-DDQN
    # ------------------------------------------------------------------
    print("\n[Step 3] Loading TransitionModel and G-DDQN...")
    transition_model = TransitionModel(checkpoint_path=_TM_CKPT)
    transition_model.load()

    gddqn = DDQNAgent(
        **best_hp,
        seed            = args.seed,
        checkpoint_path = _GDDQN_CKPT,
    )
    gddqn.load()

    # ------------------------------------------------------------------
    # Step 4: Train P-DDQN per cluster
    # ------------------------------------------------------------------
    print("\n[Step 4] Training P-DDQN agents per cluster...")
    pddqn_agents = train_pddqn_agents(
        df_train         = df_train,
        df_eval          = df_eval,
        transition_model = transition_model,
        best_weights     = best_weights,
        best_hp          = best_hp,
        final_episodes   = args.final_episodes,
        eval_episodes    = args.eval_episodes,
        seed             = args.seed,
        out_dir          = out_dir,
    )

    # ------------------------------------------------------------------
    # Step 5: Per-patient evaluation
    # ------------------------------------------------------------------
    print("\n[Step 5] Evaluating all agents on held-out patients...")
    df_results = evaluate_all_agents(
        df_eval          = df_eval,
        gddqn            = gddqn,
        pddqn_agents     = pddqn_agents,
        transition_model = transition_model,
        best_weights     = best_weights,
        eval_episodes    = args.eval_episodes,
        seed             = args.seed,
    )

    results_path = out_dir / "patient_factor_analysis.csv"
    df_results.to_csv(results_path, index=False)
    print(f"  Patient results -> {results_path}")

    # Print headline numbers
    print(f"\n  G-DDQN mean improvement : "
          f"{df_results['gddqn_total'].mean():.4f} "
          f"(±{df_results['gddqn_total'].std():.4f})")
    print(f"  P-DDQN mean improvement : "
          f"{df_results['pddqn_total'].mean():.4f} "
          f"(±{df_results['pddqn_total'].std():.4f})")
    print(f"  Mean personalisation benefit: "
          f"{df_results['personalisation_benefit'].mean():.4f}")

    # ------------------------------------------------------------------
    # Step 6: Statistical comparison
    # ------------------------------------------------------------------
    print("\n[Step 6] Statistical comparison...")
    stats_df   = statistical_comparison(df_results)
    stats_path = out_dir / "pddqn_vs_gddqn.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n{stats_df[['comparison','test','n','statistic','p_value','significant']].to_string(index=False)}")
    print(f"\n  Stats -> {stats_path}")

    # ------------------------------------------------------------------
    # Step 7: Patient factor analysis
    # ------------------------------------------------------------------
    print("\n[Step 7] Patient factor analysis...")
    factor_df  = patient_factor_analysis(df_results)
    factor_path = out_dir / "factor_correlations.csv"
    factor_df.to_csv(factor_path, index=False)
    print(f"\n{factor_df[['factor','pearson_r','p_value','significant']].to_string(index=False)}")
    print(f"\n  Factor analysis -> {factor_path}")

    # ------------------------------------------------------------------
    # Step 8: Multiple linear regression
    # ------------------------------------------------------------------
    print("\n[Step 8] Multiple linear regression...")
    reg_df   = multiple_linear_regression(df_results)
    reg_path = out_dir / "regression_results.csv"
    reg_df.to_csv(reg_path, index=False)
    if not reg_df.empty:
        print(f"\n{reg_df.to_string(index=False)}")
    print(f"\n  Regression -> {reg_path}")

    # ------------------------------------------------------------------
    # Step 9: Plots
    # ------------------------------------------------------------------
    print("\n[Step 9] Generating plots...")
    plot_personalisation_benefit(
        df_results, str(out_dir / "personalisation_benefit.png")
    )
    plot_factor_correlations(
        factor_df, str(out_dir / "factor_correlations.png")
    )
    plot_regression_coefficients(
        reg_df, str(out_dir / "regression_coefficients.png")
    )
    plot_per_metric_comparison(
        df_results, str(out_dir / "per_metric_comparison.png")
    )
    plot_action_profiles(
        df_results, str(out_dir / "action_profiles.png")
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  RQ4 complete.")
    print(f"  G-DDQN mean  : {df_results['gddqn_total'].mean():.4f}")
    print(f"  P-DDQN mean  : {df_results['pddqn_total'].mean():.4f}")
    print(f"  Benefit mean : {df_results['personalisation_benefit'].mean():.4f}")
    print(f"\n  Results -> {out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()