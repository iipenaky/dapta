"""
cluster.py
==========
Clusters patients based on their validated discourse metrics and WAB-AQ.

Why clustering is needed
------------------------
You cannot train a separate RL agent per patient because most patients
have only one recorded session — not enough data to learn from.
Instead, patients are grouped by language ability profile, and one
RL agent is trained per group. This is the patient-specific policy.
Comparing it to one agent trained on everyone (G-DDQN) answers RQ4:
does personalisation help beyond a generalised approach?

Why cluster on discourse metrics (not diagnosis labels)?
--------------------------------------------------------
Diagnosis labels (Broca, Wernicke, Anomic) describe symptom type,
not language ability. Two patients with the same diagnosis can have
very different MLU and TTR. The RL agent needs to personalise based
on what the patient can actually do. Clustering on validated discourse
metrics groups patients by real measured ability.

Features used (all validated against CLAN, r >= 0.97):
  mlu_words     r = 0.9974
  mlu_morphemes r = 0.9953
  ttr           r = 0.9836
  ndw           r = 0.9978
  wab_aq        clinical severity (imputed where missing)

Optimal k is found using the elbow method (second derivative of inertia).

Outputs
-------
  state_vectors.csv        updated with cluster_id column added
  results/clustering/elbow_plot.png
  results/clustering/cluster_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Features we cluster on
CLUSTER_FEATURES = ["mlu_words", "mlu_morphemes", "ttr", "ndw", "wab_aq"]

# WAB-AQ population medians for imputation where score is missing.
# Source: AphasiaBank normative data.
_WAB_MEDIANS = {
    "anomic":        78.0,
    "conduction":    68.0,
    "wernicke":      52.0,
    "broca":         46.0,
    "transmotor":    55.0,
    "transsensory":  60.0,
    "global":        18.0,
    "isolation":     20.0,
    "control":       99.0,
    "notaphasicbywab": 99.0,
}


# =============================================================================
# IMPUTATION
# =============================================================================

def _impute_wab_aq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing WAB-AQ using diagnosis-based population medians.
    Any still missing after diagnosis lookup → dataset median.
    """
    df = df.copy()
    missing = df["wab_aq"].isna()

    if missing.sum() == 0:
        return df

    print(f"  Imputing WAB-AQ for {missing.sum()} participants "
          f"using diagnosis-based medians.")

    if "diagnosis" in df.columns:
        for idx in df[missing].index:
            diag = str(df.at[idx, "diagnosis"]).lower().strip()
            for key, median in _WAB_MEDIANS.items():
                if key in diag:
                    df.at[idx, "wab_aq"] = median
                    break

    still_missing = df["wab_aq"].isna()
    if still_missing.sum() > 0:
        dataset_median = df["wab_aq"].median()
        df.loc[still_missing, "wab_aq"] = dataset_median
        print(f"  {still_missing.sum()} still missing → "
              f"dataset median WAB-AQ = {dataset_median:.1f}")

    return df


# =============================================================================
# FEATURE MATRIX
# =============================================================================

def _build_feature_matrix(df: pd.DataFrame):
    """
    Build the scaled feature matrix for k-means.

    Returns
    -------
    X_scaled  : np.ndarray shape (N, 5), StandardScaled
    valid_idx : original DataFrame index of rows that were kept
    """
    df = _impute_wab_aq(df)

    # Drop rows with no discourse data at all
    discourse_cols = [c for c in CLUSTER_FEATURES if c != "wab_aq"]
    valid_mask = df[discourse_cols].notna().any(axis=1)
    df_valid   = df[valid_mask].copy()
    valid_idx  = df_valid.index.values

    dropped = len(df) - len(df_valid)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with no discourse metrics.")

    X_raw = df_valid[CLUSTER_FEATURES].values.astype(float)

    # Impute any remaining NaN with column median
    imputer  = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # StandardScale — essential before k-means so no feature dominates
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled.astype(np.float32), valid_idx


# =============================================================================
# ELBOW METHOD
# =============================================================================

def _compute_inertias(X: np.ndarray, max_k: int, seed: int) -> list:
    """Compute k-means inertia for k = 1 .. max_k."""
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def _find_elbow_k(inertias: list) -> int:
    """
    Find optimal k using the second derivative of the inertia curve.

    The elbow is where adding another cluster gives the least additional
    improvement. The second derivative peaks at this point.
    Returns k >= 2.
    """
    if len(inertias) < 3:
        return 2

    first_deriv  = np.diff(inertias)
    second_deriv = np.diff(first_deriv)
    elbow_idx    = int(np.argmax(np.abs(second_deriv)))
    k            = elbow_idx + 2   # +2 because two np.diff() calls

    return max(2, k)


def _plot_elbow(inertias: list, best_k: int, out_path: Path):
    """Save the elbow plot for the thesis."""
    ks = list(range(1, len(inertias) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "o-", color="steelblue", lw=2, markersize=7)
    ax.axvline(x=best_k, color="red", ls="--", lw=1.8,
               label=f"Elbow  k = {best_k}")
    ax.set_xlabel("Number of clusters  k", fontsize=12)
    ax.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=12)
    ax.set_title("K-Means Elbow Method — Patient Language Ability Profiles",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Elbow plot → {out_path}")


# =============================================================================
# CLUSTER SUMMARY
# =============================================================================

def _cluster_summary(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """
    Print and save a summary table: one row per cluster showing
    mean discourse metrics and most common diagnosis.
    """
    summary_cols = CLUSTER_FEATURES + ["diagnosis"]
    available    = [c for c in summary_cols if c in df.columns]

    grouped = df.groupby("cluster_id")

    rows = []
    for cluster_id, group in grouped:
        row = {"cluster_id": cluster_id, "n_patients": len(group)}
        for col in CLUSTER_FEATURES:
            if col in group.columns:
                row[f"mean_{col}"] = round(group[col].mean(), 3)
        if "diagnosis" in group.columns:
            row["top_diagnosis"] = (
                group["diagnosis"].value_counts().idxmax()
                if group["diagnosis"].notna().any() else "unknown"
            )
        rows.append(row)

    df_summary = pd.DataFrame(rows).sort_values("cluster_id")
    df_summary.to_csv(out_path, index=False)
    print(f"  Cluster summary → {out_path}")

    # Print to terminal
    print(f"\n{'Cluster':>8} {'N':>6} {'MLU-w':>7} {'MLU-m':>7} "
          f"{'TTR':>6} {'NDW':>6} {'WAB-AQ':>8} {'Top diagnosis'}")
    print("-" * 75)
    for _, row in df_summary.iterrows():
        print(
            f"  {int(row['cluster_id']):>6} "
            f"  {int(row['n_patients']):>4} "
            f"  {row.get('mean_mlu_words', float('nan')):>6.2f} "
            f"  {row.get('mean_mlu_morphemes', float('nan')):>6.2f} "
            f"  {row.get('mean_ttr', float('nan')):>5.3f} "
            f"  {row.get('mean_ndw', float('nan')):>5.0f} "
            f"  {row.get('mean_wab_aq', float('nan')):>7.1f} "
            f"  {row.get('top_diagnosis', '')}"
        )

    return df_summary


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_clustering(
    state_vectors_csv: str,
    out_dir:           str = "results/clustering",
    max_k:             int = 10,
    seed:              int = 42,
) -> pd.DataFrame:
    """
    Run the full clustering pipeline on aphasia patients only.

    Controls are excluded before clustering — they would dominate
    the feature space and produce a trivial split (aphasia vs healthy)
    rather than clinically meaningful subgroups.

    Parameters
    ----------
    state_vectors_csv : path to state_vectors.csv from run_rq1.py
    out_dir           : directory for elbow plot and cluster summary
    max_k             : maximum k to try in elbow method
    seed              : random seed for reproducibility

    Returns
    -------
    pd.DataFrame : state_vectors with cluster_id column added.
                   Controls and rows with no data get cluster_id = -1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Patient Clustering (k-means + elbow method)")
    print(f"  Features: {CLUSTER_FEATURES}")
    print(f"{'='*60}")

    # Load state vectors
    df = pd.read_csv(state_vectors_csv)
    print(f"\nLoaded {len(df)} participants from {state_vectors_csv}")

    # ---- Filter to aphasia patients only ----
    # Controls are not therapy candidates. Clustering them in would simply
    # separate aphasia vs healthy, which is useless for the RL agent.
    # Controls remain in the CSV with cluster_id = -1 for reference.
    CONTROL_LABELS = {
        "control", "notaphasicbywab", "notaphasicbywab",
        "normal", "healthy", "typical"
    }
    if "diagnosis" in df.columns:
        is_control = df["diagnosis"].str.lower().str.strip().isin(CONTROL_LABELS)
        df_aphasia = df[~is_control].copy()
        print(f"\nExcluded {is_control.sum()} control participants from clustering.")
        print(f"Clustering on {len(df_aphasia)} aphasia patients.")
    else:
        df_aphasia = df.copy()
        is_control = pd.Series(False, index=df.index)

    # Build feature matrix on aphasia patients only
    print("\nBuilding feature matrix...")
    df_aphasia_reset = df_aphasia.reset_index(drop=False)  # saves old index in "index
    X_scaled, valid_local = _build_feature_matrix(df_aphasia_reset)
    # Map local indices back to original df index
    valid_idx = df_aphasia_reset.loc[valid_local, "index"].values 
    print(f"  Feature matrix shape: {X_scaled.shape}")

    # Find optimal k using elbow method
    print(f"\nRunning elbow method (k = 1 to {max_k})...")
    inertias = _compute_inertias(X_scaled, max_k, seed)
    best_k   = _find_elbow_k(inertias)
    print(f"  Optimal k = {best_k}  (selected by second-derivative elbow)")

    for k, inertia in enumerate(inertias, start=1):
        marker = " <- elbow" if k == best_k else ""
        print(f"    k={k:2d}  inertia={inertia:12.1f}{marker}")

    # Save elbow plot
    _plot_elbow(inertias, best_k, out_dir / "elbow_plot.png")

    # Run final k-means with best k
    print(f"\nRunning k-means with k={best_k}...")
    km     = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)

    counts = np.bincount(labels)
    for i, c in enumerate(counts):
        print(f"  Cluster {i}: {c} patients")

    # Add cluster_id — controls and no-data rows get -1
    df["cluster_id"] = -1
    df.loc[valid_idx, "cluster_id"] = labels

    # Save updated state vectors
    df.to_csv(state_vectors_csv, index=False)
    print(f"\nCluster IDs added to {state_vectors_csv}")

    # Cluster summary (aphasia patients only)
    df_with_clusters = df[df["cluster_id"] >= 0].copy()
    _cluster_summary(df_with_clusters, out_dir / "cluster_summary.csv")

    n_assigned = (df["cluster_id"] >= 0).sum()
    n_excluded = (df["cluster_id"] < 0).sum()
    print(f"\n{'='*60}")
    print(f"  Clustering complete.")
    print(f"  k = {best_k} clusters")
    print(f"  {n_assigned} aphasia patients assigned to clusters")
    print(f"  {n_excluded} excluded (controls or missing data)")
    print(f"{'='*60}")
    print("\nNext step: build the RL environment (run_rq2.py)")

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster patients by discourse ability profile"
    )
    parser.add_argument(
        "--state_vectors",
        default="data/processed/state_vectors.csv",
        help="Path to state_vectors.csv from run_rq1.py"
    )
    parser.add_argument("--out_dir", default="results/clustering")
    parser.add_argument("--max_k",   default=10, type=int)
    parser.add_argument("--seed",    default=42, type=int)
    args = parser.parse_args()

    run_clustering(
        state_vectors_csv = args.state_vectors,
        out_dir           = args.out_dir,
        max_k             = args.max_k,
        seed              = args.seed,
    )