"""
find_optimal_k.py
-----------------
Diagnostic script for finding the optimal number of patient clusters (k)
before running run_pes.py.

Matches PatientClusterer exactly:
  - Excludes non-aphasic / control patients from clustering
  - Uses discourse means + WAB-AQ + subtype one-hot as features
  - Applies StandardScaler before clustering

Run this once, inspect the plot and table, then set min_k in
PatientClusterer (or just let it auto-select via elbow).
"""

from pathlib import Path
from typing import List, Set

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Must match PatientClusterer exactly ──────────────────────────
NON_APHASIA_SUBTYPES: Set[str] = {
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
}

N_METRICS_PER_TASK = 5
N_TASKS            = 5


def extract_discourse_means(state_vectors: np.ndarray) -> np.ndarray:
    """Average each metric across all tasks — matches PatientClusterer._extract_discourse_means."""
    N          = state_vectors.shape[0]
    disc_means = np.zeros((N, N_METRICS_PER_TASK), dtype=np.float32)
    for m in range(N_METRICS_PER_TASK):
        dims = [t * N_METRICS_PER_TASK + m for t in range(N_TASKS)]
        disc_means[:, m] = state_vectors[:, dims].mean(axis=1)
    return disc_means


def build_feature_matrix(
    aphasia_profiles: list,
    aphasia_states:   np.ndarray,
) -> np.ndarray:
    """Builds the same feature matrix PatientClusterer._fit_transform produces."""
    subtypes        = np.array([[p["aphasia_subtype"]] for p in aphasia_profiles])
    wab_aq          = np.array([
        float(p.get("wab_aq") or 55.0) for p in aphasia_profiles
    ]).reshape(-1, 1)

    encoder         = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    subtype_encoded = encoder.fit_transform(subtypes)

    disc_means      = extract_discourse_means(aphasia_states)
    numeric         = np.hstack([wab_aq, disc_means])
    numeric_scaled  = StandardScaler().fit_transform(numeric)

    return np.hstack([subtype_encoded, numeric_scaled]).astype(np.float32)


def kneedle_elbow(k_range: range, inertias: List[float]) -> int:
    """Kneedle algorithm — perpendicular distance from line p1→p2."""
    p1        = np.array([k_range[0],  inertias[0]])
    p2        = np.array([k_range[-1], inertias[-1]])
    distances = []
    for i, k in enumerate(k_range):
        p0   = np.array([k, inertias[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    return k_range[int(np.argmax(distances))]


def second_diff_elbow(k_range: range, inertias: List[float]) -> int:
    """Second-difference elbow — matches PatientClusterer.find_elbow_k."""
    deltas        = np.diff(inertias)
    second_deltas = np.diff(deltas)
    if len(second_deltas) == 0:
        return k_range[0]
    return k_range[int(np.argmax(np.abs(second_deltas))) + 2]


def find_optimal_k(
    state_vectors_path: str = "outputs/dae/state_vectors.npz",
    profiles_path:      str = "outputs/dae/patient_profiles.json",
    max_k:              int = 15,
    min_k:              int = 4,
) -> int:
    # ── Load data ─────────────────────────────────────────────────
    if not Path(state_vectors_path).exists():
        print(f"Error: {state_vectors_path} not found. Run run_dae.py first.")
        return -1

    data          = np.load(state_vectors_path, allow_pickle=True)
    state_vectors = data["state_vectors"]
    session_ids   = list(data["session_ids"])

    with open(profiles_path) as f:
        profiles = json.load(f)

    session_to_profile = {p["session_id"]: p for p in profiles}
    ordered_profiles   = [
        session_to_profile.get(str(sid), {"aphasia_subtype": "Other", "wab_aq": None})
        for sid in session_ids
    ]

    # ── Filter to aphasia patients only — matches PatientClusterer ─
    aphasia_mask = np.array([
        p["aphasia_subtype"] not in NON_APHASIA_SUBTYPES
        for p in ordered_profiles
    ], dtype=bool)

    aphasia_profiles = [p for p, m in zip(ordered_profiles, aphasia_mask) if m]
    aphasia_states   = state_vectors[aphasia_mask]

    n_aphasia  = int(aphasia_mask.sum())
    n_controls = int((~aphasia_mask).sum())
    print(f"\nPatients loaded:  {len(ordered_profiles)}")
    print(f"Aphasia patients: {n_aphasia}")
    print(f"Controls excluded: {n_controls}")

    if n_aphasia < 4:
        print("Error: fewer than 4 aphasia patients — cannot cluster.")
        return -1

    # ── Build feature matrix ──────────────────────────────────────
    X         = build_feature_matrix(aphasia_profiles, aphasia_states)
    max_k_use = min(max_k, n_aphasia - 1)
    k_range   = range(2, max_k_use + 1)

    # ── Run KMeans for each k ─────────────────────────────────────
    inertias    = []
    silhouettes = []

    print(f"\nRunning KMeans for k=2 to {max_k_use} on {n_aphasia} aphasia patients...")
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        silhouettes.append(sil)

    # ── Elbow detection — both methods ───────────────────────────
    elbow_kneedle     = kneedle_elbow(k_range, inertias)
    elbow_second_diff = second_diff_elbow(k_range, inertias)
    best_sil_k        = k_range[int(np.argmax(silhouettes))]

    # Enforce min_k
    recommended_k = max(min_k, elbow_kneedle)

    # ── Print results ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"CLUSTERING RESULTS  (n={n_aphasia} aphasia patients)")
    print("=" * 50)
    print(f"Elbow (Kneedle):      k = {elbow_kneedle}")
    print(f"Elbow (second diff):  k = {elbow_second_diff}  ← used by PatientClusterer")
    print(f"Best silhouette:      k = {best_sil_k}  (score={max(silhouettes):.4f})")
    print(f"Recommended (min_k={min_k} enforced): k = {recommended_k}")
    print("=" * 50)

    print("\nDetailed breakdown:")
    print(f"{'k':<4} {'Inertia':>10} {'Silhouette':>12}  {'Note'}")
    print("-" * 50)
    for i, k in enumerate(k_range):
        notes = []
        if k == elbow_kneedle:     notes.append("kneedle elbow")
        if k == elbow_second_diff: notes.append("2nd-diff elbow")
        if k == best_sil_k:        notes.append("best silhouette")
        if k == recommended_k:     notes.append("★ RECOMMENDED")
        note = ", ".join(notes)
        print(f"{k:<4} {inertias[i]:>10.2f} {silhouettes[i]:>12.4f}  {note}")

    # ── Save plot ─────────────────────────────────────────────────
    save_path = Path("outputs/plots/optimal_k_results.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Optimal k for {n_aphasia} aphasia patients "
        f"({n_controls} controls excluded)",
        fontsize=12,
    )

    # Elbow plot
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=1.5)
    axes[0].axvline(
        x=elbow_kneedle, color="red", linestyle="--",
        label=f"Kneedle elbow k={elbow_kneedle}",
    )
    axes[0].axvline(
        x=elbow_second_diff, color="orange", linestyle=":",
        label=f"2nd-diff elbow k={elbow_second_diff}",
    )
    axes[0].axvline(
        x=recommended_k, color="green", linestyle="-",
        alpha=0.4, linewidth=3,
        label=f"Recommended k={recommended_k}",
    )
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")
    axes[0].legend(fontsize=8)

    # Silhouette plot
    axes[1].plot(list(k_range), silhouettes, "go-", linewidth=1.5)
    axes[1].axvline(
        x=best_sil_k, color="red", linestyle="--",
        label=f"Best silhouette k={best_sil_k}",
    )
    axes[1].axvline(
        x=recommended_k, color="green", linestyle="-",
        alpha=0.4, linewidth=3,
        label=f"Recommended k={recommended_k}",
    )
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score (higher = better)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    print(f"\nPlot saved to: {save_path}")

    print(f"\n→ Set min_k={recommended_k} in PatientClusterer (or run_pes.py)")
    print(f"  If kneedle ({elbow_kneedle}) and second-diff ({elbow_second_diff}) agree, "
          f"that k is very reliable.")

    return recommended_k


if __name__ == "__main__":
    find_optimal_k()