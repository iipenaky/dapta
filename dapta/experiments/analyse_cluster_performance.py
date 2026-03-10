"""
Cluster Performance Analysis — run after run_evaluation.py

Answers the question: which patient clusters benefited most from
personalisation (DAPTA vs G-DDQN), and which subtypes drove the results?

Reads:
  outputs/evaluation/improvements_DAPTA.npy
  outputs/evaluation/improvements_G_DDQN.npy
  outputs/pes/cluster_assignments.json
  outputs/dae/patient_profiles.json
  outputs/dae/splits.json
  outputs/pes/cluster_summary.json   (from visualise_clusters.py)

Outputs:
  outputs/evaluation/cluster_performance.json   — numbers for thesis
  outputs/evaluation/cluster_performance.png    — bar chart

Usage:
    python experiments/analyse_cluster_performance.py
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Paths ─────────────────────────────────────────────────────────────────────

eval_dir    = Path("outputs/evaluation")
pes_dir     = Path("outputs/pes")
dae_dir     = Path("outputs/dae")

METRIC_NAMES = ["CIU_rate", "MC_score", "MLU_morphemes", "TTR", "SynComp", "Surprisal"]

# ── Load data ─────────────────────────────────────────────────────────────────

dapta_imp  = np.load(eval_dir / "improvements_DAPTA.npy")    # (N_test, 6)
gddqn_imp  = np.load(eval_dir / "improvements_G_DDQN.npy")   # (N_test, 6)

with open(pes_dir  / "cluster_assignments.json") as f:
    cluster_data = json.load(f)
with open(dae_dir  / "patient_profiles.json") as f:
    profiles_data = json.load(f)
with open(dae_dir  / "splits.json") as f:
    splits = json.load(f)

# Load cluster summary labels if available
cluster_labels_map = {}
cluster_summary_path = pes_dir / "cluster_summary.json"
if cluster_summary_path.exists():
    with open(cluster_summary_path) as f:
        cluster_summary = json.load(f)
    cluster_labels_map = {
        int(k): v["suggested_label"]
        for k, v in cluster_summary.items()
    }

# ── Map test patients to clusters and profiles ────────────────────────────────

assignments = cluster_data["assignments"]           # {participant_id: cluster_id}
test_ids    = splits["test"]                        # list of session_ids

# patient_profiles.json is indexed by session order — map session_id → profile
profiles_by_id = {p["participant_id"]: p for p in profiles_data}

# Build per-test-patient records
# test_ids order matches row order in improvements_*.npy
records = []
for i, sid in enumerate(test_ids):
    if i >= len(dapta_imp):
        break
    cluster = assignments.get(sid, -1)
    profile = profiles_by_id.get(sid, {})
    records.append({
        "idx":     i,
        "sid":     sid,
        "cluster": cluster,
        "subtype": profile.get("aphasia_subtype", "Unknown"),
        "wab_aq":  profile.get("wab_aq", 50.0),
    })

n_clusters = max(r["cluster"] for r in records if r["cluster"] >= 0) + 1

# ── Per-cluster analysis ──────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d: positive = a > b (DAPTA better than G-DDQN)."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)

results = {}

print("\n" + "═" * 80)
print("CLUSTER PERFORMANCE: DAPTA vs G-DDQN")
print("═" * 80)

for c in range(n_clusters):
    indices = [r["idx"] for r in records if r["cluster"] == c]
    if len(indices) < 3:
        continue

    d_imp = dapta_imp[indices]   # (n, 6)
    g_imp = gddqn_imp[indices]   # (n, 6)

    # CIU rate personalisation effect (primary metric)
    ciu_d   = d_imp[:, 0]
    ciu_g   = g_imp[:, 0]
    d_stat  = cohens_d(ciu_d, ciu_g)
    t_stat, p_val = scipy_stats.ttest_ind(ciu_d, ciu_g)

    # Mean improvement across all non-surprisal metrics (0:5)
    dapta_overall = d_imp[:, :5].mean(axis=1)
    gddqn_overall = g_imp[:, :5].mean(axis=1)
    overall_d     = cohens_d(dapta_overall, gddqn_overall)

    # Subtypes in this cluster
    subtypes = Counter(r["subtype"] for r in records if r["cluster"] == c)
    dominant = subtypes.most_common(1)[0][0]
    label    = cluster_labels_map.get(c, f"Cluster {c}")

    personalisation_verdict = (
        "STRONG"    if d_stat >= 0.40 else
        "MODERATE"  if d_stat >= 0.20 else
        "WEAK"      if d_stat >= 0.0  else
        "NEGATIVE"
    )

    results[c] = {
        "label":                  label,
        "n":                      len(indices),
        "dominant_subtype":       dominant,
        "subtype_counts":         dict(subtypes),
        "mean_wab_aq":            round(float(np.mean([r["wab_aq"] for r in records if r["cluster"] == c])), 1),
        "dapta_mean_ciu":         round(float(ciu_d.mean()), 4),
        "gddqn_mean_ciu":         round(float(ciu_g.mean()), 4),
        "ciu_cohens_d":           round(d_stat, 3),
        "ciu_p_value":            round(float(p_val), 4),
        "overall_cohens_d":       round(overall_d, 3),
        "personalisation_verdict": personalisation_verdict,
        # Per-metric breakdown
        "per_metric": {
            m: {
                "dapta_mean": round(float(d_imp[:, i].mean()), 4),
                "gddqn_mean": round(float(g_imp[:, i].mean()), 4),
                "cohens_d":   round(cohens_d(d_imp[:, i], g_imp[:, i]), 3),
            }
            for i, m in enumerate(METRIC_NAMES)
        },
    }

    print(f"\nCluster {c}: {label}  (n={len(indices)}, WAB={results[c]['mean_wab_aq']})")
    print(f"  Subtypes: " + "  ".join(f"{k}={v}" for k, v in subtypes.most_common()))
    print(f"  CIU:  DAPTA={ciu_d.mean():.3f}  G-DDQN={ciu_g.mean():.3f}  "
          f"d={d_stat:.3f}  p={p_val:.4f}")
    print(f"  Overall d (all metrics): {overall_d:.3f}")
    print(f"  Personalisation: {personalisation_verdict}")

print("\n" + "═" * 80)

# ── Ranking ───────────────────────────────────────────────────────────────────

ranked = sorted(results.items(), key=lambda x: x[1]["overall_cohens_d"], reverse=True)
print("\nClusters ranked by personalisation benefit (overall Cohen's d, DAPTA vs G-DDQN):")
for rank, (c, r) in enumerate(ranked, 1):
    print(f"  {rank}. Cluster {c} — {r['label']}  "
          f"d={r['overall_cohens_d']:.3f}  [{r['personalisation_verdict']}]")

# ── Bar chart ─────────────────────────────────────────────────────────────────

PALETTE = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261","#264653"]

cluster_ids  = list(results.keys())
d_vals       = [results[c]["overall_cohens_d"] for c in cluster_ids]
labels_short = [f"C{c}\n{results[c]['label'][:18]}\nn={results[c]['n']}" for c in cluster_ids]
colors       = [
    "#2A9D8F" if d >= 0.40 else
    "#E9C46A" if d >= 0.20 else
    "#F4A261" if d >= 0.0  else
    "#E63946"
    for d in d_vals
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#0F1117")

# Left — overall Cohen's d per cluster
ax = axes[0]
ax.set_facecolor("#161B22")
bars = ax.bar(range(len(cluster_ids)), d_vals, color=colors, width=0.6, edgecolor="none")
ax.axhline(0.40, color="white",  linestyle="--", linewidth=0.8, alpha=0.5, label="Clinical threshold (d=0.40)")
ax.axhline(0.0,  color="#666666",linestyle="-",  linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, d_vals)):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.01 if val >= 0 else -0.03),
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
            color="white", fontsize=8, fontweight="bold")
ax.set_xticks(range(len(cluster_ids)))
ax.set_xticklabels(labels_short, fontsize=7.5, color="#AAAAAA")
ax.set_ylabel("Cohen's d  (DAPTA vs G-DDQN)", color="#AAAAAA", fontsize=10)
ax.set_title("Personalisation Benefit by Cluster\n(Overall — all metrics)",
             color="white", fontsize=11, fontweight="bold", pad=10)
ax.tick_params(colors="#666666")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.legend(fontsize=8, facecolor="#1C2128", edgecolor="#444444", labelcolor="white")

# Right — per-metric Cohen's d heatmap style for top 2 clusters
ax2 = axes[1]
ax2.set_facecolor("#161B22")

# Build matrix: clusters × metrics
matrix = np.array([
    [results[c]["per_metric"][m]["cohens_d"] for m in METRIC_NAMES]
    for c in cluster_ids
])

im = ax2.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.8, vmax=0.8)
ax2.set_xticks(range(len(METRIC_NAMES)))
ax2.set_xticklabels(METRIC_NAMES, rotation=35, ha="right", color="#AAAAAA", fontsize=9)
ax2.set_yticks(range(len(cluster_ids)))
ax2.set_yticklabels([f"C{c}: {results[c]['label'][:20]}" for c in cluster_ids],
                     color="#AAAAAA", fontsize=8)
for i in range(len(cluster_ids)):
    for j in range(len(METRIC_NAMES)):
        val = matrix[i, j]
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 color="black" if abs(val) < 0.5 else "white", fontsize=8)
ax2.set_title("Cohen's d per Metric per Cluster\n(green = DAPTA better, red = G-DDQN better)",
              color="white", fontsize=10, fontweight="bold", pad=10)
plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color="#AAAAAA")

plt.tight_layout(pad=2.0)
out_path = eval_dir / "cluster_performance.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"\nChart saved → {out_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────

with open(eval_dir / "cluster_performance.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Data saved  → {eval_dir}/cluster_performance.json")
print()