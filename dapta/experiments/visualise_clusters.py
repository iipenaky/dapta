"""
Cluster Visualisation Script — run after run_pes.py

Reads outputs/pes/cluster_assignments.json and outputs/dae/patient_profiles.json
and produces:
  1. A printed table showing cluster composition (subtype × severity breakdown)
  2. A PCA scatter plot saved as outputs/pes/cluster_pca.png
  3. A summary JSON for thesis reporting

Usage:
    python visualise_clusters.py

Run from the dapta/ root directory.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ── Load data ────────────────────────────────────────────────────────────────

dae_dir   = Path("outputs/dae")
pes_dir   = Path("outputs/pes")
out_dir   = Path("outputs/pes")
out_dir.mkdir(parents=True, exist_ok=True)

with open(dae_dir / "patient_profiles.json") as f:
    profiles_data = json.load(f)

with open(pes_dir / "cluster_assignments.json") as f:
    cluster_data = json.load(f)

assignments = cluster_data["assignments"]          # {participant_id: cluster_id}
n_clusters  = max(assignments.values()) + 1

# ── Build per-patient records ─────────────────────────────────────────────────

SEVERITY_BAND = {
    "Severe":        (0,   25),
    "Moderate":      (25,  50),
    "Mild-moderate": (50,  75),
    "Mild":          (75, 100),
}

def severity_label(wab_aq: float) -> str:
    if wab_aq < 25:  return "Severe"
    if wab_aq < 50:  return "Moderate"
    if wab_aq < 75:  return "Mild-moderate"
    return "Mild"

records = []
for p in profiles_data:
    pid = p["participant_id"]
    if pid not in assignments:
        continue
    records.append({
        "participant_id": pid,
        "cluster":        assignments[pid],
        "subtype":        p.get("aphasia_subtype", "Other"),
        "wab_aq":         p.get("wab_aq", 50.0),
        "months":         p.get("months_post_onset", 12.0),
        "severity":       severity_label(p.get("wab_aq", 50.0)),
    })

# ── Table: cluster composition ───────────────────────────────────────────────

SUBTYPES   = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]
SEVERITIES = ["Severe", "Moderate", "Mild-moderate", "Mild"]

print("\n" + "═" * 72)
print("CLUSTER COMPOSITION")
print("═" * 72)

cluster_summaries = {}

for c in range(n_clusters):
    members = [r for r in records if r["cluster"] == c]
    if not members:
        continue

    wab_vals  = [r["wab_aq"] for r in members]
    subtype_counts  = Counter(r["subtype"]  for r in members)
    severity_counts = Counter(r["severity"] for r in members)

    dominant_subtype   = subtype_counts.most_common(1)[0][0]
    dominant_severity  = severity_counts.most_common(1)[0][0]

    print(f"\nCluster {c}  (n={len(members)})  "
          f"WAB-AQ: {np.mean(wab_vals):.1f} ± {np.std(wab_vals):.1f}")
    print(f"  Subtypes  : " +
          "  ".join(f"{s}={subtype_counts[s]}" for s in SUBTYPES if subtype_counts[s] > 0))
    print(f"  Severity  : " +
          "  ".join(f"{s}={severity_counts[s]}" for s in SEVERITIES if severity_counts[s] > 0))
    print(f"  → Dominant: {dominant_severity} {dominant_subtype}")

    cluster_summaries[c] = {
        "n":               len(members),
        "mean_wab_aq":     round(float(np.mean(wab_vals)), 2),
        "std_wab_aq":      round(float(np.std(wab_vals)),  2),
        "dominant_subtype":   dominant_subtype,
        "dominant_severity":  dominant_severity,
        "subtype_counts":     dict(subtype_counts),
        "severity_counts":    dict(severity_counts),
        "suggested_label":    f"{dominant_severity} {dominant_subtype}",
    }

print("\n" + "═" * 72)

# ── PCA scatter plot ──────────────────────────────────────────────────────────

le = LabelEncoder().fit(SUBTYPES)

def severity_band_numeric(wab_aq: float) -> float:
    if wab_aq < 25:  return 0.0
    if wab_aq < 50:  return 0.33
    if wab_aq < 75:  return 0.67
    return 1.0

X = np.array([
    [
        le.transform([r["subtype"] if r["subtype"] in SUBTYPES else "Other"])[0],
        severity_band_numeric(r["wab_aq"]),
        min(r["months"], 60) / 60.0,
    ]
    for r in records
], dtype=np.float32)

labels = np.array([r["cluster"] for r in records])

pca  = PCA(n_components=2, random_state=42)
X2   = pca.fit_transform(X)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#264653", "#A8DADC", "#6D6875",
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0F1117")

# ── Left: coloured by cluster ─────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#161B22")

for c in range(n_clusters):
    mask = labels == c
    if not mask.any():
        continue
    summary = cluster_summaries.get(c, {})
    lbl = f"C{c}: {summary.get('suggested_label', '')} (n={summary.get('n',0)})"
    ax.scatter(
        X2[mask, 0], X2[mask, 1],
        c=PALETTE[c % len(PALETTE)],
        s=60, alpha=0.85, edgecolors="none", label=lbl,
    )

ax.set_xlabel(f"PC1 ({var1:.1f}% var)", color="#AAAAAA", fontsize=10)
ax.set_ylabel(f"PC2 ({var2:.1f}% var)", color="#AAAAAA", fontsize=10)
ax.set_title("Patient Clusters (PCA)", color="white", fontsize=12, fontweight="bold", pad=10)
ax.tick_params(colors="#666666")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

legend = ax.legend(
    loc="upper right", fontsize=7.5,
    facecolor="#1C2128", edgecolor="#444444", labelcolor="white",
    framealpha=0.9,
)

# ── Right: coloured by severity ───────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#161B22")

SEV_COLORS = {
    "Severe":        "#E63946",
    "Moderate":      "#F4A261",
    "Mild-moderate": "#2A9D8F",
    "Mild":          "#457B9D",
}
SEV_ORDER = ["Severe", "Moderate", "Mild-moderate", "Mild"]

for sev in SEV_ORDER:
    mask = np.array([r["severity"] == sev for r in records])
    if not mask.any():
        continue
    ax2.scatter(
        X2[mask, 0], X2[mask, 1],
        c=SEV_COLORS[sev],
        s=50, alpha=0.80, edgecolors="none", label=sev,
    )

# Annotate cluster centroids
for c in range(n_clusters):
    mask = labels == c
    if not mask.any():
        continue
    cx, cy = X2[mask, 0].mean(), X2[mask, 1].mean()
    ax2.text(cx, cy, str(c), color="white", fontsize=9,
             ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000088", edgecolor="none"))

ax2.set_xlabel(f"PC1 ({var1:.1f}% var)", color="#AAAAAA", fontsize=10)
ax2.set_ylabel(f"PC2 ({var2:.1f}% var)", color="#AAAAAA", fontsize=10)
ax2.set_title("Severity Distribution Across Clusters", color="white",
              fontsize=12, fontweight="bold", pad=10)
ax2.tick_params(colors="#666666")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")

ax2.legend(
    loc="upper right", fontsize=8,
    facecolor="#1C2128", edgecolor="#444444", labelcolor="white",
    framealpha=0.9,
)

plt.tight_layout(pad=2.0)
out_path = out_dir / "cluster_pca.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"\nPCA plot saved → {out_path}")

# ── WAB-AQ distribution per cluster ──────────────────────────────────────────

fig2, ax3 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("#0F1117")
ax3.set_facecolor("#161B22")

cluster_wab = [
    [r["wab_aq"] for r in records if r["cluster"] == c]
    for c in range(n_clusters)
]
cluster_wab = [w for w in cluster_wab if w]  # drop empty

bp = ax3.boxplot(
    cluster_wab,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#666666"),
    capprops=dict(color="#666666"),
    flierprops=dict(marker="o", markerfacecolor="#666666", markersize=4, linestyle="none"),
)

for i, (patch, wab_list) in enumerate(zip(bp["boxes"], cluster_wab)):
    patch.set_facecolor(PALETTE[i % len(PALETTE)])
    patch.set_alpha(0.85)
    summary = cluster_summaries.get(i, {})
    ax3.text(
        i + 1, -5,
        f"C{i}\n{summary.get('suggested_label','')}\nn={summary.get('n',0)}",
        ha="center", va="top", color="#AAAAAA", fontsize=7.5,
    )

# Draw WAB severity lines
for threshold, label in [(25, "Severe/Mod"), (50, "Mod/Mild-mod"), (75, "Mild-mod/Mild")]:
    ax3.axhline(threshold, color="#FF6B6B", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.text(len(cluster_wab) + 0.55, threshold, label,
             color="#FF6B6B", fontsize=7, va="center")

ax3.set_ylim(-12, 105)
ax3.set_ylabel("WAB-AQ Score", color="#AAAAAA", fontsize=10)
ax3.set_title("WAB-AQ Distribution per Cluster", color="white",
              fontsize=12, fontweight="bold", pad=10)
ax3.tick_params(colors="#666666")
ax3.set_xticklabels([])
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")

plt.tight_layout(pad=1.5)
out_path2 = out_dir / "cluster_wab_distribution.png"
plt.savefig(str(out_path2), dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.close()
print(f"WAB distribution plot saved → {out_path2}")

# ── Save summary JSON ─────────────────────────────────────────────────────────

summary_path = out_dir / "cluster_summary.json"
with open(summary_path, "w") as f:
    json.dump(cluster_summaries, f, indent=2)
print(f"Cluster summary saved → {summary_path}")

print("\n── Suggested cluster labels for thesis ──────────────────────────────")
for c, s in cluster_summaries.items():
    print(f"  Cluster {c}: \"{s['suggested_label']}\"  "
          f"(WAB {s['mean_wab_aq']:.1f} ± {s['std_wab_aq']:.1f},  n={s['n']})")
print()