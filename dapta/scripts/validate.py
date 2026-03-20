"""
experiments/plot_inferred_actions.py
-------------------------------------
Generates a heatmap showing which therapy action is most aligned with
each aphasia subtype's deficit profile.

Rows    : Aphasia subtypes (Broca, Wernicke, Anomic, Conduction, Global, Other)
Columns : Therapy actions  (SFA, SVO, CILT, Script Training, Phonological Cue)
Values  : Proportion of patients in each subtype assigned each therapy

State vector layout (from state_builder.py):
  [0:25]  Discourse block — 5 metrics × 5 tasks
          per task: [ciu_rate, mc_score, mlu_morphemes, syn_comp, mattr]
          cookie_theft[0:5], cinderella[5:10], sandwich[10:15],
          stroke_narrative[15:20], conversation[20:25]
  [25:30] Task presence flags
  [30:36] Aphasia subtype one-hot: Broca/Wernicke/Anomic/Conduction/Global/Other
  [36:40] Static: wab_aq, wab_aq_known, mean_surprisal, log_session_num
  [40:47] Signal: maze_rate, utt_length_std×5, mean_pause_ms

Usage:
    python experiments/plot_inferred_actions.py
    python experiments/plot_inferred_actions.py --dae_dir outputs/dae --out_dir outputs/figures
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------------------
# State vector constants (mirrors state_builder.py)
# ---------------------------------------------------------------------------

METRICS        = ["ciu_rate", "mc_score", "mlu_morphemes", "syn_comp", "mattr"]
METRIC_LABELS  = ["CIU Rate", "MC Score", "MLU-m", "Syn. Comp.", "MATTR"]
N_METRICS      = 5
N_TASKS        = 5
SUBTYPES       = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]
SLICE_DISCOURSE = slice(0, 25)   # 5 metrics × 5 tasks
SLICE_SUBTYPE   = slice(30, 36)  # one-hot: Broca/Wernicke/Anomic/Conduction/Global/Other

NON_APHASIA = {
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
}

# ---------------------------------------------------------------------------
# Therapy prior vectors — shape (5,) matching METRICS order
# [ciu_rate, mc_score, mlu_morphemes, syn_comp, mattr]
#
# Values represent the expected *improvement* direction for each metric.
# Higher = this therapy strongly targets that metric.
# Based on Section 3.5.3 priors.
# ---------------------------------------------------------------------------

THERAPY_PRIORS = {
    "SFA\n(Naming)":      [0.8, 0.1, 0.1, 0.0, 0.2],  # CIU primary
    "SVO\n(Syntax)":      [0.0, 0.0, 0.5, 0.9, 0.0],  # SynComp primary, MLU secondary
    "CILT\n(Discourse)":  [0.5, 0.9, 0.2, 0.0, 0.1],  # MC primary, CIU secondary
    "Script\nTraining":   [0.1, 0.2, 0.8, 0.1, 0.6],  # MLU primary, MATTR secondary
    "Phonological\nCue":  [0.7, 0.0, 0.0, 0.0, 0.7],  # CIU + MATTR
}

THERAPY_NAMES = list(THERAPY_PRIORS.keys())
PRIOR_MATRIX  = np.array(list(THERAPY_PRIORS.values()), dtype=np.float32)  # (5, 5)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_discourse_means(state_vectors: np.ndarray) -> np.ndarray:
    """
    Extract mean per discourse metric across all tasks.

    Parameters
    ----------
    state_vectors : (N, 47)

    Returns
    -------
    (N, 5) — one mean value per metric [ciu, mc, mlu, syn, mattr]
    """
    N          = state_vectors.shape[0]
    disc_means = np.zeros((N, N_METRICS), dtype=np.float32)

    for m in range(N_METRICS):
        # Dims for metric m across all 5 tasks
        dims = [t * N_METRICS + m for t in range(N_TASKS)]
        disc_means[:, m] = state_vectors[:, dims].mean(axis=1)

    return disc_means


def get_subtype_from_vector(state_vector: np.ndarray) -> str:
    """Read aphasia subtype from one-hot encoding in state vector."""
    one_hot = state_vector[SLICE_SUBTYPE]
    idx     = int(np.argmax(one_hot))
    return SUBTYPES[idx]


# ---------------------------------------------------------------------------
# Action inference
# ---------------------------------------------------------------------------

def infer_best_therapy(deficit_vector: np.ndarray) -> str:
    """
    Given a patient's deficit vector (1 - metric_means),
    find the therapy whose prior is most cosine-similar to the deficit.

    Parameters
    ----------
    deficit_vector : (5,) — values in [0, 1], higher = bigger deficit

    Returns
    -------
    Name of the best-matching therapy
    """
    # Clip to avoid negative values confusing cosine similarity
    deficit = np.clip(deficit_vector, 0.0, 1.0).reshape(1, -1)

    if np.all(deficit == 0):
        return THERAPY_NAMES[0]  # fallback

    sims = cosine_similarity(deficit, PRIOR_MATRIX)[0]  # (5,)
    return THERAPY_NAMES[int(np.argmax(sims))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dae_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load state vectors
    npz             = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    state_vectors   = npz["state_vectors"]          # (N_sessions, 47)
    all_session_ids = list(npz["session_ids"])
    sid_to_idx      = {sid: i for i, sid in enumerate(all_session_ids)}

    print(f"Loaded {len(all_session_ids)} sessions, state_dim={state_vectors.shape[1]}")

    # 2. Load patient profiles to get subtypes
    with open(dae_dir / "patient_profiles.json") as f:
        all_profiles = json.load(f)

    # 3. Aggregate sessions to per-patient means
    #    Key: participant_id -> list of session indices
    patient_sessions: dict = defaultdict(list)
    for sid, idx in sid_to_idx.items():
        # Session IDs follow pattern like 'adler01a' — strip trailing digits/letter
        import re
        pid = re.sub(r'\d+[a-z]?$', '', str(sid).lower())
        patient_sessions[pid].append(idx)

    print(f"Unique patients: {len(patient_sessions)}")

    # 4. Build per-patient feature rows
    #    We also need subtype — read from profile or state vector one-hot
    profiles_by_pid: dict = {}
    for profile in all_profiles:
        pid = re.sub(r'\d+[a-z]?$', '', str(profile.get("participant_id", "")).lower())
        profiles_by_pid[pid] = profile

    rows = []
    for pid, indices in patient_sessions.items():
        sv_patient  = state_vectors[indices]             # (n_sessions, 47)
        disc_means  = extract_discourse_means(sv_patient).mean(axis=0)  # (5,)

        # Get subtype from profile (more reliable than one-hot)
        profile  = profiles_by_pid.get(pid, {})
        subtype  = profile.get("aphasia_subtype", "Other")

        # Skip non-aphasic patients
        if subtype in NON_APHASIA:
            continue

        # Normalise subtype to known list
        subtype_map = {
            "broca": "Broca", "brocas": "Broca",
            "wernicke": "Wernicke", "wernickes": "Wernicke",
            "anomic": "Anomic", "anomia": "Anomic",
            "conduction": "Conduction",
            "global": "Global",
        }
        subtype = subtype_map.get(subtype.lower().strip(), "Other")

        # Infer best therapy from deficit profile
        deficit         = 1.0 - disc_means
        best_therapy    = infer_best_therapy(deficit)

        rows.append({
            "pid":          pid,
            "subtype":      subtype,
            "best_therapy": best_therapy,
            **{m: float(disc_means[i]) for i, m in enumerate(METRICS)},
        })

    print(f"Aphasia patients for analysis: {len(rows)}")
    if len(rows) == 0:
        print("[ERROR] No aphasia patients found. Check dae_dir path.")
        return

    # 5. Build heatmap matrix: rows=subtypes, cols=therapies
    #    Values = proportion of patients in each subtype assigned each therapy
    count_matrix = np.zeros((len(SUBTYPES), len(THERAPY_NAMES)), dtype=np.float32)

    for row in rows:
        r = SUBTYPES.index(row["subtype"]) if row["subtype"] in SUBTYPES else SUBTYPES.index("Other")
        c = THERAPY_NAMES.index(row["best_therapy"])
        count_matrix[r, c] += 1

    # Normalise rows to proportions
    row_sums         = count_matrix.sum(axis=1, keepdims=True)
    row_sums         = np.where(row_sums == 0, 1, row_sums)  # avoid div by zero
    proportion_matrix = count_matrix / row_sums

    # Count per subtype for axis labels
    subtype_counts = {s: sum(1 for r in rows if r["subtype"] == s) for s in SUBTYPES}

    # 6. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(proportion_matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(THERAPY_NAMES)))
    ax.set_xticklabels(THERAPY_NAMES, fontsize=10)
    ax.set_yticks(range(len(SUBTYPES)))
    ax.set_yticklabels(
        [f"{s}\n(n={subtype_counts.get(s, 0)})" for s in SUBTYPES],
        fontsize=10,
    )

    # Annotate cells
    for r in range(len(SUBTYPES)):
        for c in range(len(THERAPY_NAMES)):
            val = proportion_matrix[r, c]
            if val > 0:
                ax.text(c, r, f"{val:.0%}",
                        ha="center", va="center", fontsize=9,
                        fontfamily="monospace",
                        color="white" if val > 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Proportion of patients", fontsize=9)
    ax.set_title(
        "Inferred Therapy Action by Aphasia Subtype\n"
        "(based on deficit profile cosine similarity)",
        pad=12,
    )

    # Right: mean deficit profile per subtype
    ax2 = axes[1]
    subtype_disc = {s: [] for s in SUBTYPES}
    for row in rows:
        subtype_disc[row["subtype"]].append(
            [row[m] for m in METRICS]
        )

    deficit_means = np.array([
        np.mean(subtype_disc[s], axis=0) if subtype_disc[s] else np.zeros(N_METRICS)
        for s in SUBTYPES
    ])  # (6, 5)

    im2 = ax2.imshow(1 - deficit_means, cmap="RdYlGn_r", aspect="auto",
                     vmin=0, vmax=1)
    ax2.set_xticks(range(N_METRICS))
    ax2.set_xticklabels(METRIC_LABELS, fontsize=9, rotation=30, ha="right")
    ax2.set_yticks(range(len(SUBTYPES)))
    ax2.set_yticklabels(SUBTYPES, fontsize=10)

    for r in range(len(SUBTYPES)):
        for c in range(N_METRICS):
            val = 1 - deficit_means[r, c]
            ax2.text(c, r, f"{val:.2f}",
                     ha="center", va="center", fontsize=8,
                     fontfamily="monospace",
                     color="white" if val > 0.7 else "black")

    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("Deficit severity\n(darker = worse)", fontsize=8)
    ax2.set_title("Mean Deficit Profile\nper Subtype", pad=12)

    fig.suptitle(
        "Figure 9 — Therapy Action Inference from Patient Deficit Profiles",
        fontsize=13,
    )
    plt.tight_layout()

    path = out_dir / "fig9_inferred_actions_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save CSV
    import csv
    csv_path = out_dir.parent / "evaluation" / "inferred_actions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument("--dae_dir", default="outputs/dae")
    parser.add_argument("--out_dir", default="outputs/figures")
    args = parser.parse_args()
    main(Path(args.dae_dir), Path(args.out_dir))