"""
RQ4 reanalysis — aphasia-only patients.

Reads:
  outputs/evaluation/improvements_DAPTA.npy
  outputs/evaluation/improvements_G_DDQN.npy
  outputs/evaluation/improvements_RBDE.npy
  outputs/evaluation/improvements_RTS.npy
  outputs/dae/patient_profiles.json
  outputs/dae/splits.json

Filters at the PATIENT level (not cluster level) using each patient's
aphasia_subtype field, then re-runs RQ4 on aphasia-only patients.

Usage:
    python experiments/rq4_aphasia_only.py
    python experiments/rq4_aphasia_only.py --data_dir outputs/evaluation --dae_dir outputs/dae
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

NON_APHASIA = {
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
}

METRICS      = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
METRIC_LABEL = ["CIU Rate", "MC Score", "MLU-m",         "MATTR", "SynComp"]
BENCHMARK_D  = 0.42


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def wilcoxon_gt(a, b):
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def bootstrap_ci(diff, n=2000, seed=42):
    rng  = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bonferroni(pvals, alpha=0.05):
    k = len(pvals)
    return [min(p * k, 1.0) for p in pvals], [min(p * k, 1.0) <= alpha for p in pvals]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, dae_dir: Path):

    # 1. Load improvement arrays
    dapta = np.load(data_dir / "improvements_DAPTA.npy")
    gddqn = np.load(data_dir / "improvements_G_DDQN.npy")
    rbde  = np.load(data_dir / "improvements_RBDE.npy")
    rts   = np.load(data_dir / "improvements_RTS.npy")

    n_total = len(dapta)

    for name, arr in [("DAPTA", dapta), ("G_DDQN", gddqn), ("RBDE", rbde), ("RTS", rts)]:
        if arr.ndim != 2 or arr.shape[1] != 5:
            print(f"[ERROR] {name} array has shape {arr.shape}, expected (N, 5).")
            print("        Re-run run_evaluation.py to regenerate improvement arrays.")
            return

    # 2. Load per-patient subtypes for the test set
    dae_data        = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    all_session_ids = list(dae_data["session_ids"])
    sid_to_idx      = {sid: i for i, sid in enumerate(all_session_ids)}

    with open(dae_dir / "splits.json") as f:
        test_ids = json.load(f)["test"]

    with open(dae_dir / "patient_profiles.json") as f:
        all_profiles = json.load(f)

    test_indices  = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    test_profiles = [all_profiles[i] for i in test_indices]

    if len(test_profiles) != n_total:
        print(
            f"[WARNING] test_profiles length ({len(test_profiles)}) != "
            f"improvements array length ({n_total}). Truncating to shorter."
        )
        min_n         = min(len(test_profiles), n_total)
        test_profiles = test_profiles[:min_n]
        dapta         = dapta[:min_n]
        gddqn         = gddqn[:min_n]
        rbde          = rbde[:min_n]
        rts           = rts[:min_n]
        n_total       = min_n

    # 3. Build aphasia mask directly from patient subtypes
    subtypes     = [p.get("aphasia_subtype", "Other") for p in test_profiles]
    aphasia_mask = np.array([s not in NON_APHASIA for s in subtypes])
    n_aphasia    = int(aphasia_mask.sum())
    n_excluded   = int((~aphasia_mask).sum())

    if n_aphasia == 0:
        print("[ERROR] No aphasia patients found after filtering.")
        print(f"        Unique subtypes in your data: {set(subtypes)}")
        print(f"        NON_APHASIA set: {NON_APHASIA}")
        return

    dapta_ap = dapta[aphasia_mask]
    gddqn_ap = gddqn[aphasia_mask]
    rbde_ap  = rbde[aphasia_mask]

    # Print subtype breakdown
    from collections import Counter
    subtype_counts = Counter(subtypes)

    SEP  = "─" * 72
    SEP2 = "═" * 72

    print()
    print(SEP2)
    print("  DAPTA RQ4 REANALYSIS — APHASIA-ONLY PATIENTS")
    print(f"  Patient-level filter: excluding {NON_APHASIA}")
    print(SEP2)

    # Section 1: Patient subtype audit
    print()
    print("SECTION 1 — PATIENT SUBTYPE BREAKDOWN")
    print(SEP)
    print(f"  {'Subtype':<28} {'n':>6}  {'Included?':>10}")
    print(SEP)
    for subtype, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        status = "✗ excluded" if subtype in NON_APHASIA else "✓ included"
        print(f"  {subtype:<28} {count:>6}  {status:>10}")
    print(SEP)
    print(
        f"  Aphasia patients : {n_aphasia}   |   "
        f"Excluded (non-aphasic) : {n_excluded}   |   "
        f"Total : {n_total}"
    )

    # Section 2: Pooled CIU
    print()
    print("SECTION 2 — POOLED RESULTS  (DAPTA vs G-DDQN, aphasia patients only)")
    print(SEP)

    ciu_d        = dapta_ap[:, 0]
    ciu_g        = gddqn_ap[:, 0]
    d_val        = cohens_d(ciu_d, ciu_g)
    p_val        = wilcoxon_gt(ciu_d, ciu_g)
    ci_lo, ci_hi = bootstrap_ci(ciu_d - ciu_g)
    var_red      = (
        (np.var(ciu_g, ddof=1) - np.var(ciu_d, ddof=1))
        / max(np.var(ciu_g, ddof=1), 1e-8) * 100
    )

    print(f"  n (aphasia-only)         : {n_aphasia}")
    print(f"  DAPTA  mean CIU          : {np.mean(ciu_d):.4f}")
    print(f"  G-DDQN mean CIU          : {np.mean(ciu_g):.4f}")
    print(
        f"  Cohen's d                : {d_val:+.3f}  "
        f"({'large' if abs(d_val)>=0.8 else 'medium' if abs(d_val)>=0.5 else 'small' if abs(d_val)>=0.2 else 'negligible'})"
    )
    print(f"  Wilcoxon p  (DAPTA>G)    : {p_val:.4f}  {'✓ significant' if p_val < 0.05 else '✗ not significant'}")
    print(f"  95% bootstrap CI (diff)  : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  iTalkBetter benchmark    : d = {BENCHMARK_D}")
    print(
        f"  Exceeds benchmark?       : "
        f"{'YES ✓' if d_val >= BENCHMARK_D else 'NO  (approaching)' if d_val >= BENCHMARK_D * 0.75 else 'NO'}"
    )
    print(f"  Variance reduction       : {var_red:+.1f}%")

    # Section 3: All metrics
    print()
    print("SECTION 3 — ALL METRICS  (aphasia-only, Bonferroni-corrected)")
    print(SEP)
    print(f"  {'Metric':<14} {'DAPTA':>8} {'G-DDQN':>8} {'RBDE':>8}   {'d':>8} {'p-corr':>8}  Benchmark?")
    print(SEP)

    raw_pvals = []
    rows      = []
    for mi, (metric, label) in enumerate(zip(METRICS, METRIC_LABEL)):
        d_arr = dapta_ap[:, mi]
        g_arr = gddqn_ap[:, mi]
        r_arr = rbde_ap[:, mi]
        d_val_m = cohens_d(d_arr, g_arr)
        p_raw   = wilcoxon_gt(d_arr, g_arr)
        raw_pvals.append(p_raw)
        rows.append((label, np.mean(d_arr), np.mean(g_arr), np.mean(r_arr), d_val_m))

    p_corr, sig = bonferroni(raw_pvals)

    for i, (label, d_mean, g_mean, r_mean, d_val_m) in enumerate(rows):
        bench = (
            "YES ✓"       if d_val_m >=  BENCHMARK_D else
            "YES ✓ (neg)" if d_val_m <= -BENCHMARK_D else
            "no"
        )
        print(
            f"  {label:<14} {d_mean:>8.4f} {g_mean:>8.4f} {r_mean:>8.4f}   "
            f"{d_val_m:>8.3f} {p_corr[i]:>8.4f}  {bench}"
        )

    print(SEP)
    print("  Positive d = DAPTA > G-DDQN")

    # Section 4: Original vs aphasia-only comparison
    orig_d_val = cohens_d(dapta[:, 0], gddqn[:, 0])
    orig_p     = wilcoxon_gt(dapta[:, 0], gddqn[:, 0])
    new_d_val  = cohens_d(dapta_ap[:, 0], gddqn_ap[:, 0])
    new_p      = wilcoxon_gt(dapta_ap[:, 0], gddqn_ap[:, 0])

    print()
    print(f"SECTION 4 — ORIGINAL (n={n_total}) vs APHASIA-ONLY (n={n_aphasia}) COMPARISON")
    print(SEP)
    print(f"  {'':28} {'Original (n='+str(n_total)+')':>18}  {'Aphasia-only (n='+str(n_aphasia)+')':>22}")
    print(SEP)
    print(f"  {'DAPTA mean CIU':<28} {float(np.mean(dapta[:,0])):>18.4f}  {float(np.mean(dapta_ap[:,0])):>22.4f}")
    print(f"  {'G-DDQN mean CIU':<28} {float(np.mean(gddqn[:,0])):>18.4f}  {float(np.mean(gddqn_ap[:,0])):>22.4f}")
    print(f"  {'Cohens d':<28} {orig_d_val:>18.3f}  {new_d_val:>22.3f}")
    print(f"  {'Wilcoxon p':<28} {orig_p:>18.4f}  {new_p:>22.4f}")
    print(SEP)

    print()
    print("  CONCLUSION:")
    print(
        f"  Original result (n={n_total}): d={orig_d_val:.3f}, p={orig_p:.3f}  "
        f"({'significant' if orig_p < 0.05 else 'not significant'})"
    )
    print(
        f"  Aphasia-only (n={n_aphasia}): d={new_d_val:.3f}, p={new_p:.4f}  "
        f"({'significant' if new_p < 0.05 else 'not significant'})"
    )
    print(f"  {n_excluded} non-aphasic patients excluded.")

    if new_d_val > orig_d_val and new_p < 0.05:
        print(
            f"  Removing controls strengthened the personalisation signal. "
            f"DAPTA outperforms G-DDQN on aphasia patients (d={new_d_val:+.3f})."
        )
    elif new_d_val > 0 and new_p < 0.05:
        print(
            f"  DAPTA outperforms G-DDQN on aphasia-only patients (d={new_d_val:+.3f}), "
            f"consistent with the overall result."
        )
    elif new_d_val > 0:
        print(
            f"  DAPTA shows a positive trend on aphasia-only patients (d={new_d_val:+.3f}) "
            f"but does not reach significance (p={new_p:.4f})."
        )
    else:
        print(
            f"  DAPTA does not outperform G-DDQN even on aphasia-only patients "
            f"(d={new_d_val:+.3f}, p={new_p:.4f})."
        )

    print()
    print(SEP2)
    print()

    # Save results
    results = {
        "n_total":    n_total,
        "n_aphasia":  n_aphasia,
        "n_excluded": n_excluded,
        "pooled": {
            "dapta_ciu_mean":    round(float(np.mean(ciu_d)), 4),
            "gddqn_ciu_mean":    round(float(np.mean(ciu_g)), 4),
            "cohens_d":          round(new_d_val, 3),
            "wilcoxon_p":        round(new_p, 4),
            "ci_95":             [round(ci_lo, 4), round(ci_hi, 4)],
            "significant":       bool(new_p < 0.05),
            "exceeds_benchmark": bool(new_d_val >= BENCHMARK_D),
        },
        "original_vs_aphasia_only": {
            "original_d":     round(orig_d_val, 3),
            "original_p":     round(orig_p, 4),
            "aphasia_only_d": round(new_d_val, 3),
            "aphasia_only_p": round(new_p, 4),
        },
    }

    out_path = data_dir / "rq4_aphasia_only_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="outputs/evaluation")
    parser.add_argument("--dae_dir",  default="outputs/dae")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.dae_dir))