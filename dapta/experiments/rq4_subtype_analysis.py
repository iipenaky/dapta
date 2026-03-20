import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

METRICS      = ["ciu_rate", "mc_score", "mlu_morphemes", "mattr", "syntactic_complexity"]
METRIC_LABEL = ["CIU Rate", "MC Score", "MLU-m",         "MATTR", "SynComp"]
BENCHMARK_D  = 0.42

APHASIA_SUBTYPES = ["Broca", "Wernicke", "Anomic", "Conduction", "Global", "Other"]

NON_APHASIA = {
    "control", "Control", "CONTROL",
    "NotAphasicByWAB", "NotAphasicByWab", "notaphasicbywab",
    "not_aphasic", "healthy",
}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def wilcoxon_gt(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.all(diff == 0) or len(diff) < 2:
        return 1.0
    try:
        _, p = stats.wilcoxon(diff, alternative="greater")
        return float(p)
    except ValueError:
        return 1.0


def bootstrap_ci(diff: np.ndarray, n: int = 2000, seed: int = 42):
    rng  = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bonferroni(pvals: List[float], alpha: float = 0.05):
    k = len(pvals)
    return [min(p * k, 1.0) for p in pvals], [min(p * k, 1.0) <= alpha for p in pvals]


def effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def load_test_profiles(dae_dir: Path) -> List[dict]:
    dae_data        = np.load(dae_dir / "state_vectors.npz", allow_pickle=True)
    all_session_ids = list(dae_data["session_ids"])
    sid_to_idx      = {sid: i for i, sid in enumerate(all_session_ids)}

    with open(dae_dir / "splits.json") as f:
        splits = json.load(f)

    with open(dae_dir / "patient_profiles.json") as f:
        all_profiles = json.load(f)

    test_ids     = splits["test"]
    test_indices = [sid_to_idx[sid] for sid in test_ids if sid in sid_to_idx]
    return [all_profiles[i] for i in test_indices]


def main(data_dir: Path, dae_dir: Path) -> None:

    dapta = np.load(data_dir / "improvements_DAPTA.npy")
    gddqn = np.load(data_dir / "improvements_G_DDQN.npy")

    for name, arr in [("DAPTA", dapta), ("G_DDQN", gddqn)]:
        if arr.shape[1] != 5:
            print(f"[ERROR] {name} array has shape {arr.shape}, expected (N, 5).")
            print("        Re-run run_evaluation.py to regenerate improvement arrays.")
            return

    n_total = len(dapta)


    test_profiles = load_test_profiles(dae_dir)

    if len(test_profiles) != n_total:
        min_n         = min(len(test_profiles), n_total)
        test_profiles = test_profiles[:min_n]
        dapta         = dapta[:min_n]
        gddqn         = gddqn[:min_n]
        n_total       = min_n
        print(f"[WARNING] Length mismatch — truncated to {min_n} patients.")

    subtypes = np.array([
        p.get("aphasia_subtype", "Other") for p in test_profiles
    ])


    aphasia_mask = np.array([s not in NON_APHASIA for s in subtypes])
    n_aphasia    = int(aphasia_mask.sum())
    n_controls   = int((~aphasia_mask).sum())

    dapta_ap   = dapta[aphasia_mask]
    gddqn_ap   = gddqn[aphasia_mask]
    subtypes_ap = subtypes[aphasia_mask]
    profiles_ap = [p for p, m in zip(test_profiles, aphasia_mask) if m]


    benefit_ciu = dapta_ap[:, 0] - gddqn_ap[:, 0]

    SEP  = "─" * 76
    SEP2 = "═" * 76

    print()
    print(SEP2)
    print("  RQ4 SUPPLEMENTARY: APHASIA SUBTYPE MODERATES PERSONALISATION BENEFIT")
    print(SEP2)
    print(f"  Total test patients : {n_total}")
    print(f"  Aphasia patients    : {n_aphasia}")
    print(f"  Controls excluded   : {n_controls}")

    print()
    print("SECTION 1 — PER-SUBTYPE PERSONALISATION BENEFIT (DAPTA vs G-DDQN)")
    print(SEP)
    print(
        f"  {'Subtype':<14} {'n':>4}   "
        f"{'DAPTA CIU':>10} {'G-DDQN CIU':>11} {'Benefit':>9} "
        f"{'d':>8}   {'p':>8}  Verdict"
    )
    print(SEP)

    subtype_results = {}
    raw_pvals       = []
    subtype_order   = []

    for subtype in APHASIA_SUBTYPES:
        mask = subtypes_ap == subtype
        n    = int(mask.sum())
        if n == 0:
            continue

        d_ciu    = dapta_ap[mask, 0]
        g_ciu    = gddqn_ap[mask, 0]
        benefit  = float(np.mean(d_ciu - g_ciu))
        d_val    = cohens_d(d_ciu, g_ciu)
        p_val    = wilcoxon_gt(d_ciu, g_ciu)

        raw_pvals.append(p_val)
        subtype_order.append(subtype)

        subtype_results[subtype] = {
            "n":               n,
            "dapta_ciu_mean":  round(float(np.mean(d_ciu)), 4),
            "gddqn_ciu_mean":  round(float(np.mean(g_ciu)), 4),
            "mean_benefit":    round(benefit, 4),
            "cohens_d":        round(d_val, 3),
            "effect_size":     effect_size_label(d_val),
            "p_raw":           round(p_val, 4),
            "clinically_meaningful": abs(d_val) >= BENCHMARK_D,
        }

    if raw_pvals:
        p_corr, sig = bonferroni(raw_pvals)
        for subtype, pc, s in zip(subtype_order, p_corr, sig):
            subtype_results[subtype]["p_corrected"] = round(pc, 4)
            subtype_results[subtype]["significant"]  = s

    for subtype in APHASIA_SUBTYPES:
        if subtype not in subtype_results:
            continue
        r = subtype_results[subtype]
        verdict = (
            "STRONG ✓"   if r["cohens_d"] >= 0.8          else
            "MODERATE ✓" if r["cohens_d"] >= BENCHMARK_D   else
            "WEAK"        if r["cohens_d"] >  0             else
            "NEGATIVE"
        )
        sig_marker = "*" if r.get("significant", False) else " "
        print(
            f"  {subtype:<14} {r['n']:>4}   "
            f"{r['dapta_ciu_mean']:>10.4f} {r['gddqn_ciu_mean']:>11.4f} "
            f"{r['mean_benefit']:>+9.4f} "
            f"{r['cohens_d']:>8.3f} {sig_marker} "
            f"{r.get('p_corrected', r['p_raw']):>8.4f}  {verdict}"
        )

    print(SEP)
    print("  * = Bonferroni-corrected p < 0.05   |   Positive benefit = DAPTA > G-DDQN")
    print()
    print("SECTION 2 — KRUSKAL-WALLIS TEST: DOES SUBTYPE MODERATE PERSONALISATION?")
    print(SEP)

    groups = [
        benefit_ciu[subtypes_ap == s]
        for s in APHASIA_SUBTYPES
        if (subtypes_ap == s).sum() >= 2
    ]
    group_names = [
        s for s in APHASIA_SUBTYPES
        if (subtypes_ap == s).sum() >= 2
    ]

    if len(groups) >= 2:
        stat, p_kw = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H = {stat:.3f}  p = {p_kw:.4f}")
        print(
            f"  {'✓ Subtype significantly moderates personalisation benefit' if p_kw < 0.05 else '✗ No significant moderation by subtype'}"
        )
        if p_kw < 0.05:
            print()
            print("  Post-hoc pairwise comparisons (Mann-Whitney U, Bonferroni):")
            pairs     = []
            pair_pvals = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    _, p_mw = stats.mannwhitneyu(
                        groups[i], groups[j], alternative="two-sided"
                    )
                    pairs.append((group_names[i], group_names[j]))
                    pair_pvals.append(float(p_mw))

            pair_pcorr, pair_sig = bonferroni(pair_pvals)
            for (s1, s2), pc, s in zip(pairs, pair_pcorr, pair_sig):
                if s:
                    print(f"    {s1} vs {s2}: p_corr = {pc:.4f} *")
    else:
        print("  Insufficient subtype groups (need at least 2 with n>=2) for Kruskal-Wallis.")
        stat, p_kw = None, None

    print()
    print("SECTION 3 — WAB-AQ MODERATES BENEFIT WITHIN EACH SUBTYPE")
    print(SEP)
    print(f"  {'Subtype':<14} {'n':>4}  {'r (WAB-AQ)':>12}  {'p':>8}  Interpretation")
    print(SEP)

    wab_corr_results = {}
    for subtype in APHASIA_SUBTYPES:
        mask = subtypes_ap == subtype
        n    = int(mask.sum())
        if n < 5:
            continue

        wab_vals = np.array([
            float(p.get("wab_aq") or 55.0)
            for p, m in zip(profiles_ap, mask) if m
        ])
        ben_vals = benefit_ciu[mask]

        r, p_r = stats.pearsonr(wab_vals, ben_vals)
        interp  = (
            "higher severity benefits more"  if r < -0.3 and p_r < 0.05 else
            "lower severity benefits more"   if r >  0.3 and p_r < 0.05 else
            "WAB-AQ does not moderate"
        )
        wab_corr_results[subtype] = {
            "n": n, "r": round(float(r), 3), "p": round(float(p_r), 4),
            "interpretation": interp,
        }
        print(
            f"  {subtype:<14} {n:>4}  {r:>+12.3f}  {p_r:>8.4f}  {interp}"
        )

    if not wab_corr_results:
        print("  No subtype has n >= 5 for within-subtype WAB-AQ correlation.")

    print(SEP)
    print()
    print("SECTION 4 — SUBTYPE RANKING BY PERSONALISATION BENEFIT (CIU rate)")
    print(SEP)
    print(f"  {'Rank':<6} {'Subtype':<14} {'Mean benefit':>14}  {'Cohen d':>9}  {'Effect'}")
    print(SEP)

    ranked = sorted(
        subtype_results.items(),
        key=lambda x: x[1]["mean_benefit"],
        reverse=True,
    )
    for rank, (subtype, r) in enumerate(ranked, 1):
        print(
            f"  {rank:<6} {subtype:<14} {r['mean_benefit']:>+14.4f}  "
            f"{r['cohens_d']:>9.3f}  {r['effect_size']}"
        )

    print(SEP)
    print()
    print("  INTERPRETATION FOR THESIS:")
    if ranked:
        top    = ranked[0][0]
        bottom = ranked[-1][0]
        top_d  = ranked[0][1]["cohens_d"]
        print(
            f"  {top} patients show the largest personalisation benefit "
            f"(d={top_d:.3f}), suggesting that patient-specific RL "
            f"sequencing is most advantageous for this subtype."
        )
        if ranked[-1][1]["cohens_d"] < 0:
            print(
                f"  {bottom} patients show a negative effect, indicating "
                f"that G-DDQN may be preferable for this subtype, possibly "
                f"due to limited training data in this cluster."
            )
    print()
    print(SEP2)
    print()
    out = {
        "n_total":          n_total,
        "n_aphasia":        n_aphasia,
        "n_controls_excluded": n_controls,
        "per_subtype":      subtype_results,
        "kruskal_wallis": {
            "H":           round(float(stat), 3) if stat is not None else None,
            "p":           round(float(p_kw), 4) if p_kw is not None else None,
            "significant": bool(p_kw < 0.05)     if p_kw is not None else None,
        },
        "wabaq_within_subtype": wab_corr_results,
        "subtype_ranking": [
            {"subtype": s, "mean_benefit": r["mean_benefit"], "cohens_d": r["cohens_d"]}
            for s, r in ranked
        ],
    }

    out_path = data_dir / "rq4_subtype_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ4 subtype moderation analysis"
    )
    parser.add_argument(
        "--data_dir", default="outputs/evaluation",
        help="Folder containing improvements_DAPTA.npy and improvements_G_DDQN.npy"
    )
    parser.add_argument(
        "--dae_dir", default="outputs/dae",
        help="Folder containing patient_profiles.json and splits.json"
    )
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.dae_dir))