import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import re

import numpy as np
import pandas as pd
from scipy import stats

from dapta.dae.parser import CHATParser
from dapta.dae.metrics import DiscourseMetricExtractor
from dapta.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/validate_against_clan.log")


METRIC_MAP = [
    (
        "mlu_morphemes", "MLU_Morphemes",
        "MLU in morphemes",
        0.80, None,
    ),
    (
        "mattr", "FREQ_TTR",
        "MATTR (auto) vs raw TTR (CLAN)",
        0.80,
        "NOTE: MATTR and raw TTR are different measures. MATTR is length-robust; "
        "raw TTR is length-dependent. Lower r here does not mean the computation "
        "is wrong — it means the constructs differ.",
    ),
    (
        "wpm", "Words_Min",
        "Words per minute",
        0.90,
        None,
    ),
    (
        "syntactic_complexity", "Verbs_Utt",
        "Syntactic complexity vs verbs/utterance",
        0.65,
        "NOTE: auto metric = proportion of utterances with subordinate clause; "
        "CLAN Verbs_Utt = mean verbs per utterance. These are correlated but "
        "not equivalent.",
    ),
    (
        "maze_rate", "clan_repair_rate",
        "Maze rate vs CLAN repair rate",
        0.75, None,
    ),
]

WAB_AQ_COL = "wab_aq"

WAB_METRICS = [
    ("mlu_morphemes",        "MLU_Morphemes",   "auto_mlu"),
    ("mattr",                "FREQ_TTR",        "auto_mattr"),
    ("syntactic_complexity", "Verbs_Utt",       "auto_syn_comp"),
    ("maze_rate",            "clan_repair_rate", "auto_maze_rate"),
    ("wpm",                  "Words_Min",       "auto_wpm"),
]


def normalise_id(path) -> str:
    stem = Path(str(path)).stem
    stem = Path(stem).stem
    return stem.lower().strip()

def extract_wabaq_from_cha(cha_dir: str, session_ids: set) -> pd.DataFrame:
    cha_dir = Path(cha_dir)
    parser  = CHATParser(participant_tier="PAR")
    rows    = []

    for cha in sorted(cha_dir.rglob("*.cha")):
        sid = cha.stem.lower()
        if sid not in session_ids:
            continue
        try:
            transcript = parser.parse_file(cha)
            wab_aq = transcript.metadata.get("wab_aq", None)
            if wab_aq is not None:
                try:
                    wab_aq = float(wab_aq)
                except (ValueError, TypeError):
                    wab_aq = None

            rows.append({
                "_session_id": sid,
                "wab_aq":      wab_aq,
            })
            status = f"{wab_aq:.1f}" if wab_aq is not None else "not found"
            logger.info(f"  {cha.name}: WAB-AQ = {status}")

        except Exception as e:
            logger.warning(f"  Failed to parse {cha.name}: {e}")

    df = pd.DataFrame(rows)
    n_found = df["wab_aq"].notna().sum()
    logger.info(f"WAB-AQ extracted for {n_found} / {len(df)} matched transcripts")
    return df

def read_clan_eval(path: Path) -> Optional[Dict]:
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
        ns_match = re.match(r"\{.*\}", root.tag)
        ns = ns_match.group(0) if ns_match else ""

        rows_data = []
        for worksheet in root.iter(f"{ns}Worksheet"):
            for table in worksheet.iter(f"{ns}Table"):
                for row in table.iter(f"{ns}Row"):
                    cells = []
                    for cell in row.iter(f"{ns}Cell"):
                        data = cell.find(f"{ns}Data")
                        cells.append(data.text if data is not None else "")
                    rows_data.append(cells)
            break

        if not rows_data or len(rows_data) < 2:
            logger.warning(f"No data rows in {path.name}")
            return None

        headers = [str(h).strip() for h in rows_data[0]]
        values  = rows_data[1]
        result  = {}
        for header, value in zip(headers, values):
            if not header:
                continue
            try:
                result[header] = float(str(value).replace(",", "."))
            except (ValueError, TypeError):
                result[header] = value
        return result or None

    except ET.ParseError as e:
        logger.warning(f"XML parse error in {path.name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error reading {path.name}: {e}")
        return None

def load_all_clan_evals(clan_dir: str) -> pd.DataFrame:
    clan_dir   = Path(clan_dir)
    eval_files = sorted(clan_dir.rglob("*.eval.xls")) + \
                 sorted(clan_dir.rglob("*.eval.xlsx"))

    if not eval_files:
        raise FileNotFoundError(f"No *.eval.xls files found in {clan_dir}.")

    logger.info(f"Found {len(eval_files)} CLAN eval files in {clan_dir}")

    rows = []
    for f in eval_files:
        metrics = read_clan_eval(f)
        if metrics is None:
            continue
        metrics["_session_id"] = normalise_id(f)
        rows.append(metrics)

    if not rows:
        raise ValueError("Could not parse any CLAN eval files.")

    df = pd.DataFrame(rows)
    logger.info(f"Successfully loaded CLAN scores for {len(df)} transcripts")

    if {"retracing", "repetition", "Total_Utts"}.issubset(df.columns):
        df["clan_repair_rate"] = (
            (df["retracing"].fillna(0) + df["repetition"].fillna(0))
            / df["Total_Utts"].replace(0, np.nan)
        ).fillna(0.0)

    return df



def compute_auto_metrics(
    cha_dir: str,
    clan_session_ids: set,
    clan_df: pd.DataFrame,
) -> pd.DataFrame:
    cha_dir     = Path(cha_dir)
    all_cha     = sorted(cha_dir.rglob("*.cha"))
    matched_cha = [f for f in all_cha if f.stem.lower() in clan_session_ids]

    logger.info(
        f"Found {len(all_cha)} .cha files, "
        f"{len(matched_cha)} match a CLAN eval file"
    )

    if not matched_cha:
        raise FileNotFoundError(
            "No .cha files matched the CLAN eval session IDs.\n"
            f"  CLAN IDs (sample): {sorted(clan_session_ids)[:5]}\n"
            f"  .cha stems (sample): {[f.stem.lower() for f in all_cha[:5]]}"
        )

    duration_lookup = {}
    if "Duration_(sec)" in clan_df.columns:
        for _, row in clan_df.iterrows():
            try:
                duration_lookup[row["_session_id"]] = (
                    float(row["Duration_(sec)"]) / 60.0
                )
            except (ValueError, TypeError):
                pass

    TASK_PRIORITY = [
        "cookie_theft", "cinderella", "sandwich",
        "stroke_narrative", "conversation",
    ]

    parser = CHATParser(participant_tier="PAR")
    rows   = []

    for cha in matched_cha:
        try:
            transcript = parser.parse_file(cha)
            best       = None
            best_task  = None

            for task in TASK_PRIORITY:
                if not transcript.has_task(task):
                    continue
                raw_utts   = [u.raw  for u in transcript.tasks[task] if u.text.strip()]
                clean_utts = [u.text for u in transcript.tasks[task] if u.text.strip()]
                if not clean_utts:
                    continue

                real_duration = duration_lookup.get(cha.stem.lower(), 0.0)
                dur_source    = "clan" if real_duration > 0 else "fallback"

                best = DiscourseMetricExtractor(
                    task=task,
                    duration_minutes=real_duration,
                ).compute(clean_utts, raw_utterances=raw_utts)
                best_task = task
                break

            if best is None:
                logger.warning(f"No usable utterances in {cha.name}")
                continue

            rows.append({
                "_session_id":       cha.stem.lower(),
                "task":              best_task,
                "duration_source":   dur_source,
                "auto_mlu":          best.mlu_morphemes,
                "auto_mattr":        best.mattr,
                "auto_ciu_rate":     best.ciu_rate,
                "auto_wpm":          best.wpm,
                "auto_syn_comp":     best.syntactic_complexity,
                "auto_mc_score":     best.mc_score,
                "auto_maze_rate":    best.maze_rate,
                "auto_n_utterances": best.n_utterances,
                "auto_n_words":      best.n_words,
            })

        except Exception as e:
            logger.warning(f"Failed to process {cha.name}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"Auto-scored {len(df)} transcripts")
    return df

def _correlate_pair(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    if len(x) < 3:
        return None
    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    mae  = float(np.mean(np.abs(x - y)))
    bias = float(np.mean(x - y))
    return {
        "n_pairs":    int(len(x)),
        "pearson_r":  round(float(pearson_r),  4),
        "pearson_p":  round(float(pearson_p),  6),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 6),
        "mae":        round(mae,  4),
        "bias":       round(bias, 4),
        "auto_mean":  round(float(np.mean(x)), 4),
        "clan_mean":  round(float(np.mean(y)), 4),
        "threshold":  threshold,
        "acceptable": int(abs(pearson_r) >= threshold),
    }


def _correlate_with_wabaq(values: np.ndarray, wab_aq: np.ndarray, label: str) -> dict:
    if len(values) < 5:
        return None

    r, p = stats.pearsonr(values, wab_aq)
    rng = np.random.default_rng(42)
    boot_rs = []
    n = len(values)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        boot_r, _ = stats.pearsonr(values[idx], wab_aq[idx])
        boot_rs.append(boot_r)
    boot_rs = np.array(boot_rs)
    ci_low  = float(np.percentile(boot_rs, 2.5))
    ci_high = float(np.percentile(boot_rs, 97.5))

    return {
        "label":       label,
        "n":           int(len(values)),
        "r":           round(float(r), 3),
        "p_value":     round(float(p), 4),
        "ci_95_low":   round(ci_low,  3),
        "ci_95_high":  round(ci_high, 3),
        "significant": bool(p < 0.05),
    }

def run_validation(clan_df: pd.DataFrame, auto_df: pd.DataFrame):
    merged     = pd.merge(clan_df, auto_df, on="_session_id", how="inner")
    n_match    = len(merged)
    unmatched  = sorted(set(clan_df["_session_id"]) - set(auto_df["_session_id"]))
    n_fallback = int((merged.get("duration_source", pd.Series()) == "fallback").sum())

    logger.info(
        f"Matched {n_match} / {len(clan_df)} transcripts "
        f"({len(unmatched)} unmatched, {n_fallback} used fallback duration)"
    )

    AUTO_COL = {
        "mlu_morphemes":        "auto_mlu",
        "mattr":                "auto_mattr",
        "wpm":                  "auto_wpm",
        "syntactic_complexity": "auto_syn_comp",
        "maze_rate":            "auto_maze_rate",
    }

    results = {
        "n_clan_files":               len(clan_df),
        "n_auto_scored":              len(auto_df),
        "n_matched":                  n_match,
        "n_fallback_duration":        n_fallback,
        "unmatched_ids":              unmatched,
        "metrics":                    {},
        "metrics_clan_duration_only": {},
        "per_task":                   {},
    }

    for auto_attr, clan_col, label, threshold, note in METRIC_MAP:
        if clan_col not in merged.columns:
            logger.warning(f"Column '{clan_col}' not in CLAN data — skipping {label}")
            continue

        auto_col = AUTO_COL[auto_attr]
        pair     = merged[[auto_col, clan_col]].dropna()
        x = pair[auto_col].values.astype(float)
        y = pair[clan_col].values.astype(float)

        entry = _correlate_pair(x, y, threshold)
        if entry is None:
            continue
        entry["label"]       = label
        entry["clan_column"] = clan_col
        if note:
            entry["note"] = note
        results["metrics"][auto_attr] = entry

        if "duration_source" in merged.columns:
            clan_only = merged[merged["duration_source"] == "clan"]
            pair_c    = clan_only[[auto_col, clan_col]].dropna()
            if len(pair_c) >= 3:
                xc = pair_c[auto_col].values.astype(float)
                yc = pair_c[clan_col].values.astype(float)
                entry_c = _correlate_pair(xc, yc, threshold)
                if entry_c:
                    entry_c["label"] = label + " (clan duration only)"
                    results["metrics_clan_duration_only"][auto_attr] = entry_c

        logger.info(
            f"  {label:<50}  r={entry['pearson_r']:.3f}  "
            f"{'OK' if entry['acceptable'] else 'needs work'} (threshold={threshold})"
        )

    if "task" in merged.columns:
        for task in merged["task"].dropna().unique():
            task_rows = merged[merged["task"] == task]
            results["per_task"][task] = {}
            for auto_attr, clan_col, label, threshold, _ in METRIC_MAP:
                if clan_col not in task_rows.columns:
                    continue
                auto_col = AUTO_COL[auto_attr]
                pair     = task_rows[[auto_col, clan_col]].dropna()
                if len(pair) < 3:
                    continue
                x = pair[auto_col].values.astype(float)
                y = pair[clan_col].values.astype(float)
                entry = _correlate_pair(x, y, threshold)
                if entry:
                    results["per_task"][task][auto_attr] = entry

    return results, merged

def run_wabaq_analysis(merged: pd.DataFrame) -> dict:
    if WAB_AQ_COL not in merged.columns:
        logger.warning(
            f"WAB-AQ column '{WAB_AQ_COL}' not found in merged data. "
            f"Skipping WAB-AQ analysis."
        )
        return {}

    wab_aq_raw   = merged[WAB_AQ_COL].values.astype(float)
    valid_mask   = ~np.isnan(wab_aq_raw)
    wab_aq       = wab_aq_raw[valid_mask]
    merged_valid = merged[valid_mask].copy()

    logger.info(
        f"WAB-AQ available for {valid_mask.sum()} / {len(merged)} matched transcripts"
    )

    if valid_mask.sum() < 5:
        logger.warning("Fewer than 5 patients with WAB-AQ — skipping WAB-AQ analysis.")
        return {}

    results = {
        "n_with_wabaq": int(valid_mask.sum()),
        "wab_aq_mean":  round(float(np.mean(wab_aq)), 2),
        "wab_aq_std":   round(float(np.std(wab_aq)),  2),
        "step2_metric_wabaq_correlations": {},
        "step3_clan_vs_auto_wabaq":        {},
    }

    for metric_attr, clan_col, auto_col in WAB_METRICS:
        if auto_col in merged_valid.columns:
            auto_vals = merged_valid[auto_col].values.astype(float)
            auto_mask = ~np.isnan(auto_vals)
            if auto_mask.sum() >= 5:
                entry = _correlate_with_wabaq(
                    auto_vals[auto_mask],
                    wab_aq[auto_mask],
                    label=f"AUTO {metric_attr} vs WAB-AQ",
                )
                if entry:
                    results["step2_metric_wabaq_correlations"][metric_attr] = entry
        if clan_col in merged_valid.columns and auto_col in merged_valid.columns:
            pair = merged_valid[[clan_col, auto_col]].copy()
            pair[WAB_AQ_COL] = wab_aq
            pair = pair.dropna()

            if len(pair) >= 5:
                clan_vals = pair[clan_col].values.astype(float)
                auto_vals = pair[auto_col].values.astype(float)
                wab_vals  = pair[WAB_AQ_COL].values.astype(float)

                clan_entry = _correlate_with_wabaq(
                    clan_vals, wab_vals,
                    label=f"CLAN {clan_col} vs WAB-AQ",
                )
                auto_entry = _correlate_with_wabaq(
                    auto_vals, wab_vals,
                    label=f"AUTO {metric_attr} vs WAB-AQ",
                )

                if clan_entry and auto_entry:
                    diff = round(auto_entry["r"] - clan_entry["r"], 3)
                    results["step3_clan_vs_auto_wabaq"][metric_attr] = {
                        "clan":       clan_entry,
                        "auto":       auto_entry,
                        "r_diff":     diff,
                        "conclusion": (
                            "automation preserves functional signal"
                            if abs(diff) < 0.10
                            else "automation degrades functional signal"
                        ),
                    }

    return results

def make_scatter_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping scatter plots.")
        return

    AUTO_COL_MAP = {
        "mlu_morphemes":        "auto_mlu",
        "mattr":                "auto_mattr",
        "wpm":                  "auto_wpm",
        "syntactic_complexity": "auto_syn_comp",
        "maze_rate":            "auto_maze_rate",
    }

    plot_specs = [
        (AUTO_COL_MAP[a], clan_col, label, threshold)
        for a, clan_col, label, threshold, _ in METRIC_MAP
        if AUTO_COL_MAP.get(a) in merged.columns and clan_col in merged.columns
    ]

    if not plot_specs:
        return

    n         = len(plot_specs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    has_source = "duration_source" in merged.columns
    colors_map = {"clan": "#1D9E75", "fallback": "#D85A30"}

    for ax, (auto_col, clan_col, label, threshold) in zip(axes, plot_specs):
        pair = merged[["_session_id", auto_col, clan_col]].dropna()
        if pair.empty:
            continue

        if has_source:
            pair = pair.join(merged[["duration_source"]], how="left")
            for src, grp in pair.groupby(
                pair.get("duration_source", pd.Series("clan", index=pair.index))
            ):
                ax.scatter(
                    grp[auto_col], grp[clan_col],
                    alpha=0.7, s=70, zorder=3,
                    color=colors_map.get(str(src), "#888780"),
                    label=f"{src} duration",
                )
            ax.legend(fontsize=7)
        else:
            ax.scatter(pair[auto_col], pair[clan_col],
                       alpha=0.7, s=70, color="#1D9E75", zorder=3)

        x, y = pair[auto_col].values, pair[clan_col].values
        for _, row in pair.iterrows():
            ax.annotate(
                row["_session_id"],
                (row[auto_col], row[clan_col]),
                fontsize=6, alpha=0.6,
                xytext=(4, 4), textcoords="offset points",
            )

        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="perfect agreement")
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            ax.plot([lo, hi], [m*lo+b, m*hi+b],
                    color="#D85A30", lw=1.2, label="regression")

        r, _ = stats.pearsonr(x, y)
        ok    = abs(r) >= threshold
        ax.set_xlabel(f"Automatic ({auto_col})", fontsize=9)
        ax.set_ylabel(f"CLAN ({clan_col})",       fontsize=9)
        ax.set_title(
            f"{label}\nr = {r:.3f}  threshold ≥ {threshold} "
            f"{'✓' if ok else '✗'}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Automatic vs CLAN gold standard  (n={len(merged)} transcripts)",
        fontsize=12,
    )
    fig.tight_layout()
    out = output_dir / "scatter_plots.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    logger.info(f"Scatter plots saved to {out}")
    plt.close(fig)


def print_rq1_report(validation_results: dict, wabaq_results: dict) -> None:
    print("\n" + "=" * 76)
    print("RQ1 FULL ANSWER")
    print("=" * 76)

    print("\nSTEP 1: Can the LLM compute metrics accurately? (vs CLAN gold standard)")
    print(f"  {'Metric':<35} {'r':>7} {'ρ':>7} {'MAE':>8} {'Pass?':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
    for attr, r in validation_results.get("metrics", {}).items():
        print(
            f"  {attr:<35} {r['pearson_r']:>7.3f} {r['spearman_r']:>7.3f} "
            f"{r['mae']:>8.3f} {'  ✓' if r['acceptable'] else '  ✗':>6}"
        )

    if not wabaq_results:
        print("\n  WAB-AQ analysis not available (not found in .cha headers).")
        print("=" * 76)
        return

    print(f"\nSTEP 2: Which metrics reflect functional ability? (correlation with WAB-AQ)")
    print(f"  n = {wabaq_results['n_with_wabaq']} patients  |  "
          f"WAB-AQ mean = {wabaq_results['wab_aq_mean']}  "
          f"std = {wabaq_results['wab_aq_std']}")
    print(f"\n  {'Metric':<25} {'r':>7} {'p':>8} {'95% CI':>18} {'Sig?':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*18} {'-'*6}")
    for metric, entry in wabaq_results.get("step2_metric_wabaq_correlations", {}).items():
        ci  = f"[{entry['ci_95_low']:+.3f}, {entry['ci_95_high']:+.3f}]"
        sig = "  *" if entry["significant"] else "   "
        print(
            f"  {metric:<25} {entry['r']:>+7.3f} {entry['p_value']:>8.4f} "
            f"{ci:>18} {sig:>6}"
        )
    print("  * = p < 0.05")

    print(f"\nSTEP 3: Does automation degrade the functional signal?")
    print(f"  {'Metric':<25} {'CLAN r':>8} {'Auto r':>8} {'Diff':>8} {'Conclusion'}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*35}")
    for metric, entry in wabaq_results.get("step3_clan_vs_auto_wabaq", {}).items():
        print(
            f"  {metric:<25} "
            f"{entry['clan']['r']:>+8.3f} "
            f"{entry['auto']['r']:>+8.3f} "
            f"{entry['r_diff']:>+8.3f}  "
            f"{entry['conclusion']}"
        )

    print("\n" + "=" * 76)
    print("  Full results : outputs/validation/rq1_full_report.json")
    print("=" * 76)


def main(args) -> None:
    output_dir = Path("outputs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    clan_df          = load_all_clan_evals(args.clan_dir)
    clan_session_ids = set(clan_df["_session_id"].tolist())
    auto_df = compute_auto_metrics(args.cha_dir, clan_session_ids, clan_df)
    validation_results, merged = run_validation(clan_df, auto_df)
    logger.info("Extracting WAB-AQ from .cha headers...")
    wabaq_df = extract_wabaq_from_cha(args.cha_dir, clan_session_ids)
    wabaq_df.to_csv(str(output_dir / "wabaq_lookup.csv"), index=False)

    n_before = len(merged)
    merged = pd.merge(merged, wabaq_df, on="_session_id", how="left")
    logger.info(
        f"WAB-AQ joined: {merged['wab_aq'].notna().sum()} / {n_before} patients have a value"
    )
    wabaq_results = run_wabaq_analysis(merged)
    merged.to_csv(str(output_dir / "matched_comparison.csv"), index=False)

    with open(output_dir / "correlation_report.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    full_report = {
        "step1_auto_vs_clan":         validation_results,
        "step2_and_3_wabaq_analysis": wabaq_results,
    }
    with open(output_dir / "rq1_full_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print_rq1_report(validation_results, wabaq_results)
    make_scatter_plots(merged, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1 Full Analysis: automated metrics vs CLAN and WAB-AQ"
    )
    parser.add_argument("--clan_dir", type=str, required=True)
    parser.add_argument("--cha_dir",  type=str, required=True)
    args = parser.parse_args()
    main(args)