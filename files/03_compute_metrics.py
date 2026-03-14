"""
03_compute_metrics.py
=====================
Phase 3: Compute ALL discourse metrics directly from raw .cha transcript text.
         CLAN outputs are NOT used here — they are the validation ground truth
         used in Phase 4.

Metrics computed here from scratch:
  1. MLU (mean length of utterance in words)
  2. TTR (type-token ratio)
  3. VOCD-D (approximated via iterative resampling — Malvern et al. 2004)
  4. NDW (number of different words)
  5. CIU% (correct information units — approximated)
  6. Maze ratio (maze words / total tokens)
  7. Composite reward (weighted z-score combination)

Output: data/processed/master_metrics.csv
        data/processed/scaler_params.yaml
        results/metric_correlation_heatmap.png

Usage:
    python 03_compute_metrics.py
"""

import re
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

PROC_DIR = Path(CFG["paths"]["processed"])
RES_DIR  = Path(CFG["paths"]["results"])
RES_DIR.mkdir(parents=True, exist_ok=True)
M        = CFG["metrics"]
WEIGHTS  = M["reward_weights"]
MIN_TOK  = M["min_tokens"]
P        = CFG["parsing"]

logging.basicConfig(
    filename=Path(CFG["paths"]["logs"]) / "03_compute_metrics.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation functions — all from raw transcript text
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [t for t in text.split() if t.strip()]


def compute_mlu(utterances: list[str]) -> float | None:
    """
    Mean Length of Utterance in words.
    Only counts utterances with at least 1 word.
    """
    lengths = [len(tokenize(u)) for u in utterances if tokenize(u)]
    if not lengths:
        return None
    return round(float(np.mean(lengths)), 4)


def compute_ttr(tokens: list[str]) -> float | None:
    """
    Type-Token Ratio = unique words / total words.
    Sensitive to transcript length — use VOCD-D for length-robust version.
    """
    if not tokens:
        return None
    return round(len(set(tokens)) / len(tokens), 4)


def compute_vocd_d(tokens: list[str],
                    min_len: int = 35,
                    n_samples: int = 100,
                    sample_sizes: tuple = (35, 50)) -> float | None:
    """
    VOCD-D: lexical diversity measure robust to transcript length.
    Method: Malvern, Richards, Chipere & Duran (2004).

    For each sample size, draw n_samples random samples and compute
    mean TTR. Fit D such that: TTR = D/N * (sqrt(1 + 2N/D) - 1)
    Uses scipy.optimize to fit D to the empirical TTR means.
    """
    from scipy.optimize import minimize_scalar

    if len(tokens) < min_len:
        return None

    def expected_ttr(n: int, d: float) -> float:
        return (d / n) * (np.sqrt(1 + 2 * n / d) - 1)

    empirical_ttrs = []
    for size in sample_sizes:
        if size > len(tokens):
            continue
        ttrs = []
        for _ in range(n_samples):
            sample = np.random.choice(tokens, size=size, replace=False)
            ttrs.append(len(set(sample)) / size)
        empirical_ttrs.append((size, float(np.mean(ttrs))))

    if not empirical_ttrs:
        return None

    def loss(d: float) -> float:
        if d <= 0:
            return 1e9
        return sum((expected_ttr(n, d) - ttr) ** 2
                   for n, ttr in empirical_ttrs)

    result = minimize_scalar(loss, bounds=(1, 200), method="bounded")
    return round(float(result.x), 4) if result.success else None


def compute_ndw(tokens: list[str]) -> int | None:
    """Number of different words (unique lemmas — here by lowercased form)."""
    return len(set(tokens)) if tokens else None


def compute_maze_ratio(raw_utterances: list[str],
                        tokens: list[str]) -> float | None:
    """
    Maze ratio = maze tokens / total tokens.
    Maze markers from config: &-um, &+w, [/], [//] etc.
    """
    if not tokens:
        return None
    maze_count = 0
    for utt in raw_utterances:
        for marker in P["maze_markers"]:
            maze_count += utt.count(marker)
    return round(maze_count / len(tokens), 4)


def compute_ciu_approx(tokens: list[str],
                        raw_utterances: list[str]) -> float | None:
    """
    Approximate CIU% from token counts.
    CIU% ≈ (total_tokens - unintelligible - maze_words) / total_tokens × 100

    Operational definition: proportion of words that are intelligible,
    relevant, and non-repetitive (Yorkston & Beukelman, 1980).

    Note in thesis: 'CIU was approximated as the proportion of total tokens
    remaining after excluding unintelligible (xxx) and maze markers,
    consistent with the operational CIU definition.'
    """
    if not tokens:
        return None

    total = len(tokens)

    # Count unintelligible tokens
    unintelligible = sum(
        utt.lower().count(P["unintelligible_marker"])
        for utt in raw_utterances
    )

    # Count maze words
    maze_count = 0
    for utt in raw_utterances:
        for marker in P["maze_markers"]:
            maze_count += utt.count(marker)

    valid = max(total - unintelligible - maze_count, 0)
    return round(valid / total * 100, 2)


def compute_all_metrics(row: pd.Series) -> dict:
    """
    Compute all six discourse metrics for one transcript row.
    Input row must have: clean_transcript, raw_transcript, n_utterances.
    """
    clean_text = str(row.get("clean_transcript", "") or "")
    raw_text   = str(row.get("raw_transcript",   "") or "")

    # Reconstruct utterance lists
    clean_utts = [u.strip() for u in clean_text.split(".") if u.strip()]
    raw_utts   = [u.strip() for u in raw_text.split(".")   if u.strip()]

    tokens = tokenize(clean_text)

    out = {
        "total_tokens":  len(tokens),
        "mlu_words":     compute_mlu(clean_utts),
        "ttr":           compute_ttr(tokens),
        "vocd_d":        compute_vocd_d(tokens),
        "ndw":           compute_ndw(tokens),
        "maze_ratio":    compute_maze_ratio(raw_utts, tokens),
        "ciu_approx":    compute_ciu_approx(tokens, raw_utts),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composite reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_reward(row: pd.Series,
                              means: dict,
                              stds: dict) -> float | None:
    """
    Weighted composite of z-scored discourse metrics.
    maze_ratio is inverted (lower maze = better communication).
    Returns None if insufficient metrics are available.
    """
    score, w_sum = 0.0, 0.0

    for metric, w in WEIGHTS.items():
        val = row.get(metric)
        if val is None or pd.isna(val):
            continue
        mean = means.get(metric, 0.0)
        std  = stds.get(metric,  1.0) or 1.0
        z    = (float(val) - mean) / std
        if metric == "maze_ratio":
            z = -z   # invert — lower maze = higher reward
        score  += w * z
        w_sum  += w

    if w_sum < 0.3:   # need at least some metrics present
        return None
    return round(score / w_sum, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 3: Computing Discourse Metrics from Transcript Text")

    trans_path = PROC_DIR / "parsed_transcripts.csv"
    if not trans_path.exists():
        console.print("[red]parsed_transcripts.csv not found. Run 01_parse_cha.py first.[/red]")
        return

    df = pd.read_csv(trans_path)
    console.print(f"Loaded {len(df)} transcript rows from {df['participant_id'].nunique()} participants\n")

    # ── Compute metrics row by row ────────────────────────────────────────
    console.print("Computing metrics from raw transcript text...")
    metric_rows = []
    for _, row in df.iterrows():
        m = compute_all_metrics(row)
        metric_rows.append(m)

    df_metrics = pd.DataFrame(metric_rows)
    df = pd.concat([df.reset_index(drop=True), df_metrics], axis=1)

    # ── Filter short transcripts ──────────────────────────────────────────
    before = len(df)
    df = df[df["total_tokens"].fillna(0) >= MIN_TOK].copy()
    console.print(f"Removed {before - len(df)} rows with < {MIN_TOK} tokens")

    # ── Fit z-score normalisation params (saved for RL environment) ───────
    metric_cols = list(WEIGHTS.keys())
    available   = [c for c in metric_cols if c in df.columns]

    means, stds = {}, {}
    for col in available:
        vals = df[col].dropna()
        means[col] = float(vals.mean()) if len(vals) > 0 else 0.0
        stds[col]  = float(vals.std())  if len(vals) > 1 else 1.0

    scaler_path = PROC_DIR / "scaler_params.yaml"
    with open(scaler_path, "w") as f:
        yaml.dump({"means": means, "stds": stds}, f)

    # ── Composite reward ──────────────────────────────────────────────────
    df["composite_reward"] = df.apply(
        lambda r: compute_composite_reward(r, means, stds), axis=1
    )

    # ── Save master metrics ───────────────────────────────────────────────
    out_path = PROC_DIR / "master_metrics.csv"
    df.to_csv(out_path, index=False)

    # ── Summary table ─────────────────────────────────────────────────────
    table = Table(title="Computed Discourse Metrics — Summary")
    table.add_column("Metric",    style="cyan")
    table.add_column("Mean",      style="green")
    table.add_column("Std",       style="green")
    table.add_column("Min",       style="yellow")
    table.add_column("Max",       style="yellow")
    table.add_column("Missing%",  style="red")

    for col in available + ["ciu_approx", "maze_ratio", "composite_reward"]:
        if col not in df.columns:
            continue
        s    = df[col].dropna()
        miss = round(df[col].isna().mean() * 100, 1)
        if len(s) == 0:
            table.add_row(col, "N/A", "N/A", "N/A", "N/A", f"{miss}%")
        else:
            table.add_row(
                col,
                f"{s.mean():.3f}",
                f"{s.std():.3f}",
                f"{s.min():.3f}",
                f"{s.max():.3f}",
                f"{miss}%",
            )
    console.print(table)

    # ── Correlation heatmap ───────────────────────────────────────────────
    plot_cols = [c for c in available + ["ciu_approx", "maze_ratio",
                                          "composite_reward", "wab_aq"]
                 if c in df.columns]
    corr = df[plot_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Discourse Metric Correlation (Python-computed)")
    plt.tight_layout()
    heatmap_path = RES_DIR / "metric_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    console.print(f"\n[green]✓ Master metrics  → {out_path}[/green]")
    console.print(f"[green]✓ Scaler params   → {scaler_path}[/green]")
    console.print(f"[green]✓ Correlation map → {heatmap_path}[/green]")
    console.print(f"\nFinal: [bold]{len(df)}[/bold] rows, "
                  f"[bold]{df['participant_id'].nunique()}[/bold] participants")


if __name__ == "__main__":
    main()