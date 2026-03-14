"""
state_vectors.py
================
Builds the state vector table — one row per participant.

Each row contains:
  - Participant metadata (from parser.py)
  - Five validated discourse metrics (from metrics.py)
  - Mean RoBERTa surprisal (from surprisal.py, optional)

This CSV is the foundation for:
  1. K-means clustering (cluster.py)
  2. The RL environment state space
  3. The reward function (change in metrics between sessions)

Columns in output CSV:
  participant_id, corpus, age, sex, diagnosis, wab_aq, session_date,
  n_utterances, tasks_found,
  mlu_words, mlu_morphemes, ttr, total_tokens, ndw, total_morphemes,
  mean_surprisal
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from parser  import Transcript, parse_directory
from metrics import compute_metrics


def build_state_vectors(
    transcripts:     List[Transcript],
    surprisal_map:   Optional[dict] = None,
    out_path:        Optional[str]  = None,
) -> pd.DataFrame:
    """
    Build state vector table from a list of parsed transcripts.

    Parameters
    ----------
    transcripts   : list of Transcript objects from parse_directory()
    surprisal_map : dict mapping participant_id → mean_surprisal float.
                    If None, surprisal column is filled with NaN.
                    Compute this separately with surprisal.py because
                    it requires GPU/transformers and takes time.
    out_path      : if provided, saves the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame with one row per participant
    """
    rows = []

    for t in transcripts:
        # Compute metrics from ALL raw utterances (all tasks combined)
        raw_all = t.all_raw()
        m       = compute_metrics(raw_all)
        task_metrics = compute_task_metrics(t)
        row = {
            # Metadata
            "participant_id": t.participant_id,
            "corpus":         t.corpus,
            "age":            t.age,
            "sex":            t.sex,
            "diagnosis":      t.diagnosis,     # raw string, not normalised
            "wab_aq":         t.wab_aq,
            "session_date":   t.session_date,
            "n_utterances":   m.n_utterances,
            "tasks_found":    "|".join(t.tasks_found) if t.tasks_found else "",

            # Five validated discourse metrics
            "mlu_words":      m.mlu_words,
            "mlu_morphemes":  m.mlu_morphemes,
            "ttr":            m.ttr,
            "total_tokens":   m.total_tokens,
            "ndw":            m.ndw,
            "total_morphemes":m.total_morphemes,

            # Task-specific metrics
            **task_metrics,

            # Surprisal (filled in below if surprisal_map provided)
            "mean_surprisal": None,
        }

        # Add surprisal if available
        if surprisal_map is not None:
            row["mean_surprisal"] = surprisal_map.get(t.participant_id, None)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by participant_id for reproducibility
    df = df.sort_values("participant_id").reset_index(drop=True)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"State vectors saved → {out_path}  ({len(df)} participants)")

    return df


def print_summary(df: pd.DataFrame):
    """Print a readable summary of the state vector table."""
    print(f"\n{'='*60}")
    print(f"State Vector Summary  ({len(df)} participants)")
    print(f"{'='*60}")

    metric_cols = ["mlu_words", "mlu_morphemes", "ttr",
                   "total_tokens", "ndw", "mean_surprisal"]

    print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} "
          f"{'Max':>8} {'Missing':>8}")
    print("-" * 62)

    for col in metric_cols:
        if col not in df.columns:
            continue
        s       = df[col].dropna()
        missing = df[col].isna().sum()
        if len(s) == 0:
            print(f"{col:<20} {'N/A':>8}")
            continue
        print(f"{col:<20} {s.mean():>8.3f} {s.std():>8.3f} "
              f"{s.min():>8.3f} {s.max():>8.3f} {missing:>8}")

    print(f"\nDiagnosis distribution (raw from @ID:):")
    if "diagnosis" in df.columns:
        counts = df["diagnosis"].value_counts(dropna=False)
        for diag, count in counts.items():
            print(f"  {str(diag):<25} {count}")

    print(f"\nCorpora represented:")
    if "corpus" in df.columns:
        for corpus, count in df["corpus"].value_counts(dropna=False).items():
            print(f"  {str(corpus):<25} {count}")

    print(f"\nWAB-AQ: {df['wab_aq'].notna().sum()} participants have scores, "
          f"{df['wab_aq'].isna().sum()} missing")

    print(f"\nTasks found across dataset:")
    if "tasks_found" in df.columns:
        from collections import Counter
        task_counter = Counter()
        for tasks_str in df["tasks_found"].dropna():
            for t in tasks_str.split("|"):
                if t:
                    task_counter[t] += 1
        for task, count in task_counter.most_common():
            print(f"  {task:<25} {count} participants")


# =============================================================================
# TASK METRIC HELPER
# =============================================================================

TASKS = [
    "cookie_theft",
    "cinderella",
    "sandwich",
    "stroke_narrative",
    "conversation",
]


def compute_task_metrics(transcript: Transcript) -> dict:
    """
    Compute discourse metrics separately for each AphasiaBank task.

    Returns
    -------
    dict
        Keys are metric_task combinations:
        e.g. mlu_words_cookie_theft
    """
    results = {}

    for task in TASKS:
        utts = transcript.utterances_for_task(task)

        if not utts:
            # No utterances for this task
            for metric in [
                "mlu_words",
                "mlu_morphemes",
                "ttr",
                "total_tokens",
                "ndw",
                "total_morphemes",
                "n_utterances",
            ]:
                results[f"{metric}_{task}"] = None
            continue

        raw = [u.raw for u in utts]
        m = compute_metrics(raw)

        results.update({
            f"mlu_words_{task}":      m.mlu_words,
            f"mlu_morphemes_{task}":  m.mlu_morphemes,
            f"ttr_{task}":            m.ttr,
            f"total_tokens_{task}":   m.total_tokens,
            f"ndw_{task}":            m.ndw,
            f"total_morphemes_{task}":m.total_morphemes,
            f"n_utterances_{task}":   m.n_utterances,
        })

    return results
