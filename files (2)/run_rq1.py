"""
run_rq1.py
==========
Master script for RQ1.

Answers: "Can discourse-level metrics be automatically computed with
enough accuracy to serve as a therapy reward signal?"

What this script does, in order:
  1. Parses all .cha files in data/raw/
  2. Computes the five discourse metrics for each participant
  3. Builds the state vector CSV (one row per participant)
  4. Validates our metrics against CLAN ground truth (RQ1 answer)
  5. Optionally computes RoBERTa surprisal (requires GPU + transformers)

Run:
    python run_rq1.py

Optional flags:
    python run_rq1.py --cha_dir    path/to/cha/files
    python run_rq1.py --clan_csv   path/to/clan_metrics.csv
    python run_rq1.py --surprisal          add surprisal scores (slow, needs GPU)
    python run_rq1.py --finetune_roberta   fine-tune before scoring (very slow)
"""

import argparse
from pathlib import Path

import pandas as pd

from parser        import parse_directory
from metrics       import compute_metrics
from state_vectors import build_state_vectors, print_summary
from validate      import run_validation


# =============================================================================
# DEFAULT PATHS
# =============================================================================

CHA_DIR      = "data/raw"
CLAN_CSV     = "data/processed/clan_metrics.csv"
OUT_METRICS  = "data/processed/our_metrics.csv"
OUT_STATE    = "data/processed/state_vectors.csv"
VAL_DIR      = "results/validation"
ROBERTA_CKPT = "models/roberta_finetuned"


# =============================================================================
# HELPER: compute our metrics CSV (for validation comparison)
# =============================================================================

def compute_our_metrics_csv(transcripts, out_path: str) -> pd.DataFrame:
    """
    Produce the flat metrics CSV that validate.py reads.
    One row per participant, column names matching validate.py's METRIC_PAIRS.
    """
    rows = []
    for t in transcripts:
        m = compute_metrics(t.all_raw())
        rows.append({
            "participant_id": t.participant_id,
            "mlu_words":      m.mlu_words,
            "mlu_morphemes":  m.mlu_morphemes,
            "ttr":            m.ttr,
            "total_tokens":   m.total_tokens,
            "ndw":            m.ndw,
            "n_utterances":   m.n_utterances,
        })

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DAPTA RQ1 — compute and validate discourse metrics"
    )
    parser.add_argument("--cha_dir",          default=CHA_DIR)
    parser.add_argument("--clan_csv",         default=CLAN_CSV)
    parser.add_argument("--out_metrics",      default=OUT_METRICS)
    parser.add_argument("--out_state",        default=OUT_STATE)
    parser.add_argument("--val_dir",          default=VAL_DIR)
    parser.add_argument("--roberta_ckpt",     default=ROBERTA_CKPT)
    parser.add_argument("--surprisal",        action="store_true",
                        help="Compute RoBERTa surprisal (requires transformers + GPU)")
    parser.add_argument("--finetune_roberta", action="store_true",
                        help="Fine-tune RoBERTa before scoring (very slow)")
    parser.add_argument("--skip_validation",  action="store_true",
                        help="Skip CLAN validation (if you have no clan_csv yet)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  DAPTA — Discourse-Aware Personalised Therapy Agent")
    print("  RQ1: Automatic Discourse Metric Computation & Validation")
    print("=" * 65)

    # ----------------------------------------------------------------
    # STEP 1: Parse all .cha files
    # ----------------------------------------------------------------
    print("\n[Step 1] Parsing .cha files...")
    transcripts = parse_directory(args.cha_dir)

    if not transcripts:
        print("No transcripts found. Check --cha_dir path.")
        return

    # ----------------------------------------------------------------
    # STEP 2: Compute our metrics + save flat CSV for validation
    # ----------------------------------------------------------------
    print("\n[Step 2] Computing discourse metrics...")
    df_ours = compute_our_metrics_csv(transcripts, args.out_metrics)
    print(f"Computed metrics for {len(df_ours)} participants.")
    print(f"Saved → {args.out_metrics}")

    # Quick preview
    preview_cols = ["participant_id", "n_utterances", "mlu_words",
                    "mlu_morphemes", "ttr", "total_tokens", "ndw"]
    print(f"\nPreview (first 5 rows):")
    print(df_ours[preview_cols].head().to_string(index=False))

    # ----------------------------------------------------------------
    # STEP 3 (optional): Surprisal
    # ----------------------------------------------------------------
    surprisal_map = None

    if args.surprisal or args.finetune_roberta:
        print("\n[Step 3] RoBERTa surprisal scoring...")
        from surprisal import SurprisalScorer, finetune_roberta

        # Collect all cleaned utterances for fine-tuning / scoring
        # We use the cleaned version (no CHAT markup) for the language model
        from metrics import _clean_for_freq as clean_text
        all_utterances_by_pid = {
            t.participant_id: [clean_text(r) for r in t.all_raw() if r.strip()]
            for t in transcripts
        }

        if args.finetune_roberta:
            print("Fine-tuning RoBERTa on training transcripts...")
            # Use 80% for fine-tuning, 20% for validation perplexity
            all_utts = [u for utts in all_utterances_by_pid.values() for u in utts]
            split    = int(0.8 * len(all_utts))
            train_u, val_u = all_utts[:split], all_utts[split:]
            finetune_roberta(
                train_utterances = train_u,
                val_utterances   = val_u,
                output_dir       = args.roberta_ckpt,
            )

        scorer = SurprisalScorer(checkpoint_path=args.roberta_ckpt)
        surprisal_map = {}
        for pid, utts in all_utterances_by_pid.items():
            surprisal_map[pid] = scorer.mean_surprisal(utts)
            print(f"  {pid}: surprisal={surprisal_map[pid]:.4f}")

    else:
        print("\n[Step 3] Skipping surprisal (use --surprisal to enable).")
        print("          Surprisal column will be NaN in state vectors.")
        print("          You can add it later and re-run.")

    # ----------------------------------------------------------------
    # STEP 4: Build state vector CSV
    # ----------------------------------------------------------------
    print("\n[Step 4] Building state vector table...")
    df_state = build_state_vectors(
        transcripts   = transcripts,
        surprisal_map = surprisal_map,
        out_path      = args.out_state,
    )
    print_summary(df_state)

    # ----------------------------------------------------------------
    # STEP 5: Validate against CLAN (RQ1 answer)
    # ----------------------------------------------------------------
    if not args.skip_validation:
        clan_path = Path(args.clan_csv)
        if clan_path.exists():
            print("\n[Step 5] Validating against CLAN ground truth (RQ1)...")
            df_val = run_validation(
                our_csv  = args.out_metrics,
                clan_csv = args.clan_csv,
                out_dir  = args.val_dir,
            )
        else:
            print(f"\n[Step 5] CLAN CSV not found at {clan_path}")
            print("         Skipping validation. To validate:")
            print("         1. Run 02_clan_metrics.py to generate clan_metrics.csv")
            print("         2. Re-run this script")
    else:
        print("\n[Step 5] Validation skipped (--skip_validation flag set).")

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RQ1 complete.")
    print(f"  State vectors  → {args.out_state}")
    print(f"  Our metrics    → {args.out_metrics}")
    if not args.skip_validation and Path(args.clan_csv).exists():
        print(f"  Validation     → {args.val_dir}/rq1_validation_results.csv")
        print(f"  Scatter plots  → {args.val_dir}/rq1_validation_scatter.png")
    print("=" * 65)
    print("\nNext step: run cluster.py to discover patient groups (k-means).")


if __name__ == "__main__":
    main()
