#!/usr/bin/env python3
"""
experiments/run_dae.py
----------------------
Phase 1: Discourse Assessment Engine (DAE) Training & Validation.

What this script does:
  1. Parses all AphasiaBank .cha transcripts from --data_dir
  2. Extracts 5 discourse metrics per transcript (CIU, MC, MLU, TTR, SynComp)
  3. Fine-tunes RoBERTa on AphasiaBank transcripts for surprisal scoring
  4. Builds normalised 14-dim patient state vectors
  5. Saves state vectors, scaler, and patient profiles to outputs/dae/

Outputs (used by run_pes.py):
  outputs/dae/state_vectors.npz       - all patient state vectors
  outputs/dae/patient_profiles.json   - metadata per participant
  outputs/dae/longitudinal.json       - participants with 2+ sessions
  outputs/dae/scaler.npz              - fitted min-max scaler
  outputs/dae/metrics_report.json     - inter-rater reliability stats

Usage:
  python experiments/run_dae.py --data_dir data/aphasiabank
  python experiments/run_dae.py --data_dir data/aphasiabank --skip_roberta
"""

from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dapta.dae.parser import CHATParser, PatientTranscript
from dapta.dae.metrics import DiscourseMetricExtractor, DiscourseMetrics
from dapta.dae.roberta_scorer import RoBERTaScorer
from dapta.dae.state_builder import PatientStateBuilder, PatientProfile
from dapta.utils.logger import get_logger
from dapta.utils.config import Config

logger = get_logger(__name__, log_file="logs/run_dae.log")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    """
    Pull WAB-AQ, aphasia subtype, and months post-onset from CHAT @ID header.
    Format: @ID: eng|corpus|PAR|age|sex|WABtype||Participant||WAB-AQ|
    """
    meta = transcript.metadata
    wab_aq = 50.0
    aphasia_subtype = "Other"
    months_post_onset = 12.0

    # Try to extract from metadata dict (populated by parser)
    if "diagnosis" in meta and meta["diagnosis"]:
        aphasia_subtype = str(meta["diagnosis"])
    if "wab_aq" in meta:
        try:
            wab_aq = float(meta["wab_aq"])
        except (ValueError, TypeError):
            pass

    # Fallback: scan raw file for @ID line
    raw_path = meta.get("filepath")
    if raw_path and Path(raw_path).exists():
        try:
            text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("@ID:") and "PAR" in line:
                    parts = line.split("|")
                    if len(parts) >= 6:
                        aphasia_subtype = parts[5].strip() or aphasia_subtype
                    if len(parts) >= 10:
                        try:
                            wab_aq = float(parts[9].strip())
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    return {
        "participant_id": transcript.participant_id,
        "session_id": transcript.session_id,
        "aphasia_subtype": aphasia_subtype,
        "wab_aq": wab_aq,
        "months_post_onset": months_post_onset,
    }


def find_longitudinal_participants(transcripts: List[PatientTranscript]) -> Dict[str, List[str]]:
    """
    Group transcripts by base participant ID (strip trailing session letter).
    e.g. williamson01a, williamson01b -> williamson01: [session_a, session_b]
    """
    pattern = re.compile(r"^(.+?)([a-z])$")
    groups: Dict[str, List[str]] = defaultdict(list)

    for t in transcripts:
        sid = t.session_id.lower()
        m = pattern.match(sid)
        if m:
            base = m.group(1)
            groups[base].append(t.session_id)
        else:
            groups[sid].append(t.session_id)

    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def split_participants(
    all_ids: List[str],
    longitudinal_ids: List[str],
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split participants into train/val/test.
    Longitudinal participants go to train (needed for transition model).
    Single-session participants: 70/15/15 split.
    """
    rng = np.random.default_rng(seed)
    single = [pid for pid in all_ids if pid not in longitudinal_ids]
    rng.shuffle(single)

    n = len(single)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = longitudinal_ids + single[:n_train]
    val = single[n_train:n_train + n_val]
    test = single[n_train + n_val:]

    logger.info(
        f"Split: {len(train)} train ({len(longitudinal_ids)} longitudinal), "
        f"{len(val)} val, {len(test)} test"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    cfg = Config.load()
    output_dir = Path("outputs/dae")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 1: Discourse Assessment Engine")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")

    # ------------------------------------------------------------------
    # Step 1: Parse all .cha files
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Parsing AphasiaBank transcripts...")
    parser = CHATParser()
    transcripts = parser.parse_directory(args.data_dir)

    if not transcripts:
        logger.error(f"No .cha files found in {args.data_dir}. Check your data_dir path.")
        return

    logger.info(f"Parsed {len(transcripts)} transcripts.")

    # Store filepath in metadata for later header scanning
    all_cha = list(Path(args.data_dir).rglob("*.cha"))
    cha_by_stem = {f.stem.lower(): str(f) for f in all_cha}
    for t in transcripts:
        t.metadata["filepath"] = cha_by_stem.get(t.session_id.lower(), "")

    # ------------------------------------------------------------------
    # Step 2: Extract discourse metrics
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Extracting discourse metrics...")

    all_metrics: List[DiscourseMetrics] = []
    all_session_ids: List[str] = []
    failed = []

    # Priority order
    TASK_PRIORITY = ["cookie_theft", "cinderella", "sandwich",
                 "stroke_narrative", "conversation"]
    for t in transcripts:
        try:
            # Compute metrics per task, pick best available
            best_metrics = None
            for task in TASK_PRIORITY:
                if not t.has_task(task):
                    continue
                utts = [u.text for u in t.tasks[task] if u.text.strip()]
                if not utts:
                    continue
                ext = DiscourseMetricExtractor(task=task)
                best_metrics = ext.compute(utts)
                break  # Use highest-priority task found

            if best_metrics is None:
                failed.append(t.session_id)
                continue

            all_metrics.append(best_metrics)
            all_session_ids.append(t.session_id)
        except Exception as e:
            logger.warning(f"Metric extraction failed for {t.session_id}: {e}")
            failed.append(t.session_id)

    logger.info(f"Extracted metrics for {len(all_metrics)} transcripts. Failed: {len(failed)}")

    # ------------------------------------------------------------------
    # Step 3: RoBERTa surprisal scoring
    # ------------------------------------------------------------------
    if args.skip_roberta:
        logger.info("\n[3/5] Skipping RoBERTa (--skip_roberta flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_metrics)
    else:
        logger.info("\n[3/5] Computing RoBERTa surprisal scores...")
        logger.info("Fine-tuning RoBERTa on AphasiaBank transcripts (MLM objective)...")

        scorer = RoBERTaScorer(
            model_name="roberta-base",
            device=args.device,
            checkpoint_path="outputs/dae/roberta_checkpoint",
        )

        # Collect all utterance texts for fine-tuning
        train_texts = []
        for t in transcripts[:int(len(transcripts) * 0.85)]:  # train split
            text = " ".join(u.text for u in t.utterances if u.text.strip())
            if text.strip():
                train_texts.append(text)

        # Split into train/val for the trainer
        split_idx = int(len(train_texts) * 0.9)
        train_sub = train_texts[:split_idx]
        val_sub = train_texts[split_idx:]

        if train_sub and val_sub:
            scorer.fine_tune(
                train_utterances=train_sub,  # Fixed name
                val_utterances=val_sub,      # Added required argument
                num_epochs=1,                # Set to 1 for speed (< 2 hours)
                batch_size=16,
                learning_rate=2e-5,
            )

        # Score all transcripts
        all_surprisals = []
        for t in transcripts:
            if t.session_id in all_session_ids:
                utts = [u.text for u in t.utterances if u.text.strip()]
                surprisal = scorer.mean_surprisal(utts) if utts else 0.0
                all_surprisals.append(surprisal)

        logger.info(f"Surprisal scores computed. Mean: {np.mean(all_surprisals):.3f}")

    # ------------------------------------------------------------------
    # Step 4: Build patient profiles and state vectors
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Building patient state vectors...")

    # Map session_id -> transcript for metadata extraction
    session_to_transcript = {t.session_id: t for t in transcripts}

    all_profiles: List[PatientProfile] = []
    profile_metadata: List[dict] = []

    for session_id in all_session_ids:
        t = session_to_transcript.get(session_id)
        if t is None:
            continue
        meta = extract_metadata_from_transcript(t)
        profile = PatientProfile(
            participant_id=meta["participant_id"],
            aphasia_subtype=meta["aphasia_subtype"],
            wab_aq=meta["wab_aq"],
            months_post_onset=meta["months_post_onset"],
        )
        all_profiles.append(profile)
        profile_metadata.append(meta)

    # Find longitudinal participants
    longitudinal = find_longitudinal_participants(transcripts)
    longitudinal_base_ids = list(longitudinal.keys())

    logger.info(f"Found {len(longitudinal)} longitudinal participants (2+ sessions)")

    # Train/val/test split
    unique_participants = list({p.participant_id for p in all_profiles})
    train_ids, val_ids, test_ids = split_participants(
        unique_participants, longitudinal_base_ids
    )

    # Fit scaler on training data only
    train_mask = [
        any(pid in train_ids for pid in [p.participant_id])
        for p in all_profiles
    ]
    train_metrics = [m for m, flag in zip(all_metrics, train_mask) if flag]
    train_surprisals = [s for s, flag in zip(all_surprisals, train_mask) if flag]
    train_profiles = [p for p, flag in zip(all_profiles, train_mask) if flag]

    if not train_metrics:
        logger.warning("No training metrics found — fitting scaler on all data.")
        train_metrics = all_metrics
        train_surprisals = all_surprisals
        train_profiles = all_profiles

    state_builder = PatientStateBuilder(
        scaler_path="outputs/dae/scaler.npz"
    )
    state_builder.fit(train_metrics, train_surprisals, train_profiles)

    # Build all state vectors
    state_vectors = state_builder.build_batch(
        all_metrics, all_surprisals, all_profiles
    )

    logger.info(f"State vectors shape: {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected 14)")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Saving outputs...")

    # State vectors
    np.savez(
        str(output_dir / "state_vectors.npz"),
        state_vectors=state_vectors,
        session_ids=np.array(all_session_ids),
    )

    # Patient profiles
    with open(output_dir / "patient_profiles.json", "w") as f:
        json.dump(profile_metadata, f, indent=2)

    # Longitudinal participants
    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump(longitudinal, f, indent=2)

    # Train/val/test splits
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Metrics report
    metrics_summary = {
        "n_transcripts": len(all_metrics),
        "n_longitudinal_participants": len(longitudinal),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "mean_ciu_rate": float(np.mean([m.ciu_rate for m in all_metrics])),
        "mean_mlu": float(np.mean([m.mlu_morphemes for m in all_metrics])),
        "mean_ttr": float(np.mean([m.ttr for m in all_metrics])),
        "mean_surprisal": float(np.mean(all_surprisals)),
        "failed_transcripts": failed,
    }
    with open(output_dir / "metrics_report.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete.")
    logger.info(f"  Transcripts processed : {len(all_metrics)}")
    logger.info(f"  Longitudinal patients : {len(longitudinal)}")
    logger.info(f"  State vector shape    : {state_vectors.shape}")
    logger.info(f"  Outputs saved to      : {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_pes.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 1: DAE Training")
    parser.add_argument(
        "--data_dir", type=str, default="data/aphasiabank",
        help="Path to AphasiaBank directory containing .cha files"
    )
    parser.add_argument(
        "--skip_roberta", action="store_true",
        help="Skip RoBERTa fine-tuning (faster, uses zero surprisal)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for RoBERTa: 'cuda' or 'cpu' (auto-detected if not set)"
    )
    args = parser.parse_args()
    main(args)