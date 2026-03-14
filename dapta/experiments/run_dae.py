"""
Phase 1: Discourse Assessment Engine (DAE) Training & Validation.

What this script does:
  1. Parses all AphasiaBank .cha transcripts from --data_dir
  2. Extracts discourse metrics per transcript per task (CIU, MC, MLU, SynComp)
  3. Fine-tunes RoBERTa on AphasiaBank transcripts for surprisal scoring
  4. Builds normalised 43-dim patient state vectors
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


import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dapta.dae.parser import CHATParser, PatientTranscript
from dapta.dae.metrics import (
    DiscourseMetricExtractor,
    DiscourseMetrics,
    compute_utt_length_std,
    compute_mean_pause_ms,
)
from dapta.dae.roberta_scorer import RoBERTaScorer
from dapta.dae.state_builder import (
    PatientStateBuilder,
    PatientProfile,
    SessionSignals,
    TASKS as TASK_PRIORITY,
)
from dapta.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/run_dae.log")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    """
    Pull WAB-AQ and aphasia subtype from CHAT @ID header.
    Format: @ID: eng|corpus|PAR|age|sex|WABtype||Participant||WAB-AQ|
    """
    meta = transcript.metadata
    wab_aq = None
    aphasia_subtype = "Other"

    if "diagnosis" in meta and meta["diagnosis"]:
        aphasia_subtype = str(meta["diagnosis"])
    if "wab_aq" in meta:
        try:
            wab_aq = float(meta["wab_aq"])
        except (ValueError, TypeError):
            pass

    # Fallback: scan raw file for @ID line when parser metadata is incomplete
    raw_path = meta.get("filepath")
    if raw_path and Path(raw_path).exists():
        try:
            text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("@ID:") and "PAR" in line:
                    parts = line.split("|")
                    if len(parts) >= 6 and parts[5].strip():
                        aphasia_subtype = parts[5].strip()
                    if len(parts) >= 10 and parts[9].strip():
                        try:
                            wab_aq = float(parts[9].strip())
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    return {
        "participant_id":  transcript.participant_id,
        "session_id":      transcript.session_id,
        "aphasia_subtype": aphasia_subtype,
        "wab_aq":          wab_aq,   # None triggers PatientProfile imputation
    }


def find_longitudinal_participants(
    transcripts: List[PatientTranscript],
) -> Dict[str, List[str]]:
    """
    Group transcripts by base participant ID (strip trailing session letter).
    e.g. williamson01a, williamson01b -> williamson01: [session_a, session_b]
    """
    pattern = re.compile(r"^(.+?)([a-z])$")
    groups: Dict[str, List[str]] = defaultdict(list)

    for t in transcripts:
        sid = t.session_id.lower()
        m = pattern.match(sid)
        base = m.group(1) if m else sid
        groups[base].append(t.session_id)

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

    n       = len(single)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = longitudinal_ids + single[:n_train]
    val   = single[n_train : n_train + n_val]
    test  = single[n_train + n_val :]

    logger.info(
        f"Split: {len(train)} train ({len(longitudinal_ids)} longitudinal), "
        f"{len(val)} val, {len(test)} test"
    )
    return train, val, test


def build_session_signals(
    transcript: PatientTranscript,
    task_metrics: Dict[str, DiscourseMetrics],
) -> SessionSignals:
    """
    Build SessionSignals from a parsed transcript.

    - maze_rate    : taken from the first available task's DiscourseMetrics
                     (already computed from raw CHAT in ext.compute())
    - utt_length_std: per-task std of utterance word counts
    - mean_pause_ms : from raw CHAT timestamps across all utterances
    - wpm          : from the first available task's DiscourseMetrics
                     (already computed from raw CHAT in ext.compute())
    """
    # maze_rate and wpm: use global values averaged across tasks
    maze_rates = [m.maze_rate for m in task_metrics.values()]
    wpms       = [m.wpm       for m in task_metrics.values()]
    maze_rate  = float(np.mean(maze_rates)) if maze_rates else 0.0
    wpm        = float(np.mean(wpms))       if wpms       else 0.0

    # per-task utterance length std
    utt_length_std: Dict[str, float] = {}
    for task, utts in transcript.tasks.items():
        clean_texts = [u.text for u in utts if u.text.strip()]
        utt_length_std[task] = compute_utt_length_std(clean_texts)

    # mean pause from raw timestamps across ALL utterances
    all_raw = [u.raw for u in transcript.utterances if u.raw]
    mean_pause_ms = compute_mean_pause_ms(all_raw)

    return SessionSignals(
        maze_rate=maze_rate,
        utt_length_std=utt_length_std,
        mean_pause_ms=mean_pause_ms,
        wpm=wpm,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
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

    parser = CHATParser(participant_tier="PAR")
    transcripts = parser.parse_directory(args.data_dir)

    if not transcripts:
        logger.error(
            f"No .cha files found in {args.data_dir}. Check your data_dir path."
        )
        return

    logger.info(f"Parsed {len(transcripts)} transcripts.")

    # Store filepath in metadata for later @ID header scanning
    all_cha     = list(Path(args.data_dir).rglob("*.cha"))
    cha_by_stem = {f.stem.lower(): str(f) for f in all_cha}
    for t in transcripts:
        t.metadata["filepath"] = cha_by_stem.get(t.session_id.lower(), "")

    # ------------------------------------------------------------------
    # Step 2: Extract discourse metrics — ALL tasks per transcript
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Extracting discourse metrics...")

    all_task_metrics: List[Dict[str, DiscourseMetrics]] = []
    all_session_signals: List[SessionSignals] = []
    all_session_ids:  List[str]  = []
    failed: List[str] = []

    for t in transcripts:
        try:
            task_dict: Dict[str, DiscourseMetrics] = {}
            for task in TASK_PRIORITY:
                if not t.has_task(task):
                    continue
                raw_utts   = [u.raw  for u in t.tasks[task] if u.text.strip()]
                clean_utts = [u.text for u in t.tasks[task] if u.text.strip()]
                if not clean_utts:
                    continue
                ext = DiscourseMetricExtractor(task=task)
                # Pass raw_utterances so WPM, maze_rate, and duration are
                # computed from the original CHAT tier (not cleaned text).
                task_dict[task] = ext.compute(clean_utts, raw_utterances=raw_utts)

            if not task_dict:
                failed.append(t.session_id)
                continue

            signals = build_session_signals(t, task_dict)

            all_task_metrics.append(task_dict)
            all_session_signals.append(signals)
            all_session_ids.append(t.session_id)

        except Exception as e:
            logger.warning(f"Metric extraction failed for {t.session_id}: {e}")
            failed.append(t.session_id)

    logger.info(
        f"Extracted metrics for {len(all_task_metrics)} transcripts. "
        f"Failed: {len(failed)}"
    )

    # ------------------------------------------------------------------
    # Step 3: RoBERTa surprisal scoring
    # ------------------------------------------------------------------
    if args.skip_roberta:
        logger.info("\n[3/5] Skipping RoBERTa (--skip_roberta flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_task_metrics)

    else:
        logger.info("\n[3/5] Computing RoBERTa surprisal scores...")

        scorer = RoBERTaScorer(
            model_name="distilroberta-base",
            device=args.device,
            checkpoint_path="outputs/dae/roberta_checkpoint",
        )

        train_texts: List[str] = []
        for t in transcripts[: int(len(transcripts) * 0.85)]:
            text = " ".join(u.text for u in t.utterances if u.text.strip())
            if text.strip():
                train_texts.append(text)

        split_idx = int(len(train_texts) * 0.9)
        train_sub = train_texts[:split_idx]
        val_sub   = train_texts[split_idx:]

        if train_sub and val_sub:
            scorer.fine_tune(
                train_utterances=train_sub,
                val_utterances=val_sub,
                output_dir=None,
                num_epochs=15,
                batch_size=16,
                learning_rate=2e-5,
                mlm_probability=0.15,
                warmup_ratio=0.06,
                weight_decay=0.01,
                early_stopping_patience=3,
            )
        else:
            logger.warning(
                "Not enough utterances for fine-tuning. "
                "Loading base model instead."
            )

        scorer.load(from_checkpoint=True)

        surprisal_map: Dict[str, float] = {}
        session_id_set = set(all_session_ids)
        try:
            from tqdm import tqdm
            transcript_iter = tqdm(transcripts, desc="Scoring transcripts", unit="transcript")
        except ImportError:
            transcript_iter = transcripts

        for t in transcript_iter:
            if t.session_id not in session_id_set:
                continue
            utts = [u.text for u in t.utterances if u.text.strip()]
            surprisal_map[t.session_id] = (
                scorer.mean_surprisal(utts) if utts else 0.0
            )

        all_surprisals = [surprisal_map.get(sid, 0.0) for sid in all_session_ids]
        logger.info(f"Surprisal scores computed. Mean: {np.mean(all_surprisals):.3f}")

    # ------------------------------------------------------------------
    # Step 4: Build patient profiles and state vectors
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Building patient state vectors...")

    session_to_transcript = {t.session_id: t for t in transcripts}

    all_profiles:     List[PatientProfile] = []
    profile_metadata: List[dict]           = []

    for session_id in all_session_ids:
        t = session_to_transcript.get(session_id)
        if t is None:
            continue
        meta = extract_metadata_from_transcript(t)

        profile = PatientProfile(
            participant_id=meta["participant_id"],
            aphasia_subtype=meta["aphasia_subtype"],
            wab_aq=meta["wab_aq"],  # None → subtype-median imputation
        )
        all_profiles.append(profile)
        profile_metadata.append(meta)

    # Longitudinal grouping & train/val/test split
    longitudinal          = find_longitudinal_participants(transcripts)
    longitudinal_base_ids = list(longitudinal.keys())
    logger.info(f"Found {len(longitudinal)} longitudinal participants (2+ sessions)")

    unique_participants = list({p.participant_id for p in all_profiles})
    train_ids, val_ids, test_ids = split_participants(
        unique_participants, longitudinal_base_ids
    )

    # Fit scaler on training data only
    train_mask = [p.participant_id in train_ids for p in all_profiles]
    train_task_metrics  = [m for m, f in zip(all_task_metrics,    train_mask) if f]
    train_surprisals    = [s for s, f in zip(all_surprisals,       train_mask) if f]
    train_profiles      = [p for p, f in zip(all_profiles,         train_mask) if f]
    train_signals       = [s for s, f in zip(all_session_signals,  train_mask) if f]

    if not train_task_metrics:
        logger.warning("No training metrics found — fitting scaler on all data.")
        train_task_metrics = all_task_metrics
        train_surprisals   = all_surprisals
        train_profiles     = all_profiles
        train_signals      = all_session_signals

    state_builder = PatientStateBuilder(scaler_path="outputs/dae/scaler.npz")
    state_builder.fit(
        train_task_metrics,
        train_surprisals,
        train_profiles,
        all_signals=train_signals,
    )

    state_vectors = state_builder.build_batch(
        all_task_metrics,
        all_surprisals,
        all_profiles,
        all_signals=all_session_signals,
    )

    logger.info(f"State vectors shape: {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected {state_builder.state_dim()})")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Saving outputs...")

    np.savez(
        str(output_dir / "state_vectors.npz"),
        state_vectors=state_vectors,
        session_ids=np.array(all_session_ids),
    )

    with open(output_dir / "patient_profiles.json", "w") as f:
        json.dump(profile_metadata, f, indent=2)

    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump(longitudinal, f, indent=2)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    metrics_summary = {
        "n_transcripts":               len(all_task_metrics),
        "n_longitudinal_participants":  len(longitudinal),
        "n_train":                     len(train_ids),
        "n_val":                       len(val_ids),
        "n_test":                      len(test_ids),
        "mean_ciu_rate": float(np.mean([
            m.ciu_rate
            for task_dict in all_task_metrics
            for m in task_dict.values()
        ])),
        "mean_mlu": float(np.mean([
            m.mlu_morphemes
            for task_dict in all_task_metrics
            for m in task_dict.values()
        ])),
        "mean_wpm": float(np.mean([
            m.wpm
            for task_dict in all_task_metrics
            for m in task_dict.values()
        ])),
        "mean_maze_rate": float(np.mean([
            sig.maze_rate for sig in all_session_signals
        ])),
        "mean_surprisal":     float(np.mean(all_surprisals)),
        "failed_transcripts": failed,
    }
    with open(output_dir / "metrics_report.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete.")
    logger.info(f"  Transcripts processed : {len(all_task_metrics)}")
    logger.info(f"  Longitudinal patients : {len(longitudinal)}")
    logger.info(f"  State vector shape    : {state_vectors.shape}")
    logger.info(f"  Outputs saved to      : {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_pes.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 1: DAE Training")
    parser.add_argument(
        "--data_dir", type=str, default="data/aphasiabank",
        help="Path to AphasiaBank directory containing .cha files",
    )
    parser.add_argument(
        "--skip_roberta", action="store_true",
        help="Skip RoBERTa fine-tuning (faster, uses zero surprisal)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for RoBERTa: 'cuda' or 'cpu' (auto-detected if not set)",
    )
    args = parser.parse_args()
    main(args)