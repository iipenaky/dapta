"""
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


<<<<<<< Updated upstream

# Helpers


def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    """
    Pull WAB-AQ, aphasia subtype, and months post-onset from CHAT @ID header.
    Format: @ID: eng|corpus|PAR|age|sex|WABtype||Participant||WAB-AQ|
    """
    meta = transcript.metadata
    wab_aq = 50.0
=======
# ======================================================================
def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    meta          = transcript.metadata
    wab_aq        = None
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        "wab_aq": wab_aq,
        "months_post_onset": months_post_onset,
    }


def find_longitudinal_participants(transcripts: List[PatientTranscript]) -> Dict[str, List[str]]:
    """
    Group transcripts by base participant ID (strip trailing session letter).
    e.g. williamson01a, williamson01b -> williamson01: [session_a, session_b]
    """
=======
        "wab_aq":          wab_aq,
    }


# ======================================================================
def find_longitudinal_participants(
    transcripts: List[PatientTranscript],
) -> Dict[str, List[str]]:
>>>>>>> Stashed changes
    pattern = re.compile(r"^(.+?)([a-z])$")
    groups: Dict[str, List[str]] = defaultdict(list)
    for t in transcripts:
        sid = t.session_id.lower()
<<<<<<< Updated upstream
        m = pattern.match(sid)
        if m:
            base = m.group(1)
            groups[base].append(t.session_id)
        else:
            groups[sid].append(t.session_id)

=======
        m   = pattern.match(sid)
        base = m.group(1) if m else sid
        groups[base].append(t.session_id)
>>>>>>> Stashed changes
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


# ======================================================================
def split_participants(
    all_ids:          List[str],
    longitudinal_ids: List[str],
    seed:             int = 42,
) -> Tuple[List[str], List[str], List[str]]:
<<<<<<< Updated upstream
    """
    Split participants into train/val/test.
    Longitudinal participants go to train (needed for transition model).
    Single-session participants: 70/15/15 split.
    """
    rng = np.random.default_rng(seed)
=======
    rng    = np.random.default_rng(seed)
>>>>>>> Stashed changes
    single = [pid for pid in all_ids if pid not in longitudinal_ids]
    rng.shuffle(single)

    n = len(single)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = longitudinal_ids + single[:n_train]
<<<<<<< Updated upstream
    val = single[n_train:n_train + n_val]
    test = single[n_train + n_val:]
=======
    val   = single[n_train: n_train + n_val]
    test  = single[n_train + n_val:]
>>>>>>> Stashed changes

    logger.info(
        f"Split: {len(train)} train ({len(longitudinal_ids)} longitudinal), "
        f"{len(val)} val, {len(test)} test"
    )
    return train, val, test


<<<<<<< Updated upstream

# Main

=======
# ======================================================================
def build_session_signals(
    transcript:   PatientTranscript,
    task_metrics: Dict[str, DiscourseMetrics],
) -> SessionSignals:
    maze_rates = [m.maze_rate for m in task_metrics.values()]
    maze_rate  = float(np.mean(maze_rates)) if maze_rates else 0.0

    utt_length_std: Dict[str, float] = {}
    for task, utts in transcript.tasks.items():
        clean_texts = [u.text for u in utts if u.text.strip()]
        utt_length_std[task] = compute_utt_length_std(clean_texts)

    # Pause computed directly from u.start_ms / u.end_ms.
    # u.raw does NOT contain CLAN timestamps after pylangacq processing.
    mean_pause_ms = compute_mean_pause_ms(transcript.utterances)

    return SessionSignals(
        maze_rate      = maze_rate,
        utt_length_std = utt_length_std,
        mean_pause_ms  = mean_pause_ms,
    )
>>>>>>> Stashed changes


# ======================================================================
def main(args) -> None:
    cfg = Config.load()
    output_dir = Path("outputs/dae")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 1: Discourse Assessment Engine")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")

<<<<<<< Updated upstream

    # Step 1: Parse all .cha files

    logger.info("\n[1/5] Parsing AphasiaBank transcripts...")
    parser = CHATParser()
=======
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Parsing AphasiaBank transcripts...")
    parser      = CHATParser(participant_tier="PAR")
>>>>>>> Stashed changes
    transcripts = parser.parse_directory(args.data_dir)

    if not transcripts:
<<<<<<< Updated upstream
        logger.error(f"No .cha files found in {args.data_dir}. Check your data_dir path.")
=======
        logger.error(f"No .cha files found in {args.data_dir}.")
>>>>>>> Stashed changes
        return
    logger.info(f"Parsed {len(transcripts)} transcripts.")

    # Store filepath in metadata for later header scanning
    all_cha = list(Path(args.data_dir).rglob("*.cha"))
    cha_by_stem = {f.stem.lower(): str(f) for f in all_cha}
    for t in transcripts:
        t.metadata["filepath"] = cha_by_stem.get(t.session_id.lower(), "")

<<<<<<< Updated upstream

    # Step 2: Extract discourse metrics

    logger.info("\n[2/5] Extracting discourse metrics...")

    all_metrics: List[DiscourseMetrics] = []
    all_session_ids: List[str] = []
    failed = []
=======
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Extracting discourse metrics...")

    all_task_metrics:    List[Dict[str, DiscourseMetrics]] = []
    all_session_signals: List[SessionSignals]              = []
    all_session_ids:     List[str]                         = []
    failed:              List[str]                         = []
>>>>>>> Stashed changes

    # Priority order
    TASK_PRIORITY = ["cookie_theft", "cinderella", "sandwich",
                 "stroke_narrative", "conversation"]
    for t in transcripts:
        try:
<<<<<<< Updated upstream
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
=======
            task_dict: Dict[str, DiscourseMetrics] = {}

            for task in TASK_PRIORITY:
                if not t.has_task(task):
                    continue

                task_utt_objects = [u for u in t.tasks[task] if u.text.strip()]
                clean_utts       = [u.text     for u in task_utt_objects]
                raw_utts         = [u.raw      for u in task_utt_objects]

                if not clean_utts:
                    continue

                # Derive the @G marker from the first utterance in this
                # task group so WAB sub-pictures get the right mc_score
                # concept list (window / umbrella / cat / flood).
                first_utt  = task_utt_objects[0]
                g_marker   = getattr(first_utt, "g_marker", "")

                ext = DiscourseMetricExtractor(task=task, marker=g_marker)

                task_dict[task] = ext.compute(
                    clean_utts,
                    raw_utterances    = raw_utts,
                    utterance_objects = task_utt_objects,
                )
>>>>>>> Stashed changes

            if best_metrics is None:
                failed.append(t.session_id)
                continue

            all_metrics.append(best_metrics)
            all_session_ids.append(t.session_id)
        except Exception as e:
            logger.warning(f"Metric extraction failed for {t.session_id}: {e}")
            failed.append(t.session_id)

<<<<<<< Updated upstream
    logger.info(f"Extracted metrics for {len(all_metrics)} transcripts. Failed: {len(failed)}")


    # Step 3: RoBERTa surprisal scoring

    if args.skip_roberta:
        logger.info("\n[3/5] Skipping RoBERTa (--skip_roberta flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_metrics)
=======
    logger.info(
        f"Extracted metrics for {len(all_task_metrics)} transcripts. "
        f"Failed: {len(failed)}"
    )

    # ------------------------------------------------------------------
    if args.skip_roberta:
        logger.info("\n[3/5] Skipping RoBERTa (--skip_roberta flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_task_metrics)
>>>>>>> Stashed changes
    else:
        logger.info("\n[3/5] Computing RoBERTa surprisal scores...")
        logger.info("Fine-tuning RoBERTa on AphasiaBank transcripts (MLM objective)...")

        scorer = RoBERTaScorer(
<<<<<<< Updated upstream
            model_name="roberta-base",
            device=args.device,
            checkpoint_path="outputs/dae/roberta_checkpoint",
=======
            model_name      = "distilroberta-base",
            device          = args.device,
            checkpoint_path = "outputs/dae/roberta_checkpoint",
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                train_utterances=train_sub,  # Fixed name
                val_utterances=val_sub,      # Added required argument
                num_epochs=15,                # Set to 1 for speed (< 2 hours)
                batch_size=16,
                learning_rate=2e-5,
            )
=======
                train_utterances        = train_sub,
                val_utterances          = val_sub,
                output_dir              = None,
                num_epochs              = 15,
                batch_size              = 16,
                learning_rate           = 2e-5,
                mlm_probability         = 0.15,
                warmup_ratio            = 0.06,
                weight_decay            = 0.01,
                early_stopping_patience = 3,
            )
        else:
            logger.warning("Not enough utterances for fine-tuning.")
>>>>>>> Stashed changes

        # Score all transcripts
        all_surprisals = []
        from tqdm import tqdm
        for t in tqdm(transcripts, desc = "Scoring transcripts", unit = "trandcript"):
            if t.session_id in all_session_ids:
                utts = [u.text for u in t.utterances if u.text.strip()]
                surprisal = scorer.mean_surprisal(utts) if utts else 0.0
                all_surprisals.append(surprisal)

<<<<<<< Updated upstream
        logger.info(f"Surprisal scores computed. Mean: {np.mean(all_surprisals):.3f}")


    # Step 4: Build patient profiles and state vectors

=======
        surprisal_map: Dict[str, float] = {}
        session_id_set = set(all_session_ids)
        try:
            from tqdm import tqdm
            transcript_iter = tqdm(
                transcripts, desc="Scoring transcripts", unit="transcript"
            )
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
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
            participant_id=meta["participant_id"],
            aphasia_subtype=meta["aphasia_subtype"],
            wab_aq=meta["wab_aq"],
            months_post_onset=meta["months_post_onset"],
=======
            participant_id  = meta["participant_id"],
            aphasia_subtype = meta["aphasia_subtype"],
            wab_aq          = meta["wab_aq"],
>>>>>>> Stashed changes
        )
        all_profiles.append(profile)
        profile_metadata.append(meta)

<<<<<<< Updated upstream
    # Find longitudinal participants
    longitudinal = find_longitudinal_participants(transcripts)
=======
    # ------------------------------------------------------------------
    # Separate aphasia patients from controls.
    # Controls are kept in `all_*` for RQ1 DAE validation output.
    # The `rl_*` variables are used for everything RL-related:
    #   scaler fitting, state vector building, splits, clustering, PES.
    # ------------------------------------------------------------------
    n_controls = sum(1 for p in all_profiles if p.is_control)
    n_aphasia  = len(all_profiles) - n_controls
    logger.info(
        f"Participants: {n_aphasia} aphasia, {n_controls} controls "
        f"(controls excluded from RL pipeline)"
    )

    aphasia_mask = [not p.is_control for p in all_profiles]

    rl_task_metrics = [m for m, f in zip(all_task_metrics,    aphasia_mask) if f]
    rl_surprisals   = [s for s, f in zip(all_surprisals,       aphasia_mask) if f]
    rl_profiles     = [p for p, f in zip(all_profiles,         aphasia_mask) if f]
    rl_signals      = [s for s, f in zip(all_session_signals,  aphasia_mask) if f]
    rl_session_ids  = [s for s, f in zip(all_session_ids,      aphasia_mask) if f]
    rl_metadata     = [m for m, f in zip(profile_metadata,     aphasia_mask) if f]

    if not rl_task_metrics:
        logger.error("No aphasia transcripts found after filtering controls. Exiting.")
        return

    # ------------------------------------------------------------------
    longitudinal          = find_longitudinal_participants(transcripts)
>>>>>>> Stashed changes
    longitudinal_base_ids = list(longitudinal.keys())

    logger.info(f"Found {len(longitudinal)} longitudinal participants (2+ sessions)")

<<<<<<< Updated upstream
    # Train/val/test split
    unique_participants = list({p.participant_id for p in all_profiles})
=======
    unique_participants = list({p.participant_id for p in rl_profiles})
>>>>>>> Stashed changes
    train_ids, val_ids, test_ids = split_participants(
        unique_participants, longitudinal_base_ids
    )

<<<<<<< Updated upstream
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
=======
    train_mask         = [p.participant_id in train_ids for p in rl_profiles]
    train_task_metrics = [m for m, f in zip(rl_task_metrics, train_mask) if f]
    train_surprisals   = [s for s, f in zip(rl_surprisals,   train_mask) if f]
    train_profiles     = [p for p, f in zip(rl_profiles,     train_mask) if f]
    train_signals      = [s for s, f in zip(rl_signals,      train_mask) if f]

    if not train_task_metrics:
        logger.warning("No training metrics found — fitting scaler on all aphasia data.")
        train_task_metrics = rl_task_metrics
        train_surprisals   = rl_surprisals
        train_profiles     = rl_profiles
        train_signals      = rl_signals

    state_builder = PatientStateBuilder(scaler_path="outputs/dae/scaler.npz")
    state_builder.fit(
        train_task_metrics,
        train_surprisals,
        train_profiles,
        all_signals = train_signals,
>>>>>>> Stashed changes
    )
    state_builder.fit(train_metrics, train_surprisals, train_profiles)

    # Build all state vectors
    state_vectors = state_builder.build_batch(
<<<<<<< Updated upstream
        all_metrics, all_surprisals, all_profiles
    )

    logger.info(f"State vectors shape: {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected 14)")


    # Step 5: Save outputs

=======
        rl_task_metrics,
        rl_surprisals,
        rl_profiles,
        all_signals = rl_signals,
    )

    logger.info(f"State vectors shape:  {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected {state_builder.state_dim()})")

    # ------------------------------------------------------------------
>>>>>>> Stashed changes
    logger.info("\n[5/5] Saving outputs...")

    # State vectors
    np.savez(
        str(output_dir / "state_vectors.npz"),
        state_vectors = state_vectors,
        session_ids   = np.array(rl_session_ids),
    )

    # Patient profiles
    with open(output_dir / "patient_profiles.json", "w") as f:
        json.dump(rl_metadata, f, indent=2)

    # Longitudinal participants
    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump(longitudinal, f, indent=2)

    # Train/val/test splits
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # Metrics report
    metrics_summary = {
<<<<<<< Updated upstream
        "n_transcripts": len(all_metrics),
        "n_longitudinal_participants": len(longitudinal),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "mean_ciu_rate": float(np.mean([m.ciu_rate for m in all_metrics])),
        "mean_mlu": float(np.mean([m.mlu_morphemes for m in all_metrics])),
        "mean_ttr": float(np.mean([m.ttr for m in all_metrics])),
        "mean_surprisal": float(np.mean(all_surprisals)),
=======
        "n_transcripts_total":         len(all_task_metrics),
        "n_transcripts_aphasia":        len(rl_task_metrics),
        "n_transcripts_controls":       n_controls,
        "n_longitudinal_participants":  len(longitudinal),
        "n_train":                      len(train_ids),
        "n_val":                        len(val_ids),
        "n_test":                       len(test_ids),
        "mean_ciu_rate": float(np.mean([
            m.ciu_rate
            for task_dict in rl_task_metrics
            for m in task_dict.values()
        ])),
        "mean_mlu": float(np.mean([
            m.mlu_morphemes
            for task_dict in rl_task_metrics
            for m in task_dict.values()
        ])),
        "mean_maze_rate": float(np.mean([
            sig.maze_rate for sig in rl_signals
        ])),
        "mean_surprisal":     float(np.mean(rl_surprisals)),
>>>>>>> Stashed changes
        "failed_transcripts": failed,
    }
    with open(output_dir / "metrics_report.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete.")
<<<<<<< Updated upstream
    logger.info(f"  Transcripts processed : {len(all_metrics)}")
    logger.info(f"  Longitudinal patients : {len(longitudinal)}")
    logger.info(f"  State vector shape    : {state_vectors.shape}")
    logger.info(f"  Outputs saved to      : {output_dir}/")
=======
    logger.info(f"  Transcripts processed (total)  : {len(all_task_metrics)}")
    logger.info(f"  Aphasia only (RL pipeline)     : {len(rl_task_metrics)}")
    logger.info(f"  Controls (excluded from RL)    : {n_controls}")
    logger.info(f"  Longitudinal patients          : {len(longitudinal)}")
    logger.info(f"  State vector shape             : {state_vectors.shape}")
    logger.info(f"  Outputs saved to               : {output_dir}/")
>>>>>>> Stashed changes
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_pes.py")


# ======================================================================
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