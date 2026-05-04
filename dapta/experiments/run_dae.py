import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dapta.dae.parser import CHATParser, PatientTranscript
from dapta.dae.metrics import (DiscourseMetricExtractor, DiscourseMetrics, compute_utt_length_std, compute_mean_pause_ms,)
from dapta.dae.mistral_scorer import MistralScorer
from dapta.dae.state_builder import (PatientStateBuilder, PatientProfile, SessionSignals, TASKS as TASK_PRIORITY,)
from dapta.utils.logger import get_logger

# Set up logging to both the console and a persistent log file so the full
logger = get_logger(__name__, log_file="logs/run_dae.log")


def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    """
    Pulls the clinical metadata (participant ID, session ID, aphasia subtype,
    and WAB-AQ score) out of a parsed transcript object.

    Two sources are tried in order:
      1. The in-memory metadata dict already attached to the transcript object
         (populated by the parser from CHAT header lines).
      2. The raw .cha file on disk — specifically the @ID line, which encodes
         detailed participant information in a pipe-delimited format. This is
         the fallback for fields the parser may not have captured.

    Returns a plain dict with four keys:
        participant_id, session_id, aphasia_subtype, wab_aq
    """
    meta            = transcript.metadata
    wab_aq          = None        # will stay None if not found in either source
    aphasia_subtype = "Other"     # safe default if no diagnosis is recorded

    # --- Source 1: in-memory metadata dict ---
    if "diagnosis" in meta and meta["diagnosis"]:
        aphasia_subtype = str(meta["diagnosis"])
    if "wab_aq" in meta:
        try:
            wab_aq = float(meta["wab_aq"])
        except (ValueError, TypeError):
            pass  # ignore malformed values; wab_aq stays None

    # --- Source 2: raw .cha file on disk ---
    # CHAT @ID lines follow the format:
    #   @ID: language|corpus|participant_code|age|sex|group|...|diagnosis|...|wab_aq|...
    # Fields are pipe-separated; index 5 = group/subtype, index 9 = WAB-AQ score
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
                            pass  # ignore unparseable AQ values
        except Exception:
            pass  # if the file can't be read, keep whatever we already have

    return {
        "participant_id":  transcript.participant_id,
        "session_id":      transcript.session_id,
        "aphasia_subtype": aphasia_subtype,
        "wab_aq":          wab_aq,
    }


def find_longitudinal_participants(transcripts: List[PatientTranscript]) -> Dict[str, List[str]]:
    """
    Identifies participants who have two or more sessions recorded (longitudinal).

    AphasiaBank session IDs use a naming convention where repeated sessions
    for the same participant share a common base ID with a trailing letter
    suffix (e.g. "broca01a", "broca01b", "broca01c"). This function strips
    the trailing letter to find the base ID and groups sessions by it.

    Returns a dict mapping base_id → sorted list of session_ids, for any
    base_id that has at least 2 sessions. Single-session participants are
    excluded entirely.

    Example:
        {"broca01": ["broca01a", "broca01b"], "fluent03": ["fluent03a", "fluent03b"]}
    """
    # Matches any string ending in a single lowercase letter (the session suffix)
    pattern = re.compile(r"^(.+?)([a-z])$")
    groups: Dict[str, List[str]] = defaultdict(list)
    for t in transcripts:
        sid  = t.session_id.lower()
        m    = pattern.match(sid)
        # Use the base part (without the suffix) as the grouping key;
        # fall back to the full ID if no suffix pattern is found
        base = m.group(1) if m else sid
        groups[base].append(t.session_id)
    # Keep only participants with 2 or more sessions
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def split_participants(all_ids: List[str], longitudinal_ids: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits participant IDs into train / val / test sets using a 70 / 15 / 15
    ratio, with one important rule: longitudinal participants (those with
    multiple sessions) always go into the training set.

    Why keep longitudinal participants in train?
    They provide the most signal for learning how language changes over time,
    and splitting them across sets would leak session-correlated data.

    The remaining single-session participants are shuffled randomly (with a
    fixed seed for reproducibility) and then divided 70 / 15 / 15.

    Returns three lists: (train_ids, val_ids, test_ids)
    """
    rng    = np.random.default_rng(seed)
    # Separate out participants who only have one session
    single = [pid for pid in all_ids if pid not in longitudinal_ids]
    rng.shuffle(single)

    n       = len(single)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    # The rest (roughly 15%) automatically becomes the test set

    # Longitudinal participants are prepended to the training set
    train = longitudinal_ids + single[:n_train]
    val   = single[n_train: n_train + n_val]
    test  = single[n_train + n_val:]

    logger.info(
        f"Split: {len(train)} train ({len(longitudinal_ids)} longitudinal), "
        f"{len(val)} val, {len(test)} test"
    )
    return train, val, test


def build_session_signals(transcript: PatientTranscript, task_metrics: Dict[str, DiscourseMetrics]) -> SessionSignals:
    """
    Computes session-level prosodic and disfluency signals that are not
    captured by the per-task discourse metrics.

    Three signals are computed:

    maze_rate      – average disfluency rate across all tasks (proportion of
                     utterances containing a repetition or reformulation marker).
                     Averaged across tasks rather than per-task because it
                     reflects a global fluency property of the session.

    utt_length_std – per-task standard deviation of utterance length (in words).
                     A high value means the participant alternated between very
                     short and very long utterances; low means consistent length.

    mean_pause_ms  – average silence gap between consecutive utterances in
                     milliseconds, computed across the entire session.
                     Longer pauses may indicate word-finding difficulty.

    Returns a SessionSignals object ready to be passed to PatientStateBuilder.
    """
    # Average maze_rate across all tasks — single session-level disfluency estimate
    maze_rates = [m.maze_rate for m in task_metrics.values()]
    maze_rate  = float(np.mean(maze_rates)) if maze_rates else 0.0

    # Compute utterance-length variability separately for each task
    utt_length_std: Dict[str, float] = {}
    for task, utts in transcript.tasks.items():
        clean_texts          = [u.text for u in utts if u.text.strip()]
        utt_length_std[task] = compute_utt_length_std(clean_texts)

    # Mean pause uses all utterances across the whole session (not per-task)
    mean_pause_ms = compute_mean_pause_ms(transcript.utterances)

    return SessionSignals(
        maze_rate      = maze_rate,
        utt_length_std = utt_length_std,
        mean_pause_ms  = mean_pause_ms,
    )


def main(args) -> None:
    """
    End-to-end Phase 1 pipeline for the Discourse Assessment Engine (DAE).

    The pipeline has five sequential stages:

      [1] Parse   — Read all .cha (CHAT-format) transcript files from disk and
                    filter out control (non-aphasic) participants.

      [2] Metrics — Extract discourse metrics (CIU rate, MC score, MLU, MATTR,
                    syntactic complexity, WPM, maze rate) for every task in
                    every transcript.

      [3] Mistral — Fine-tune a Mistral-7B language model on the training
                    participants' speech, then score every session's surprisal
                    (how unexpected the language was for the model). Can be
                    skipped with --skip_mistral for faster runs.

      [4] States  — Fit a MinMax scaler on training data and build a fixed-length
                    normalised state vector for every session. This vector is the
                    input representation used by downstream RL / ML models.

      [5] Save    — Write all outputs (state vectors, profiles, splits, summary
                    metrics) to outputs/dae/.

    Command-line arguments (see argparse block at the bottom):
        --data_dir      : path to the AphasiaBank .cha files
        --skip_mistral  : bypass Mistral fine-tuning and scoring
        --device        : 'cuda' or 'cpu' for Mistral (auto-detected if omitted)
    """
    output_dir = Path("outputs/dae")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 1: Discourse Assessment Engine")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")

    #  Stage 1: Parse transcripts 
    logger.info("\n[1/5] Parsing AphasiaBank transcripts...")
    parser      = CHATParser(participant_tier="PAR")
    transcripts = parser.parse_directory(args.data_dir)
    if not transcripts:
        logger.error(f"No .cha files found in {args.data_dir}.")
        return
    logger.info(f"Parsed {len(transcripts)} transcripts.")

    # Build a lookup from filename stem → full path so we can later re-read
    # the raw .cha file for any metadata the parser didn't capture
    all_cha     = list(Path(args.data_dir).rglob("*.cha"))
    cha_by_stem = {f.stem.lower(): str(f) for f in all_cha}
    for t in transcripts:
        t.metadata["filepath"] = cha_by_stem.get(t.session_id.lower(), "")

    # Remove control participants (WAB-AQ ≥ 93.8 or labelled "control").
    # Controls have no aphasia deficit and should not be included in training.
    n_total     = len(transcripts)
    transcripts = [
        t for t in transcripts
        if not PatientProfile(
            participant_id  = extract_metadata_from_transcript(t)["participant_id"],
            aphasia_subtype = extract_metadata_from_transcript(t)["aphasia_subtype"],
            wab_aq          = extract_metadata_from_transcript(t)["wab_aq"],
        ).is_control
    ]
    logger.info(f"Removed {n_total - len(transcripts)} controls. {len(transcripts)} aphasia transcripts remaining.")

    #  Stage 2: Extract discourse metrics ─
    from tqdm import tqdm
    logger.info("\n[2/5] Extracting discourse metrics...")

    all_task_metrics:    List[Dict[str, DiscourseMetrics]] = []
    all_session_signals: List[SessionSignals]              = []
    all_session_ids:     List[str]                         = []
    failed:              List[str]                         = []  # sessions that errored

    pbar = tqdm(transcripts, desc="Processing Transcripts")
    for t in pbar:
        pbar.set_description(f"Processing {t.session_id}")
        try:
            task_dict: Dict[str, DiscourseMetrics] = {}

            # Process tasks in the canonical TASK_PRIORITY order so that any
            # order-dependent downstream logic sees a consistent sequence
            for task in TASK_PRIORITY:
                if not t.has_task(task):
                    continue  # this session didn't include this task — skip it

                # Separate the utterance objects into cleaned text and raw CHAT text.
                # Clean text is used for metric computation; raw text is used for
                # WPM and maze rate where CHAT markers carry meaningful information.
                task_utt_objects = [u for u in t.tasks[task] if u.text.strip()]
                clean_utts       = [u.text for u in task_utt_objects]
                raw_utts         = [u.raw  for u in task_utt_objects]

                if not clean_utts:
                    continue  # task was listed but had no actual utterances

                # g_marker identifies which WAB picture stimulus was used
                # (e.g. "cat", "flood") so the correct concept list is scored
                first_utt       = task_utt_objects[0]
                g_marker        = getattr(first_utt, "g_marker", "")
                ext             = DiscourseMetricExtractor(task=task, marker=g_marker)
                task_dict[task] = ext.compute(
                    clean_utts,
                    raw_utterances    = raw_utts,
                    utterance_objects = task_utt_objects,
                )

            if not task_dict:
                # The transcript had no usable tasks at all — skip it entirely
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

    #  Participant split 
    # The split is computed here (before Mistral) so that the language model is
    # fine-tuned ONLY on training-set utterances — no val or test data leaks in.
    longitudinal          = find_longitudinal_participants(transcripts)
    longitudinal_base_ids = list(longitudinal.keys())
    logger.info(f"Found {len(longitudinal)} longitudinal participants (2+ sessions)")

    unique_participants          = list({
        extract_metadata_from_transcript(t)["participant_id"]
        for t in transcripts
    })
    train_ids, val_ids, test_ids = split_participants(unique_participants, longitudinal_base_ids)
    train_participant_set        = set(train_ids)

    #  Stage 3: Mistral surprisal scoring 
    if args.skip_mistral:
        # Surprisal is set to 0.0 for all sessions — useful for quick debug runs
        logger.info("\n[3/5] Skipping Mistral (--skip_mistral flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_task_metrics)
    else:
        logger.info("\n[3/5] Computing Mistral surprisal scores...")

        scorer = MistralScorer(
            model_name      = "mistralai/Mistral-7B-v0.1",
            device          = args.device,
            checkpoint_path = "outputs/dae/mistral_checkpoint",
        )

        # Collect all utterance text from training participants only.
        # Concatenating all utterances from a session into one string gives the
        # model a realistic sample of how each participant speaks.
        train_texts: List[str] = []
        for t in transcripts:
            meta = extract_metadata_from_transcript(t)
            if meta["participant_id"] not in train_participant_set:
                continue  # only use training participants for fine-tuning
            text = " ".join(u.text for u in t.utterances if u.text.strip())
            if text.strip():
                train_texts.append(text)

        # Hold out the last 10% of training texts as a fine-tuning validation set
        # to monitor overfitting and trigger early stopping if needed
        split_idx = int(len(train_texts) * 0.9)
        train_sub = train_texts[:split_idx]
        val_sub   = train_texts[split_idx:]

        if train_sub and val_sub:
            scorer.fine_tune(
                train_utterances        = train_sub,
                val_utterances          = val_sub,
                output_dir              = None,   # uses default checkpoint path
                num_epochs              = 1,
                batch_size              = 16,
                learning_rate           = 2e-5,
                warmup_ratio            = 0.06,
                weight_decay            = 0.01,
                early_stopping_patience = 3,
            )
        else:
            logger.warning("Not enough utterances for fine-tuning.")

        # Load the fine-tuned checkpoint (or the base model if fine-tuning was skipped)
        scorer.load(from_checkpoint=True)

        # Score every session that made it through metric extraction.
        # session_id_set is used as a fast membership check to skip transcripts
        # that failed in Stage 2 and have no corresponding metrics entry.
        surprisal_map:  Dict[str, float] = {}
        session_id_set = set(all_session_ids)

        try:
            transcript_iter = tqdm(transcripts, desc="Scoring transcripts", unit="transcript")
        except NameError:
            transcript_iter = transcripts

        for t in transcript_iter:
            if t.session_id not in session_id_set:
                continue  # no metrics were extracted for this session — skip
            utts = [u.text for u in t.utterances if u.text.strip()]
            surprisal_map[t.session_id] = scorer.mean_surprisal(utts) if utts else 0.0

        # Build the surprisal list in the same order as all_session_ids so that
        # indices align correctly when passed to PatientStateBuilder
        all_surprisals = [surprisal_map.get(sid, 0.0) for sid in all_session_ids]
        logger.info(f"Surprisal scores computed. Mean: {np.mean(all_surprisals):.3f}")

    #  Stage 4: Build patient state vectors 
    logger.info("\n[4/5] Building patient state vectors...")

    # Build a fast session_id → transcript lookup to avoid repeated linear searches
    session_to_transcript = {t.session_id: t for t in transcripts}

    all_profiles:     List[PatientProfile] = []
    profile_metadata: List[dict]           = []

    for session_id in all_session_ids:
        t = session_to_transcript.get(session_id)
        if t is None:
            continue
        meta = extract_metadata_from_transcript(t)
        profile = PatientProfile(
            participant_id  = meta["participant_id"],
            aphasia_subtype = meta["aphasia_subtype"],
            wab_aq          = meta["wab_aq"],
        )
        all_profiles.append(profile)
        profile_metadata.append(meta)   # kept separately for JSON export later

    logger.info(f"Participants: {len(all_profiles)} aphasia (controls already removed)")

    if not all_task_metrics:
        logger.error("No aphasia transcripts found. Exiting.")
        return

    # Filter all parallel lists down to training sessions only.
    # The scaler must be fitted on training data only — fitting on val or test
    # data would leak normalisation statistics into evaluation.
    train_mask         = [p.participant_id in train_participant_set for p in all_profiles]
    train_task_metrics = [m for m, f in zip(all_task_metrics,    train_mask) if f]
    train_surprisals   = [s for s, f in zip(all_surprisals,      train_mask) if f]
    train_profiles     = [p for p, f in zip(all_profiles,        train_mask) if f]
    train_signals      = [s for s, f in zip(all_session_signals, train_mask) if f]

    if not train_task_metrics:
        # Edge case: no participants matched the training set (e.g. tiny dataset).
        # Fall back to fitting on everything so the pipeline can still complete.
        logger.warning("No training metrics found — fitting scaler on all aphasia data.")
        train_task_metrics = all_task_metrics
        train_surprisals   = all_surprisals
        train_profiles     = all_profiles
        train_signals      = all_session_signals

    # Fit the scaler on training data, then apply it to ALL sessions (train + val + test)
    state_builder = PatientStateBuilder(scaler_path="outputs/dae/scaler.npz")
    state_builder.fit(
        train_task_metrics,
        train_surprisals,
        train_profiles,
        all_signals = train_signals,
    )

    # build_batch produces one 47-dim state vector per session
    state_vectors = state_builder.build_batch(
        all_task_metrics,
        all_surprisals,
        all_profiles,
        all_signals = all_session_signals,
    )

    logger.info(f"State vectors shape: {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected {state_builder.state_dim()})")

    #  Stage 5: Save outputs ─
    logger.info("\n[5/5] Saving outputs...")

    # state_vectors.npz — the primary output used by downstream RL / ML training
    np.savez(
        str(output_dir / "state_vectors.npz"),
        state_vectors = state_vectors,
        session_ids   = np.array(all_session_ids),
    )

    # patient_profiles.json — clinical metadata for every processed session
    with open(output_dir / "patient_profiles.json", "w") as f:
        json.dump(profile_metadata, f, indent=2)

    # longitudinal.json — maps base participant ID → list of their session IDs
    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump(longitudinal, f, indent=2)

    # splits.json — which participant IDs belong to train / val / test
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # metrics_report.json — high-level summary statistics for the whole run
    metrics_summary = {
        "n_transcripts_aphasia":       len(all_task_metrics),
        "n_controls_removed":          n_total - len(transcripts),
        "n_longitudinal_participants": len(longitudinal),
        "n_train":                     len(train_ids),
        "n_val":                       len(val_ids),
        "n_test":                      len(test_ids),
        # Average CIU rate across all tasks and all sessions
        "mean_ciu_rate": float(np.mean([
            m.ciu_rate
            for task_dict in all_task_metrics
            for m in task_dict.values()
        ])),
        # Average MLU across all tasks and all sessions
        "mean_mlu": float(np.mean([
            m.mlu_morphemes
            for task_dict in all_task_metrics
            for m in task_dict.values()
        ])),
        "mean_maze_rate":     float(np.mean([sig.maze_rate for sig in all_session_signals])),
        "mean_surprisal":     float(np.mean(all_surprisals)),
        "failed_transcripts": failed,  # session IDs that errored during metric extraction
    }
    with open(output_dir / "metrics_report.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Complete.")
    logger.info(f"  Controls removed               : {n_total - len(transcripts)}")
    logger.info(f"  Aphasia transcripts processed  : {len(all_task_metrics)}")
    logger.info(f"  Longitudinal patients          : {len(longitudinal)}")
    logger.info(f"  State vector shape             : {state_vectors.shape}")
    logger.info(f"  Outputs saved to               : {output_dir}/")
    logger.info("=" * 60)
    logger.info("Next step: python experiments/run_pes.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPTA Phase 1: DAE Training")
    parser.add_argument(
        "--data_dir", type=str, default="data/aphasiabank",
        help="Path to AphasiaBank directory containing .cha files",
    )
    parser.add_argument(
        "--skip_mistral", action="store_true",
        help="Skip Mistral fine-tuning (faster, uses zero surprisal)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for Mistral: 'cuda' or 'cpu' (auto-detected if not set)",
    )
    args = parser.parse_args()
    main(args)