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

logger = get_logger(__name__, log_file="logs/run_dae.log")


def extract_metadata_from_transcript(transcript: PatientTranscript) -> dict:
    meta            = transcript.metadata
    wab_aq          = None
    aphasia_subtype = "Other"

    if "diagnosis" in meta and meta["diagnosis"]:
        aphasia_subtype = str(meta["diagnosis"])
    if "wab_aq" in meta:
        try:
            wab_aq = float(meta["wab_aq"])
        except (ValueError, TypeError):
            pass

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
        "wab_aq":          wab_aq,
    }


def find_longitudinal_participants(transcripts: List[PatientTranscript]) -> Dict[str, List[str]]:
    pattern = re.compile(r"^(.+?)([a-z])$")
    groups: Dict[str, List[str]] = defaultdict(list)
    for t in transcripts:
        sid  = t.session_id.lower()
        m    = pattern.match(sid)
        base = m.group(1) if m else sid
        groups[base].append(t.session_id)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def split_participants(all_ids: List[str], longitudinal_ids: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    rng    = np.random.default_rng(seed)
    single = [pid for pid in all_ids if pid not in longitudinal_ids]
    rng.shuffle(single)

    n       = len(single)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = longitudinal_ids + single[:n_train]
    val   = single[n_train: n_train + n_val]
    test  = single[n_train + n_val:]

    logger.info(
        f"Split: {len(train)} train ({len(longitudinal_ids)} longitudinal), "
        f"{len(val)} val, {len(test)} test"
    )
    return train, val, test


def build_session_signals(transcript: PatientTranscript, task_metrics: Dict[str, DiscourseMetrics]) -> SessionSignals:
    maze_rates = [m.maze_rate for m in task_metrics.values()]
    maze_rate  = float(np.mean(maze_rates)) if maze_rates else 0.0

    utt_length_std: Dict[str, float] = {}
    for task, utts in transcript.tasks.items():
        clean_texts          = [u.text for u in utts if u.text.strip()]
        utt_length_std[task] = compute_utt_length_std(clean_texts)
    mean_pause_ms = compute_mean_pause_ms(transcript.utterances)

    return SessionSignals(
        maze_rate      = maze_rate,
        utt_length_std = utt_length_std,
        mean_pause_ms  = mean_pause_ms,
    )


def main(args) -> None:
    output_dir = Path("outputs/dae")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAPTA Phase 1: Discourse Assessment Engine")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")

 
    logger.info("\n[1/5] Parsing AphasiaBank transcripts...")
    parser      = CHATParser(participant_tier="PAR")
    transcripts = parser.parse_directory(args.data_dir)
    if not transcripts:
        logger.error(f"No .cha files found in {args.data_dir}.")
        return
    logger.info(f"Parsed {len(transcripts)} transcripts.")

    all_cha     = list(Path(args.data_dir).rglob("*.cha"))
    cha_by_stem = {f.stem.lower(): str(f) for f in all_cha}
    for t in transcripts:
        t.metadata["filepath"] = cha_by_stem.get(t.session_id.lower(), "")

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

 
    from tqdm import tqdm
    logger.info("\n[2/5] Extracting discourse metrics...")

    all_task_metrics:    List[Dict[str, DiscourseMetrics]] = []
    all_session_signals: List[SessionSignals]              = []
    all_session_ids:     List[str]                         = []
    failed:              List[str]                         = []

    pbar = tqdm(transcripts, desc="Processing Transcripts")
    for t in pbar:
        pbar.set_description(f"Processing {t.session_id}")
        try:
            task_dict: Dict[str, DiscourseMetrics] = {}

            for task in TASK_PRIORITY:
                if not t.has_task(task):
                    continue

                task_utt_objects = [u for u in t.tasks[task] if u.text.strip()]
                clean_utts       = [u.text for u in task_utt_objects]
                raw_utts         = [u.raw  for u in task_utt_objects]

                if not clean_utts:
                    continue

                first_utt       = task_utt_objects[0]
                g_marker        = getattr(first_utt, "g_marker", "")
                ext             = DiscourseMetricExtractor(task=task, marker=g_marker)
                task_dict[task] = ext.compute(
                    clean_utts,
                    raw_utterances    = raw_utts,
                    utterance_objects = task_utt_objects,
                )

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

 
    # Compute participant split here so Mistral only trains on train participants
    longitudinal          = find_longitudinal_participants(transcripts)
    longitudinal_base_ids = list(longitudinal.keys())
    logger.info(f"Found {len(longitudinal)} longitudinal participants (2+ sessions)")

    unique_participants          = list({
        extract_metadata_from_transcript(t)["participant_id"]
        for t in transcripts
    })
    train_ids, val_ids, test_ids = split_participants(unique_participants, longitudinal_base_ids)
    train_participant_set        = set(train_ids)

 
    if args.skip_mistral:
        logger.info("\n[3/5] Skipping Mistral (--skip_mistral flag set). Using zeros.")
        all_surprisals = [0.0] * len(all_task_metrics)
    else:
        logger.info("\n[3/5] Computing Mistral surprisal scores...")

        scorer = MistralScorer(
            model_name      = "mistralai/Mistral-7B-v0.1",
            device          = args.device,
            checkpoint_path = "outputs/dae/mistral_checkpoint",
        )

        # Only use training participants for fine-tuning — no leakage
        train_texts: List[str] = []
        for t in transcripts:
            meta = extract_metadata_from_transcript(t)
            if meta["participant_id"] not in train_participant_set:
                continue
            text = " ".join(u.text for u in t.utterances if u.text.strip())
            if text.strip():
                train_texts.append(text)

        split_idx = int(len(train_texts) * 0.9)
        train_sub = train_texts[:split_idx]
        val_sub   = train_texts[split_idx:]

        if train_sub and val_sub:
            scorer.fine_tune(
                train_utterances = train_sub,
                val_utterances  = val_sub,
                output_dir = None,
                num_epochs = 1,
                batch_size = 16,
                learning_rate = 2e-5,
                mlm_probability = 0.15,
                warmup_ratio = 0.06,
                weight_decay = 0.01,
                early_stopping_patience = 3,
            )
        else:
            logger.warning("Not enough utterances for fine-tuning.")

        scorer.load(from_checkpoint=True)

        surprisal_map:  Dict[str, float] = {}
        session_id_set = set(all_session_ids)

        try:
            transcript_iter = tqdm(transcripts, desc="Scoring transcripts", unit="transcript")
        except NameError:
            transcript_iter = transcripts

        for t in transcript_iter:
            if t.session_id not in session_id_set:
                continue
            utts = [u.text for u in t.utterances if u.text.strip()]
            surprisal_map[t.session_id] = scorer.mean_surprisal(utts) if utts else 0.0

        all_surprisals = [surprisal_map.get(sid, 0.0) for sid in all_session_ids]
        logger.info(f"Surprisal scores computed. Mean: {np.mean(all_surprisals):.3f}")

 
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
            participant_id  = meta["participant_id"],
            aphasia_subtype = meta["aphasia_subtype"],
            wab_aq          = meta["wab_aq"],
        )
        all_profiles.append(profile)
        profile_metadata.append(meta)

    logger.info(f"Participants: {len(all_profiles)} aphasia (controls already removed)")

    if not all_task_metrics:
        logger.error("No aphasia transcripts found. Exiting.")
        return

    train_mask         = [p.participant_id in train_participant_set for p in all_profiles]
    train_task_metrics = [m for m, f in zip(all_task_metrics,    train_mask) if f]
    train_surprisals   = [s for s, f in zip(all_surprisals,      train_mask) if f]
    train_profiles     = [p for p, f in zip(all_profiles,        train_mask) if f]
    train_signals      = [s for s, f in zip(all_session_signals, train_mask) if f]

    if not train_task_metrics:
        logger.warning("No training metrics found — fitting scaler on all aphasia data.")
        train_task_metrics = all_task_metrics
        train_surprisals   = all_surprisals
        train_profiles     = all_profiles
        train_signals      = all_session_signals

    state_builder = PatientStateBuilder(scaler_path="outputs/dae/scaler.npz")
    state_builder.fit(
        train_task_metrics,
        train_surprisals,
        train_profiles,
        all_signals = train_signals,
    )

    state_vectors = state_builder.build_batch(
        all_task_metrics,
        all_surprisals,
        all_profiles,
        all_signals = all_session_signals,
    )

    logger.info(f"State vectors shape: {state_vectors.shape}")
    logger.info(f"State dim: {state_vectors.shape[1]} (expected {state_builder.state_dim()})")

 
    logger.info("\n[5/5] Saving outputs...")

    np.savez(
        str(output_dir / "state_vectors.npz"),
        state_vectors = state_vectors,
        session_ids   = np.array(all_session_ids),
    )

    with open(output_dir / "patient_profiles.json", "w") as f:
        json.dump(profile_metadata, f, indent=2)

    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump(longitudinal, f, indent=2)

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    metrics_summary = {
        "n_transcripts_aphasia":       len(all_task_metrics),
        "n_controls_removed":          n_total - len(transcripts),
        "n_longitudinal_participants": len(longitudinal),
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
        "mean_maze_rate":     float(np.mean([sig.maze_rate for sig in all_session_signals])),
        "mean_surprisal":     float(np.mean(all_surprisals)),
        "failed_transcripts": failed,
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