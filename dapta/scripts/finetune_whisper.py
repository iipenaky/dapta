"""
Fine-tune OpenAI Whisper-base on AphasiaBank audio for improved
transcription of aphasic speech.

Pipeline:
  1. Extract audio from MP4/MOV files using ffmpeg
  2. Align audio with .cha transcripts (PAR utterances only)
  3. Fine-tune whisper-base using HuggingFace Trainer
  4. Evaluate on held-out test split (WER)

Usage (on university A100 cluster):
  python scripts/finetune_whisper.py \
      --audio_dir data/aphasiabank_audio \
      --cha_dir data/aphasiabank \
      --output_dir outputs/whisper \
      --epochs 5 \
      --batch_size 16

Requirements:
  pip install transformers datasets accelerate jiwer librosa soundfile
  apt-get install ffmpeg  (or module load ffmpeg on cluster)
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports — checked at runtime so script can be inspected without installing
# ---------------------------------------------------------------------------
try:
    import librosa
    import soundfile as sf
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import Dataset, DatasetDict, Audio
    import evaluate
    _DEPS_OK = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install transformers datasets accelerate jiwer librosa soundfile evaluate")
    _DEPS_OK = False


SAMPLE_RATE = 16_000  # Whisper expects 16kHz mono
MODEL_NAME = "openai/whisper-base"

# CHAT markup to strip before using as Whisper targets
_CHAT_NOISE = re.compile(
    r"&[=+\-*][^\s]+"
    r"|<[^>]+>\s*\[.*?\]"
    r"|\[.*?\]"
    r"|\+[/\\.]+"
    r"|&[-+][a-z_]+"
    r"|\(\w+\)"
    r"|[<>]"
    r"|\s{2,}"
)


# ---------------------------------------------------------------------------
# Step 1: Extract audio from MP4/MOV → WAV
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Extract mono 16kHz WAV from MP4/MOV using ffmpeg.
    Returns True on success.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",           # mono
        "-ar", str(sample_rate),
        "-vn",                # no video
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def batch_extract_audio(
    audio_dir: Path,
    wav_dir: Path,
    extensions: Tuple[str, ...] = (".mp4", ".mov", ".MP4", ".MOV"),
) -> Dict[str, Path]:
    """
    Extract all video files in audio_dir to WAV.
    Returns dict mapping stem -> wav_path.
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    stem_to_wav = {}

    video_files = [f for f in audio_dir.rglob("*") if f.suffix in extensions]
    print(f"Found {len(video_files)} video files.")

    for i, vf in enumerate(video_files):
        wav_path = wav_dir / f"{vf.stem}.wav"
        if wav_path.exists():
            stem_to_wav[vf.stem.lower()] = wav_path
            continue
        ok = extract_audio(vf, wav_path)
        if ok:
            stem_to_wav[vf.stem.lower()] = wav_path
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{len(video_files)} files...")
        else:
            print(f"  WARNING: Failed to extract {vf.name}")

    print(f"Extracted {len(stem_to_wav)} audio files.")
    return stem_to_wav


# ---------------------------------------------------------------------------
# Step 2: Parse .cha transcripts → (audio_path, transcript) pairs
# ---------------------------------------------------------------------------

def clean_chat_text(raw: str) -> str:
    """Strip CHAT markup, return clean text."""
    text = _CHAT_NOISE.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def parse_cha_for_whisper(cha_path: Path) -> List[str]:
    """
    Extract PAR utterances from a .cha file as clean text.
    Returns list of utterance strings.
    """
    utterances = []
    with open(cha_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("*PAR:"):
                raw = line[5:].strip()
                # Remove timing codes like 12345_67890
                raw = re.sub(r"\d+_\d+", "", raw)
                clean = clean_chat_text(raw)
                if clean and len(clean.split()) >= 2:
                    utterances.append(clean)
    return utterances


def build_transcript_map(cha_dir: Path) -> Dict[str, str]:
    """
    Build dict mapping file stem -> full PAR transcript text.
    """
    stem_to_transcript = {}
    for cha_file in cha_dir.rglob("*.cha"):
        utterances = parse_cha_for_whisper(cha_file)
        if utterances:
            full_text = " ".join(utterances)
            stem_to_transcript[cha_file.stem.lower()] = full_text
    print(f"Built transcripts for {len(stem_to_transcript)} sessions.")
    return stem_to_transcript


# ---------------------------------------------------------------------------
# Step 3: Build HuggingFace Dataset
# ---------------------------------------------------------------------------

def load_audio_array(wav_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load WAV file as float32 numpy array at target sample rate."""
    audio, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def build_dataset(
    stem_to_wav: Dict[str, Path],
    stem_to_transcript: Dict[str, str],
    splits_path: Optional[Path] = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> DatasetDict:
    """
    Build train/val/test DatasetDict from aligned (audio, transcript) pairs.
    """
    # Find matching pairs
    common_stems = set(stem_to_wav.keys()) & set(stem_to_transcript.keys())
    print(f"Matched {len(common_stems)} audio+transcript pairs.")

    if len(common_stems) == 0:
        raise ValueError(
            "No matching audio+transcript pairs found. "
            "Check that audio filenames match .cha filenames."
        )

    # Use DAE splits if available
    train_stems, val_stems, test_stems = set(), set(), set()
    if splits_path and splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        train_stems = set(s.lower() for s in splits.get("train", []))
        val_stems = set(s.lower() for s in splits.get("val", []))
        test_stems = set(s.lower() for s in splits.get("test", []))
        print(f"Using DAE splits: {len(train_stems)} train, "
              f"{len(val_stems)} val, {len(test_stems)} test")
    else:
        # Random split
        stems = sorted(common_stems)
        np.random.shuffle(stems)
        n = len(stems)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_stems = set(stems[:n_test])
        val_stems = set(stems[n_test:n_test + n_val])
        train_stems = set(stems[n_test + n_val:])

    def make_records(stem_set: set) -> List[dict]:
        records = []
        for stem in stem_set & common_stems:
            try:
                audio = load_audio_array(stem_to_wav[stem])
                records.append({
                    "audio": {"array": audio, "sampling_rate": SAMPLE_RATE},
                    "sentence": stem_to_transcript[stem],
                    "stem": stem,
                })
            except Exception as e:
                print(f"  WARNING: Could not load {stem}: {e}")
        return records

    print("Loading audio arrays...")
    train_records = make_records(train_stems)
    val_records = make_records(val_stems)
    test_records = make_records(test_stems)

    print(f"Dataset: {len(train_records)} train, "
          f"{len(val_records)} val, {len(test_records)} test")

    return DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
        "test": Dataset.from_list(test_records),
    })


# ---------------------------------------------------------------------------
# Step 4: Preprocessing for Whisper
# ---------------------------------------------------------------------------

def make_prepare_fn(processor: "WhisperProcessor"):
    """Return a preprocessing function for the HuggingFace dataset."""
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Compute log-mel spectrogram
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]
        # Tokenise transcript
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch
    return prepare_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad input features and labels for Whisper."""
    processor: "WhisperProcessor"

    def __call__(self, features: List[dict]) -> dict:
        from transformers import WhisperProcessor
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Step 5: WER metric
# ---------------------------------------------------------------------------

def make_compute_metrics_fn(processor: "WhisperProcessor"):
    """Return metrics computation function for Trainer."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper on AphasiaBank")
    p.add_argument("--audio_dir", required=True,
                   help="Directory containing MP4/MOV files")
    p.add_argument("--cha_dir", default="data/aphasiabank",
                   help="Directory containing .cha transcript files")
    p.add_argument("--wav_dir", default="data/aphasiabank_wav",
                   help="Directory to save extracted WAV files")
    p.add_argument("--output_dir", default="outputs/whisper",
                   help="Directory to save fine-tuned model")
    p.add_argument("--splits_path", default="outputs/dae/splits.json",
                   help="DAE splits.json for consistent train/test split")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=4000,
                   help="Max training steps. Overrides epochs if set.")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use FP16 (recommended for A100)")
    p.add_argument("--skip_extract", action="store_true",
                   help="Skip audio extraction if WAVs already exist")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    if not _DEPS_OK:
        return

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    audio_dir = Path(args.audio_dir)
    cha_dir = Path(args.cha_dir)
    wav_dir = Path(args.wav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAPTA — Whisper Fine-tuning on AphasiaBank")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Extract audio
    # ------------------------------------------------------------------
    if not args.skip_extract:
        print("\n[1/5] Extracting audio from MP4/MOV files...")
        stem_to_wav = batch_extract_audio(audio_dir, wav_dir)
    else:
        print("\n[1/5] Loading existing WAV files...")
        stem_to_wav = {f.stem.lower(): f for f in wav_dir.glob("*.wav")}
        print(f"  Found {len(stem_to_wav)} WAV files.")

    # ------------------------------------------------------------------
    # 2. Parse transcripts
    # ------------------------------------------------------------------
    print("\n[2/5] Parsing .cha transcripts...")
    stem_to_transcript = build_transcript_map(cha_dir)

    # ------------------------------------------------------------------
    # 3. Build dataset
    # ------------------------------------------------------------------
    print("\n[3/5] Building dataset...")
    splits_path = Path(args.splits_path) if args.splits_path else None
    dataset = build_dataset(stem_to_wav, stem_to_transcript, splits_path)

    # ------------------------------------------------------------------
    # 4. Load model and processor
    # ------------------------------------------------------------------
    print(f"\n[4/5] Loading {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="English", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Preprocess dataset
    print("  Preprocessing dataset (computing log-mel spectrograms)...")
    prepare_fn = make_prepare_fn(processor)
    dataset = dataset.map(
        prepare_fn,
        remove_columns=["audio", "sentence", "stem"],
        num_proc=4,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics_fn(processor)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[5/5] Fine-tuning Whisper...")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_path = output_dir / "whisper-aphasia-final"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nModel saved to {final_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Test WER: {test_results.get('eval_wer', 'N/A'):.4f}")

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Whisper fine-tuning complete.")
    print(f"Model saved to: {final_path}")
    print(f"Test WER: {test_results.get('eval_wer', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
