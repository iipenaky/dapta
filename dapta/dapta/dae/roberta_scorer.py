"""
dae/roberta_scorer.py
---------------------
RoBERTa fine-tuning on AphasiaBank transcripts and per-utterance surprisal scoring.

RoBERTa is fine-tuned via masked language modelling (MLM) to adapt its
language model priors to the distributional properties of aphasic speech
(paraphasias, neologisms, agrammatism underrepresented in the original
RoBERTa pre-training corpus).

Surprisal score = negative log-likelihood of an utterance given its context.
Higher surprisal ↔ more unexpected/atypical language ↔ more severe aphasia.

References
----------
Liu et al. (2019). RoBERTa. arXiv:1907.11692.
Rezaii et al. (2024). Scientific Reports, 14, 15626.
Wolf et al. (2020). Hugging Face Transformers. EMNLP Demos.
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        RobertaTokenizerFast,
        RobertaForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from dapta.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AphasiaTranscriptDataset(Dataset):
    """
    PyTorch dataset of AphasiaBank utterances for MLM fine-tuning.
    Each sample is a single utterance encoded by RoBERTa tokenizer.
    """

    def __init__(
        self,
        utterances: List[str],
        tokenizer: "RobertaTokenizerFast",
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            utterances,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            key: val[idx]
            for key, val in self.encodings.items()
        }


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class RoBERTaScorer:
    """
    Manages RoBERTa fine-tuning on AphasiaBank and computes surprisal scores.

    Parameters
    ----------
    model_name      : HuggingFace model ID (default: "roberta-base")
    checkpoint_path : Where to save/load the fine-tuned model
    device          : "cuda", "cpu", or None (auto-detect)
    """

    MODEL_NAME = "roberta-base"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required for RoBERTaScorer. "
                "Install: pip install torch transformers"
            )
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional["RobertaTokenizerFast"] = None
        self.model: Optional["RobertaForMaskedLM"] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, from_checkpoint: bool = True) -> "RoBERTaScorer":
        """
        Load tokenizer and model.
        If from_checkpoint=True and checkpoint exists, load fine-tuned model.
        Otherwise, load the original roberta-base.
        """
        if (
            from_checkpoint
            and self.checkpoint_path is not None
            and self.checkpoint_path.exists()
        ):
            model_path = str(self.checkpoint_path)
            logger.info(f"Loading fine-tuned RoBERTa from {model_path}")
        else:
            model_path = self.model_name
            logger.info(f"Loading base {self.model_name} from HuggingFace hub")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        train_utterances: List[str],
        val_utterances: List[str],
        output_dir: Optional[str | Path] = None,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        mlm_probability: float = 0.15,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 2,
    ) -> "RoBERTaScorer":
        """
        Fine-tune RoBERTa on AphasiaBank utterances via MLM.

        Parameters
        ----------
        train_utterances : List of training utterance strings
        val_utterances   : List of validation utterance strings
        output_dir       : Directory to save fine-tuned model
        """
        if output_dir is None:
            output_dir = self.checkpoint_path or Path("models/roberta_aphasiabank")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Fine-tuning RoBERTa on {len(train_utterances)} train / "
            f"{len(val_utterances)} val utterances."
        )

        # Load base model if not already loaded
        if self.tokenizer is None:
            self.load(from_checkpoint=False)

        train_dataset = AphasiaTranscriptDataset(
            train_utterances, self.tokenizer, max_length=256
        )
        val_dataset = AphasiaTranscriptDataset(
            val_utterances, self.tokenizer, max_length=256
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            ],
        )

        logger.info("Starting RoBERTa MLM fine-tuning...")
        trainer.train()

        # Save final model
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Fine-tuned model saved to {output_dir}")

        self.checkpoint_path = output_dir
        return self

    # ------------------------------------------------------------------
    # Surprisal scoring
    # ------------------------------------------------------------------

    def compute_surprisal(
        self,
        utterances: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Compute per-utterance surprisal (negative log-likelihood) scores.

        Uses the pseudo-log-likelihood (PLL) approach: mask each token one
        at a time and sum the log-probabilities across all positions.

        Parameters
        ----------
        utterances : List of utterance strings
        batch_size : Number of utterances to process at once

        Returns
        -------
        List of surprisal scores (higher = more atypical/unexpected)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        surprisals = []
        self.model.eval()

        for utt in utterances:
            if not utt.strip():
                surprisals.append(0.0)
                continue
            surprisals.append(self._utterance_surprisal(utt))

        return surprisals

    def mean_surprisal(self, utterances: List[str]) -> float:
        """Compute mean surprisal across all utterances in a transcript."""
        scores = self.compute_surprisal(utterances)
        valid = [s for s in scores if s > 0]
        return float(np.mean(valid)) if valid else 0.0

    def _utterance_surprisal(self, text: str) -> float:
        """Pseudo-log-likelihood surprisal for a single utterance."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        n_tokens = input_ids.shape[1] - 2  # Exclude BOS/EOS

        if n_tokens <= 0:
            return 0.0

        # Build all masked versions at once
        batch = input_ids.repeat(n_tokens, 1)  # (n_tokens, seq_len)
        for i in range(n_tokens):
            batch[i, i + 1] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(batch)  # ONE call instead of n_tokens calls
            logits = outputs.logits      # (n_tokens, seq_len, vocab)

        total_nll = 0.0
        for i in range(n_tokens):
            probs = torch.softmax(logits[i, i + 1], dim=-1)
            true_id = input_ids[0, i + 1].item()
            total_nll += -math.log(probs[true_id].item() + 1e-10)

        return total_nll / n_tokens # Mean per-token NLL
