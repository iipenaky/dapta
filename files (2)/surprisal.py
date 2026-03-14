"""
surprisal.py
============
Computes per-utterance surprisal scores using a fine-tuned RoBERTa model.

Surprisal measures how unexpected a sentence is according to the language
model. We use the Pseudo-Log-Likelihood (PLL) method from Salazar et al.
(2020) because RoBERTa is a masked language model — it sees both left and
right context, so standard left-to-right probability is not correct for it.

PLL method:
  For each word in an utterance:
    1. Mask that word
    2. Ask RoBERTa to predict it from the rest of the sentence
    3. Record the negative log probability of the correct word
  Sum these across all words and divide by utterance length.
  → Higher surprisal = more unusual / less predictable speech.

For aphasia: aphasic speech tends to have higher surprisal than typical
speech because of paraphasias, agrammatism, and word-finding errors.
As therapy progresses, surprisal should decrease toward typical norms.

Fine-tuning:
  RoBERTa is fine-tuned on AphasiaBank training transcripts using the
  Masked Language Modelling (MLM) objective. This adapts it to aphasic
  speech patterns (neologisms, agrammatism) which are rare in standard
  RoBERTa training data.

Reference:
  Salazar et al. (2020). Masked Language Model Scoring. ACL 2020.
  Cong et al. (2024). Clinical efficacy of pre-trained LLMs through
    the lens of aphasia. Scientific Reports.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


# =============================================================================
# SURPRISAL SCORER
# =============================================================================

class SurprisalScorer:
    """
    Computes RoBERTa PLL surprisal for a list of utterances.

    Parameters
    ----------
    model_name      : HuggingFace model ID. Default "roberta-base".
                      Use checkpoint_path to load a fine-tuned version.
    checkpoint_path : Path to a fine-tuned model checkpoint directory.
                      If provided and exists, loaded instead of model_name.
    max_length      : Maximum token length per utterance (truncated if longer).
    """

    def __init__(
        self,
        model_name:      str           = "roberta-base",
        checkpoint_path: Optional[str] = None,
        max_length:      int           = 128,
    ):
        self.model_name      = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.max_length      = max_length
        self._model          = None
        self._tokenizer      = None
        self._device         = None

    def _load(self):
        """Lazy load — only import torch/transformers when actually needed."""
        import torch
        from transformers import RobertaTokenizerFast, RobertaForMaskedLM

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use fine-tuned checkpoint if it exists
        if self.checkpoint_path and self.checkpoint_path.exists():
            load_from = str(self.checkpoint_path)
            print(f"Loading fine-tuned RoBERTa from {load_from}")
        else:
            load_from = self.model_name
            print(f"Loading {self.model_name} from HuggingFace hub")

        self._tokenizer = RobertaTokenizerFast.from_pretrained(load_from)
        self._model     = RobertaForMaskedLM.from_pretrained(load_from)
        self._model.to(self._device)
        self._model.eval()

    def _utterance_surprisal(self, text: str) -> float:
        """
        Compute PLL surprisal for one utterance.
        Returns 0.0 for empty or very short utterances.
        """
        import torch

        if not text.strip():
            return 0.0

        inputs   = self._tokenizer(
            text,
            return_tensors  = "pt",
            truncation      = True,
            max_length      = self.max_length,
        ).to(self._device)

        input_ids = inputs["input_ids"]
        n_tokens  = input_ids.shape[1] - 2  # exclude [CLS] and [SEP]

        if n_tokens <= 0:
            return 0.0

        # Create one masked version per token
        batch = input_ids.repeat(n_tokens, 1)
        for i in range(n_tokens):
            batch[i, i + 1] = self._tokenizer.mask_token_id

        with torch.no_grad():
            logits = self._model(batch).logits

        total_nll = 0.0
        for i in range(n_tokens):
            probs    = torch.softmax(logits[i, i + 1], dim=-1)
            true_id  = input_ids[0, i + 1].item()
            prob     = probs[true_id].item()
            total_nll += -math.log(prob + 1e-10)

        return total_nll / n_tokens

    def score_utterances(self, utterances: List[str]) -> List[float]:
        """
        Compute surprisal for each utterance in the list.

        Parameters
        ----------
        utterances : list of cleaned utterance strings

        Returns
        -------
        list of float, same length as utterances
        """
        if self._model is None:
            self._load()

        scores = []
        for utt in utterances:
            try:
                scores.append(self._utterance_surprisal(utt))
            except Exception:
                scores.append(0.0)
        return scores

    def mean_surprisal(self, utterances: List[str]) -> float:
        """
        Compute mean surprisal across all non-empty utterances.
        Returns 0.0 if no scoreable utterances.

        This is the single value that goes into the state vector.
        """
        scores = self.score_utterances(utterances)
        valid  = [s for s in scores if s > 0.0]
        return round(float(np.mean(valid)), 4) if valid else 0.0


# =============================================================================
# FINE-TUNING
# =============================================================================

def finetune_roberta(
    train_utterances: List[str],
    val_utterances:   List[str],
    output_dir:       str,
    model_name:       str   = "distilroberta-base",
    epochs:           int   = 15,
    batch_size:       int   = 16,
    learning_rate:    float = 2e-5,
    mlm_probability:  float = 0.15,
    patience:         int   = 2,
) -> str:
    """
    Fine-tune RoBERTa on AphasiaBank utterances using MLM objective.

    Why fine-tune?
    RoBERTa was trained on standard English text. Aphasic speech contains
    paraphasias, neologisms, and agrammatic structures that are rare in
    standard text. Fine-tuning adapts the model to this domain so that
    surprisal scores are calibrated to aphasic language patterns rather
    than general English.

    Parameters
    ----------
    train_utterances : cleaned PAR utterances from training participants
    val_utterances   : cleaned PAR utterances from validation participants
    output_dir       : where to save the fine-tuned model
    model_name       : base model to start from
    epochs           : max training epochs (early stopping may end it sooner)
    batch_size       : per-device batch size
    learning_rate    : Adam learning rate (2e-5 from Cong et al. 2024)
    mlm_probability  : proportion of tokens masked per batch (standard = 0.15)
    patience         : early stopping patience on validation perplexity

    Returns
    -------
    str : path to the saved fine-tuned model
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        RobertaTokenizerFast,
        RobertaForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuning {model_name} on {len(train_utterances)} train / "
          f"{len(val_utterances)} val utterances.")
    print(f"Output → {output_dir}")

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model     = RobertaForMaskedLM.from_pretrained(model_name)

    class _Dataset(Dataset):
        def __init__(self, texts):
            self.enc = tokenizer(
                texts,
                truncation      = True,
                padding         = "max_length",
                max_length      = 128,
                return_tensors  = "pt",
            )
        def __len__(self):
            return self.enc["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}

    train_ds = _Dataset(train_utterances)
    val_ds   = _Dataset(val_utterances)

    collator = DataCollatorForLanguageModeling(
        tokenizer       = tokenizer,
        mlm             = True,
        mlm_probability = mlm_probability,
    )

    args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = learning_rate,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = 50,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
        callbacks     = [EarlyStoppingCallback(patience)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Fine-tuned model saved → {output_dir}")
    return str(output_dir)
