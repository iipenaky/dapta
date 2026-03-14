"""
RoBERTa fine-tuning on AphasiaBank transcripts and per-utterance surprisal scoring.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (RobertaTokenizerFast,RobertaForMaskedLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer,EarlyStoppingCallback)

from dapta.utils.logger import get_logger

logger = get_logger(__name__)

class AphasiaTranscriptDataset(Dataset):
    def __init__(self,utterances,tokenizer: "RobertaTokenizerFast",max_length: int):
        self.encodings = tokenizer(
            utterances,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int):
        return {key: val[idx] for key, val in self.encodings.items()}


class RoBERTaScorer:
    """
    Manages RoBERTa fine-tuning on AphasiaBank and computes surprisal scores.

    Parameters
    ----------
        model_name      : HuggingFace model ID. Default is distilroberta-base
        checkpoint_path : Where to save/load the fine-tuned model.
        device          : "cuda", "cpu", or None (auto-detect).
    """

    def __init__(self,model_name, checkpoint_path, device):
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional["RobertaTokenizerFast"] = None
        self.model: Optional["RobertaForMaskedLM"] = None

    def load(self, from_checkpoint) :
        if (from_checkpoint and self.checkpoint_path is not None and self.checkpoint_path.exists()):
            model_path = str(self.checkpoint_path)
            logger.info(f"Loading fine-tuned model from {model_path}")
        else:
            model_path = self.model_name
            logger.info(f"Loading {self.model_name} from HuggingFace hub")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        return self

    def fine_tune( self, train_utterances, val_utterances, output_dir, num_epochs, batch_size, learning_rate,mlm_probability, warmup_ratio,weight_decay,early_stopping_patience):
        if output_dir is None:
            output_dir = self.checkpoint_path or Path("outputs/dae/roberta_checkpoint")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Fine-tuning {self.model_name} on {len(train_utterances)} train / "
            f"{len(val_utterances)} val utterances."
        )

        if self.tokenizer is None:
            self.load(from_checkpoint=False)

        train_dataset = AphasiaTranscriptDataset(train_utterances, self.tokenizer)
        val_dataset   = AphasiaTranscriptDataset(val_utterances,   self.tokenizer)

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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        logger.info("Starting fine-tuning...")
        trainer.train()
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Model saved to {output_dir}")
        self.checkpoint_path = output_dir
        return self

    def compute_surprisal(self, utterances):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        surprisals = []
        self.model.eval()

        try:
            from tqdm import tqdm
            utt_iter = tqdm(utterances, desc="Scoring utterances", unit="utt")
        except ImportError:
            utt_iter = utterances

        for utt in utt_iter:
            if not utt.strip():
                surprisals.append(0.0)
                continue
            surprisals.append(self.utterance_surprisal(utt))

        return surprisals

    def mean_surprisal(self, utterances):
        scores = self.compute_surprisal(utterances)
        valid = [s for s in scores if s > 0]
        return float(np.mean(valid)) if valid else 0.0

    def utterance_surprisal(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        n_tokens = input_ids.shape[1] - 2

        if n_tokens <= 0:
            return 0.0

        batch = input_ids.repeat(n_tokens, 1)
        for i in range(n_tokens):
            batch[i, i + 1] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(batch)
            logits  = outputs.logits

        total_nll = 0.0
        for i in range(n_tokens):
            probs   = torch.softmax(logits[i, i + 1], dim=-1)
            true_id = input_ids[0, i + 1].item()
            total_nll += -math.log(probs[true_id].item() + 1e-10)

        return total_nll / n_tokens