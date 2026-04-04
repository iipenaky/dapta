import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dapta.utils.logger import get_logger

logger = get_logger(__name__)


class MistralScorer:
    """
    Drop-in replacement for the original RoBERTaScorer.
    Uses Mistral-7B (causal LM) for exact token-level log-likelihood surprisal.
    No masking loop needed — one forward pass per utterance.
    """

    def __init__(self, model_name, checkpoint_path, device):
        self.model_name = model_name or "mistralai/Mistral-7B-v0.1"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def load(self, from_checkpoint):
        from peft import PeftModel

        if (from_checkpoint
                and self.checkpoint_path is not None
                and self.checkpoint_path.exists()):
            model_path = str(self.checkpoint_path)
            logger.info(f"Loading base Mistral + LoRA adapter from {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None,
            )
            # Re-attach and merge the saved LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        else:
            logger.info(f"Loading {self.model_name} from HuggingFace hub")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None,
            )

        self.model.eval()
        logger.info("Mistral loaded successfully.")
        return self
    
    def fine_tune(
        self,
        train_utterances,
        val_utterances,
        output_dir,
        num_epochs,
        batch_size,
        learning_rate,
        mlm_probability,
        warmup_ratio,
        weight_decay,
        early_stopping_patience,
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
    ):
        from torch.utils.data import Dataset
        from transformers import (
            DataCollatorForLanguageModeling,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        if output_dir is None:
            output_dir = self.checkpoint_path or Path("outputs/dae/mistral_checkpoint")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Fine-tuning Mistral (LoRA) on {len(train_utterances)} train / "
            f"{len(val_utterances)} val utterances."
        )

        if self.tokenizer is None:
            self.load(from_checkpoint=False)

        # --- LoRA config ---
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
        )
        peft_model = get_peft_model(self.model, lora_config)
        peft_model.print_trainable_parameters()   # e.g. "trainable: 4.2M / 7B (0.06%)"

        class CausalDataset(Dataset):
            def __init__(self, utterances, tokenizer):
                self.encodings = tokenizer(
                    utterances,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )

            def __len__(self):
                return self.encodings["input_ids"].shape[0]

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = item["input_ids"].clone()
                return item

        train_dataset = CausalDataset(train_utterances, self.tokenizer)
        val_dataset   = CausalDataset(val_utterances,   self.tokenizer)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
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
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        logger.info("Starting Mistral LoRA fine-tuning...")
        trainer.train()

        # Save only the LoRA adapter weights (tiny — a few MB)
        peft_model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"LoRA adapter saved to {output_dir}")

        # Merge adapters back into base weights for inference
        self.model = peft_model
        self.checkpoint_path = output_dir
        return self
    def compute_surprisal(self, utterances):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        surprisals = []
        self.model.eval()

        try:
            from tqdm import tqdm
            utt_iter = tqdm(utterances, desc="Scoring utterances (Mistral)", unit="utt")
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

    def utterance_surprisal(self, text: str) -> float:
        """
        Exact causal log-likelihood surprisal.
        One forward pass — no masking loop needed.
        Returns mean NLL per token (nats).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        n_tokens = input_ids.shape[1] - 1   # exclude first token (no prediction target)

        if n_tokens <= 0:
            return 0.0

        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
            # outputs.loss = mean NLL over all tokens already
            nll = outputs.loss.item()

        return nll