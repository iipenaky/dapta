from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dapta.utils.logger import get_logger

# Module-level logger: messages will appear tagged with this file's name
logger = get_logger(__name__)


class MistralScorer:
    """
    Wraps a Mistral-7B causal language model for two purposes:

    1. **Fine-tuning**: adapts the base model to a target speech domain using
       parameter-efficient LoRA training so only a tiny fraction of weights are
       updated (typically ~0.06 % of 7 B parameters).

    2. **Surprisal scoring**: after loading (either from HuggingFace Hub or a
       saved LoRA checkpoint), computes per-utterance surprisal (mean negative
       log-likelihood per token). Higher surprisal indicates the utterance was
       less expected by the model, which can be a marker of atypical language.

    Parameters
    ----------
    model_name       : HuggingFace model ID for the base Mistral model.
                       Defaults to "mistralai/Mistral-7B-v0.1".
    checkpoint_path  : Path to a directory containing a saved LoRA adapter
                       (produced by fine_tune()). Used during load() when
                       from_checkpoint=True.
    device           : PyTorch device string ("cuda" or "cpu").
                       Auto-detected from GPU availability if not provided.
    """

    def __init__(self, model_name, checkpoint_path, device):
        # Fall back to the standard Mistral-7B base if no name is given
        self.model_name = model_name or "mistralai/Mistral-7B-v0.1"

        # Convert to a Path object for consistent file-system operations, or
        # keep as None if no checkpoint directory was specified
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Prefer GPU when available; CPU is the safe fallback
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer and model are not loaded at construction time: call load()
        # explicitly so startup cost is paid only when actually needed
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

    def load(self, from_checkpoint):
        """
        Loads the tokenizer and model into memory, then sets the model to
        evaluation mode (disables dropout and gradient tracking for inference).

        Two loading paths:

        **Checkpoint path** (from_checkpoint=True AND checkpoint_path exists):
          Loads the base Mistral weights from HuggingFace Hub, then re-attaches
          the saved LoRA adapter from checkpoint_path and merges the adapter
          weights back into the base model. Merging is done so that inference
          requires no PEFT dependency at runtime and runs at full speed.

        **Hub path** (from_checkpoint=False OR no checkpoint found):
          Downloads / uses the cached base model directly from HuggingFace Hub
          with no adapter applied. Useful for zero-shot scoring or when starting
          a new fine-tuning run.

        In both cases float16 precision is used on GPU (halves VRAM usage) and
        float32 on CPU (float16 is not well supported on most CPUs).
        """
        from peft import PeftModel

        if (from_checkpoint
                and self.checkpoint_path is not None
                and self.checkpoint_path.exists()):
            model_path = str(self.checkpoint_path)
            logger.info(f"Loading base Mistral + LoRA adapter from {model_path}")

            # The tokenizer is saved alongside the adapter so vocabulary changes
            # (e.g. added special tokens) are automatically restored
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Mistral has no dedicated padding token; reuse EOS so the collator
            # can pad batches without introducing an unknown token ID
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load the frozen base model first: adapter weights are applied next
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                # float16 halves VRAM on GPU; CPU must stay at float32
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                # "auto" spreads layers across all available GPUs automatically
                device_map="auto" if "cuda" in self.device else None,
            )
            # Re-attach the LoRA adapter and fuse its weights into the base model.
            # merge_and_unload() produces a plain nn.Module with no PEFT overhead.
            self.model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        else:
            # No saved adapter: load the vanilla base model from HuggingFace
            logger.info(f"Loading {self.model_name} from HuggingFace hub")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None,
            )

        # Switch to inference mode: disables dropout layers and tells PyTorch
        # not to build a computation graph for gradient updates
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
        warmup_ratio,
        weight_decay,
        early_stopping_patience,
        # LoRA hyperparameters: control the size and regularisation of the adapter
        lora_r: int = 16,            # rank of the low-rank decomposition matrices
        lora_alpha: int = 32,        # scaling factor applied to the adapter output
        lora_dropout: float = 0.05,  # dropout rate inside the adapter for regularisation
        lora_target_modules: list = None,  # which weight matrices to adapt (default: q, v projections)
    ):
        """
        Fine-tunes the base Mistral model on domain-specific speech utterances
        using Low-Rank Adaptation (LoRA).

        LoRA freezes all base model weights and injects small trainable rank-r
        matrices into selected attention projections. This means only ~0.06 % of
        parameters are updated, which:
          - dramatically reduces GPU memory requirements
          - speeds up training
          - prevents catastrophic forgetting of the pre-trained knowledge
          - produces a tiny adapter file (a few MB) rather than a full 7 B checkpoint

        The adapter is saved to output_dir after training. The base weights are
        NOT saved: only the delta produced by training. At inference time the
        adapter is reloaded and merged back into the base model (see load()).

        Early stopping monitors validation loss and halts training if no
        improvement is seen for `early_stopping_patience` consecutive epochs,
        restoring the best checkpoint automatically.

        Parameters
        ----------
        train_utterances         : List of raw text strings for training.
        val_utterances           : List of raw text strings for validation.
        output_dir               : Directory where the LoRA adapter will be saved.
        num_epochs               : Maximum number of training epochs.
        batch_size               : Per-device batch size for train and eval.
        learning_rate            : Peak learning rate for the AdamW optimiser.
        mlm_probability          : Passed in but not used (causal LM, not MLM).
        warmup_ratio             : Fraction of total steps used for LR warm-up.
        weight_decay             : L2 regularisation coefficient for AdamW.
        early_stopping_patience  : Stop if eval_loss does not improve for this
                                   many consecutive epochs.
        lora_r                   : Rank of the LoRA decomposition (higher = more
                                   capacity but more parameters).
        lora_alpha               : LoRA scaling factor (commonly set to 2 × lora_r).
        lora_dropout             : Dropout rate inside the adapter layers.
        lora_target_modules      : Names of the attention weight matrices to adapt.
                                   Defaults to ["q_proj", "v_proj"].
        """
        from torch.utils.data import Dataset
        from transformers import (
            DataCollatorForLanguageModeling,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        # Use the configured checkpoint path as default output location if none given
        if output_dir is None:
            output_dir = self.checkpoint_path or Path("outputs/dae/mistral_checkpoint")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Fine-tuning Mistral (LoRA) on {len(train_utterances)} train / "
            f"{len(val_utterances)} val utterances."
        )

        # Make sure the tokenizer is available before building datasets
        if self.tokenizer is None:
            self.load(from_checkpoint=False)

        # --- LoRA configuration ---
        # Tells PEFT which layers to inject adapter matrices into and how large
        # those matrices should be. TaskType.CAUSAL_LM ensures the correct
        # adapter pattern for a decoder-only language model.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # do not adapt bias terms (keeps adapter size minimal)
            target_modules=lora_target_modules or ["q_proj", "v_proj"],
        )
        # Wrap the base model with the LoRA adapter: base weights are frozen automatically
        peft_model = get_peft_model(self.model, lora_config)
        # Prints a summary like "trainable params: 4,194,304 || all params: 7,242,649,600"
        peft_model.print_trainable_parameters()

        class CausalDataset(Dataset):
            """
            A minimal PyTorch Dataset that tokenises a list of utterance strings
            into fixed-length input tensors suitable for causal language modelling.

            Each item returns the same token IDs as both `input_ids` (the model
            input) and `labels` (the prediction targets). The causal LM loss is
            computed by shifting labels one position left internally, so the model
            learns to predict each token from all previous tokens.
            """
            def __init__(self, utterances, tokenizer):
                # Tokenise the entire list at once for efficiency.
                # Sequences longer than 128 tokens are truncated; shorter ones
                # are padded to 128 so all batches have the same shape.
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
                # Retrieve a single example's tensors as a dict
                item = {k: v[idx] for k, v in self.encodings.items()}
                # Labels are a copy of input_ids: the Trainer/model handles
                # the causal shift (predicting token i+1 from tokens 0..i)
                item["labels"] = item["input_ids"].clone()
                return item

        train_dataset = CausalDataset(train_utterances, self.tokenizer)
        val_dataset   = CausalDataset(val_utterances,   self.tokenizer)

        # DataCollatorForLanguageModeling with mlm=False just stacks pre-tokenised
        # tensors into batches without any masking: appropriate for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # causal LM (next-token prediction), not masked LM
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,        # linearly ramp LR for first X% of steps
            weight_decay=weight_decay,        # L2 penalty applied to all non-bias params
            eval_strategy="epoch",            # run validation at the end of every epoch
            save_strategy="epoch",            # save a checkpoint at the end of every epoch
            load_best_model_at_end=True,      # restore the epoch with the lowest eval_loss
            metric_for_best_model="eval_loss",
            greater_is_better=False,          # lower eval_loss = better model
            logging_steps=50,                 # log training loss every 50 optimiser steps
            report_to="tensorboard",          # write loss curves to TensorBoard
            fp16=torch.cuda.is_available(),   # mixed-precision training on GPU (speeds up ~2×)
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            # Stop training early if eval_loss doesn't improve for N epochs
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        logger.info("Starting Mistral LoRA fine-tuning...")
        trainer.train()

        # Persist only the LoRA delta weights: a few MB instead of the full 14 GB base
        peft_model.save_pretrained(str(output_dir))
        # Save the tokenizer alongside the adapter so it can be restored together
        self.tokenizer.save_pretrained(str(output_dir))
        logger.info(f"LoRA adapter saved to {output_dir}")

        # Update internal state so the scorer can be used for inference immediately
        # after fine-tuning without calling load() again
        self.model = peft_model
        self.checkpoint_path = output_dir
        return self

    def compute_surprisal(self, utterances):
        """
        Computes per-utterance surprisal scores for a list of text strings.

        Each score is the mean negative log-likelihood (NLL) per token produced
        by the language model for that utterance (in nats). A higher score means
        the model found the utterance more surprising: i.e. less consistent with
        the language patterns it learned during (fine-)training.

        Empty utterances are assigned a surprisal of 0.0 and skipped.

        Uses tqdm for a progress bar if the package is installed; falls back to
        a plain iterator silently if it is not.

        Parameters
        ----------
        utterances : List of text strings to score.

        Returns
        -------
        List of float surprisal values, one per input utterance (in the same order).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        surprisals = []
        self.model.eval()

        # Wrap with a progress bar when tqdm is available for long lists
        try:
            from tqdm import tqdm
            utt_iter = tqdm(utterances, desc="Scoring utterances (Mistral)", unit="utt")
        except ImportError:
            utt_iter = utterances

        for utt in utt_iter:
            if not utt.strip():
                # Skip blank lines: assign zero surprisal as a neutral placeholder
                surprisals.append(0.0)
                continue
            surprisals.append(self.utterance_surprisal(utt))

        return surprisals

    def mean_surprisal(self, utterances):
        """
        Returns the average surprisal across all non-zero-scored utterances.

        Zero scores (from blank or skipped lines) are excluded from the mean so
        they do not artificially deflate the result. Returns 0.0 if no valid
        scores are available.
        """
        scores = self.compute_surprisal(utterances)
        valid = [s for s in scores if s > 0]
        return float(np.mean(valid)) if valid else 0.0

    def utterance_surprisal(self, text: str) -> float:
        """
        Computes the surprisal of a single utterance using one forward pass.

        Surprisal here is the mean negative log-likelihood (NLL) per token
        in nats, equivalent to per-token cross-entropy loss. It measures how
        unexpected the utterance is given the model's learned distribution:
          - Low NLL  → the model found this utterance predictable / typical
          - High NLL → the model found this utterance unusual / atypical

        Implementation note:
          For a causal (decoder-only) language model, passing `labels=input_ids`
          asks the model to compute the NLL of predicting each token from the
          tokens that precede it. The model internally shifts the labels by one
          position, so token 0 is never a prediction target (it has no context).
          `outputs.loss` is already the mean NLL across all n-1 prediction
          positions, so no manual aggregation is needed.

        Parameters
        ----------
        text : A single utterance string (whitespace-only strings are handled
               by the callers: this method expects a non-empty string).

        Returns
        -------
        Mean NLL per token in nats (float). Returns 0.0 for single-token inputs
        where there is no valid prediction target.
        """
        # Tokenise the utterance and move the tensors to the correct device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        # The number of prediction targets is one less than the sequence length
        # because the first token has no predecessor to condition on
        n_tokens = input_ids.shape[1] - 1

        if n_tokens <= 0:
            # A single-token sequence has no prediction targets: return 0
            return 0.0

        with torch.no_grad():
            # One forward pass through the model.
            # Passing labels=input_ids tells the model to compute cross-entropy
            # loss internally. outputs.loss is the mean NLL across all tokens.
            outputs = self.model(**inputs, labels=input_ids)
            nll = outputs.loss.item()

        return nll