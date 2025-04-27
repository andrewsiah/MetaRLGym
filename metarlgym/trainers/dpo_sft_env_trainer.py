"""
offline_dpo_trainer.py
======================

A lightweight **offline** Direct Preference Optimization (**DPO**) trainer with
an optional **α‑scaled SFT term**.  When `alpha > 0` the loss becomes

    **L = L_DPO + α · L_SFT**

where `L_SFT` is the standard next‑token cross‑entropy on the *winning*
(completion‑of‑choice) trajectory.  Setting `alpha = 0` recovers vanilla DPO.

Key properties
--------------
* **Offline only** – the model is *never* queried for new rollouts during
  training; every `(prompt, chosen, rejected)` triplet is prepared ahead of
  time.
* **One‑line swap** for `trl.DPOTrainer`; all tokenisation, batching, PEFT, and
  distributed logic are inherited.
* **Chat‑template helper** – optionally apply the tokenizer’s chat template to
  the `prompt` column a single time at start‑up.

Dataset schema
--------------
```
{"prompt": "…", "chosen": "…", "rejected": "…"}
```

Quick‑start
-----------
```python
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig
from offline_dpo_trainer import OfflineDPOTrainer

model_name = "meta-llama/Llama-2-7b-hf"

ds  = load_dataset("json", data_files={"train":"train.jsonl","eval":"eval.jsonl"})
Tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

cfg = DPOConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8,
                learning_rate=5e-6, beta=0.1, max_length=2048, bf16=True)

trainer = OfflineDPOTrainer(
    model=model_name,
    ref_model=model_name,      # usual DPO reference
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    tokenizer=Tok,
    alpha=0.2,                 # <‑‑ weight on SFT term
    apply_chat_template=True,
)

trainer.train()
trainer.save_model("model‑dpo‑plus‑sft")
```
"""

from __future__ import annotations

from typing import Optional, Union, Any

import torch
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import DPOTrainer, DPOConfig
from trl.data_utils import maybe_apply_chat_template

__all__ = ["OfflineDPOTrainer"]


# ---------------------------------------------------------------------------
# Helper: optional chat‑template application
# ---------------------------------------------------------------------------

def _maybe_template(batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Apply the HF chat template to the *prompt* field (in‑place)."""
    batch["prompt"] = [
        maybe_apply_chat_template({"prompt": p}, tokenizer)["prompt"]  # type: ignore[arg-type]
        for p in batch["prompt"]
    ]
    return batch


# ---------------------------------------------------------------------------
# Offline DPO + α·SFT trainer
# ---------------------------------------------------------------------------

class OfflineDPOTrainer(DPOTrainer):
    """Offline Direct Preference Optimisation trainer with an *optional* SFT term.

    Parameters
    ----------
    alpha : float, default 0.0
        Weight on the supervised fine‑tuning (SFT) loss of the *chosen*
        trajectory.  `0.0` reproduces standard DPO; higher values increasingly
        bias the policy toward absolute likelihood of the preferred answer.
    apply_chat_template : bool, default True
        If *True*, run the tokenizer’s chat template over each *prompt* exactly
        once at dataset‑construction time.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        *,
        ref_model: Optional[Union[str, PreTrainedModel]] = None,
        args: Optional[DPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        alpha: float = 0.0,
        apply_chat_template: bool = True,
        **kwargs: Any,
    ) -> None:
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided – required for optional chat templating.")
        if alpha < 0:
            raise ValueError("`alpha` must be ≥ 0.")

        self.alpha = float(alpha)

        # Optional one‑off chat‑template application
        if apply_chat_template:
            if train_dataset is not None:
                train_dataset = train_dataset.map(
                    lambda batch: _maybe_template(batch, tokenizer), batched=True
                )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    lambda batch: _maybe_template(batch, tokenizer), batched=True
                )

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Loss: L = L_DPO + α·L_SFT
    # ---------------------------------------------------------------------

    def compute_loss(self, model: PreTrainedModel, inputs: dict[str, torch.Tensor], return_outputs: bool = False):  # type: ignore[override]
        # 1) DPO loss from the parent class (always computed)
        dpo_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.alpha == 0.0:
            # Vanilla DPO – early exit to save a forward pass
            if return_outputs:
                return dpo_loss, outputs
            return dpo_loss

        # 2) SFT loss on the *chosen* completion
        chosen_input_ids = inputs["chosen_input_ids"]              # (B, Lc)
        chosen_attention_mask = inputs["chosen_attention_mask"]    # (B, Lc)

        # Run model with labels to get token‑level cross‑entropy loss.
        # We *do not* mask the prompt tokens; the contribution is usually small
        # and keeps the implementation simple.  Adjust as you like for
        # fine‑grained control.
        sft_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_input_ids,
        )
        sft_loss = sft_outputs.loss                       # scalar

        # 3) Combine losses
        loss = dpo_loss + self.alpha * sft_loss

        if return_outputs:
            # Pass through DPO’s outputs to maintain metric logging, etc.
            return loss, outputs
        return loss
