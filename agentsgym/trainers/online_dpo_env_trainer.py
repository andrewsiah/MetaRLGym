"""
alpha_online_dpo_env_trainer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DPO + α·SFT trainer for environment‑based online preference optimisation.

• Collects (prompt, chosen, rejected) triples online from a `agentsgym` Environment.
• Optimises   L_total = L_DPO + α · L_SFT,
  where L_SFT is the token‑average NLL of the chosen completion.
• Works with vLLM back‑end, multiple weighted reward functions, PEFT, Rich/WandB logging.

Set `alpha` in `trl.DPOConfig` (default 0.2).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union, List

import torch
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from trl import DPOTrainer, DPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad
from transformers.utils import is_peft_available

from agentsgym.envs.environment import Environment
from agentsgym.agents.base import Agent
from agentsgym.agents.directoutput.direct_output_agent import DirectOutputAgent
from agentsgym.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class AlphaOnlineDPOEnvTrainer(DPOTrainer):
    r"""DPO + α·SFT trainer that gathers preference data online from a
    :class:`agentsgym.envs.environment.Environment`.

    Parameters
    ----------
    model: str | PreTrainedModel
        Causal language model to fine‑tune, or a path/identifier thereof.
    env: Environment
        Environment responsible for generating completions (uses vLLM under the hood).
    reward_funcs: callable or list[callable]
        Scoring functions returning a reward per completion. Several functions can
        be combined via `reward_weights` in :class:`trl.DPOConfig`.
    args: DPOConfig, optional
        Trainer configuration. Expose `alpha` here to set the SFT weight.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: Environment,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: Optional[DPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        agent: Optional[Agent] = None,
        **kwargs,
    ):
        # Default config ----------------------------------------------------------------
        if args is None:
            args = DPOConfig()
        if not args.use_vllm:  # type: ignore[attr-defined]
            raise ValueError("vLLM must be enabled for AlphaOnlineDPOEnvTrainer")

        # α weight for SFT term ----------------------------------------------------------
        self.alpha: float = float(getattr(args, "alpha", 0.2))

        # Normalise reward_funcs to list -------------------------------------------------
        if isinstance(reward_funcs, list):
            if not all(callable(f) for f in reward_funcs):
                raise ValueError("Each element of reward_funcs must be callable")
            self.reward_funcs: List[RewardFunc] = reward_funcs
        elif callable(reward_funcs):
            self.reward_funcs = [reward_funcs]
        else:
            raise ValueError("reward_funcs must be callable or list of callables")

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Environment & agent -----------------------------------------------------------
        self.env = env
        self.num_generations = 2  # fixed for preference pairs

        from vllm import SamplingParams

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k is not None else -1,
            min_p=self.min_p if self.min_p is not None else 0.0,
            max_tokens=self.max_completion_length,
            repetition_penalty=self.repetition_penalty,
        )

        if agent is None:
            if not self.use_vllm or self.vllm_client is None:
                raise ValueError("Need vLLM client to create DirectOutputAgent")
            self.agent = DirectOutputAgent(
                llm=self.vllm_client,
                sampling_params=self.sampling_params,
                tokenizer=self.processing_class,
            )
        else:
            self.agent = agent

        # Reward weights ----------------------------------------------------------------
        w = getattr(self.args, "reward_weights", [1.0 for _ in self.reward_funcs])
        self.reward_weights = torch.tensor(
            w, dtype=torch.float32, device=self.accelerator.device
        )

        self._metrics: dict[str, dict[str, list]] = {"train": {}, "eval": {}}
        self._last_loaded_step: int = -1

    # ------------------------------------------------------------------------------
    # Utility: sequence log-probabilities (sum over completion tokens)
    # ------------------------------------------------------------------------------
    @staticmethod
    def _sequence_logps(
        model: nn.Module,
        ids: torch.Tensor,
        mask: torch.Tensor,
        completion_len: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = model(ids, attention_mask=mask).logits
            logits = logits[:, :-1]  # shift
            target = ids[:, 1:]
            mask2 = mask[:, 1:]
            logp_tok = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            logp_tok = logp_tok[:, -completion_len:]
            mask2 = mask2[:, -completion_len:]
            return (logp_tok * mask2).sum(-1)

    # ------------------------------------------------------------------------------
    # Core: training step (DPO + α·SFT)
    # ------------------------------------------------------------------------------
    def training_step(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:  # type: ignore[override]
        device = self.accelerator.device
        prompts: List[dict[str, Any]] = [x["prompt"] for x in inputs]  # type: ignore[index]

        # Move updated model weights to vLLM
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # === 1. Generate two completions per prompt ------------------------------
        env_out = self.env.generate(
            prompts=prompts,
            llm=self.vllm_client,
            sampling_params=self.sampling_params,
            agent=self.agent,
            num_generations=self.num_generations,
        )
        cmp_ids: List[List[int]] = env_out["ids"]
        cmp_msgs: List[dict] = env_out["messages"]
        cmp_mask: List[List[int]] = env_out["mask"]

        B = len(prompts)
        cmp_ids = [cmp_ids[i * 2 : (i + 1) * 2] for i in range(B)]
        cmp_msgs = [cmp_msgs[i * 2 : (i + 1) * 2] for i in range(B)]
        cmp_mask = [cmp_mask[i * 2 : (i + 1) * 2] for i in range(B)]

        # === 2. Score with reward functions -------------------------------------
        rewards = torch.zeros(B, self.num_generations, device=device)
        other_cols = [k for k in inputs[0] if k not in ["prompt"]]  # type: ignore[index]

        for j, rf in enumerate(self.reward_funcs):
            extras = {k: [ex[k] for ex in inputs] for k in other_cols}  # type: ignore[index]
            score = torch.tensor(
                rf(prompts=prompts, completions=cmp_msgs, **extras),
                dtype=torch.float32,
                device=device,
            ).view(B, self.num_generations)
            rewards += self.reward_weights[j] * score

        chosen_idx = rewards.argmax(-1)
        reject_idx = 1 - chosen_idx

        # === 3. Tokenise ---------------------------------------------------------
        prompt_txt = [
            maybe_apply_chat_template(p, self.processing_class)["prompt"] for p in prompts
        ]
        p_in = self.processing_class(
            prompt_txt,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
        p_ids, p_mask = p_in["input_ids"], p_in["attention_mask"]

        ch_ids, ch_mask, rj_ids, rj_mask = [], [], [], []
        for i in range(B):
            ch_ids.append(torch.tensor(cmp_ids[i][chosen_idx[i]], device=device))
            ch_mask.append(torch.tensor(cmp_mask[i][chosen_idx[i]], device=device))
            rj_ids.append(torch.tensor(cmp_ids[i][reject_idx[i]], device=device))
            rj_mask.append(torch.tensor(cmp_mask[i][reject_idx[i]], device=device))

        ch_ids = pad(ch_ids, self.processing_class.pad_token_id)  # type: ignore[arg-type]
        ch_mask = pad(ch_mask, 0)
        rj_ids = pad(rj_ids, self.processing_class.pad_token_id)  # type: ignore[arg-type]
        rj_mask = pad(rj_mask, 0)

        def _concat(a, b):
            return torch.cat([a, b], dim=1)

        ch_full, rj_full = _concat(p_ids, ch_ids), _concat(p_ids, rj_ids)
        ch_full_m, rj_full_m = _concat(p_mask, ch_mask), _concat(p_mask, rj_mask)
        comp_len = ch_ids.size(1)

        # === 4. Log‑probs --------------------------------------------------------
        logp_c = self._sequence_logps(self.model, ch_full, ch_full_m, comp_len)
        logp_r = self._sequence_logps(self.model, rj_full, rj_full_m, comp_len)

        if self.ref_model is not None:
            logp_c_ref = self._sequence_logps(self.ref_model, ch_full, ch_full_m, comp_len)
            logp_r_ref = self._sequence_logps(self.ref_model, rj_full, rj_full_m, comp_len)
        else:
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                logp_c_ref = self._sequence_logps(self.model, ch_full, ch_full_m, comp_len)
                logp_r_ref = self._sequence_logps(self.model, rj_full, rj_full_m, comp_len)

        # === 5. Combined loss ----------------------------------------------------
        dpo_loss = -F.logsigmoid(
            self.beta * ((logp_c - logp_r) - (logp_c_ref - logp_r_ref))
        ).mean()

        sft_loss = (-logp_c / comp_len).mean()

        total_loss = dpo_loss + self.alpha * sft_loss

        # === 6. Optional logging -------------------------------------------------
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process:
                try:
                    print_prompt_completions_sample(
                        [prompts[0][-1]["content"]],
                        [cmp_msgs[0][chosen_idx[0]]],
                        [rewards[0, chosen_idx[0]].item()],
                        self.state.global_step,
                    )
                except Exception:
                    pass
                if is_wandb_available() and wandb.run is not None:
                    import pandas as pd
                    df = pd.DataFrame(
                        {
                            "prompt": [p[-1]["content"] for p in prompts],
                            "chosen": [cmp_msgs[i][chosen_idx[i]] for i in range(B)],
                            "rejected": [cmp_msgs[i][reject_idx[i]] for i in range(B)],
                            "reward_chosen": rewards[range(B), chosen_idx].tolist(),
                            "reward_rejected": rewards[range(B), reject_idx].tolist(),
                            "dpo_loss": dpo_loss.item(),
                            "sft_loss": sft_loss.item(),
                            "total_loss": total_loss.item(),
                            "step": self.state.global_step,
                        }
                    )
                    wandb.log({"alpha_dpo/completions": wandb.Table(dataframe=df)})

        return total_loss




"""
# ----------------------------------------------------------------------------
# Example usage (run only when executed as script)-----------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from trl import DPOConfig
    from agentsgym.envs.dummy_env import DummyEnv  # fictional example

    cfg = DPOConfig(
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        beta=0.1,
        alpha=0.2,           # <‑‑ our new hyper‑param
        use_vllm=True,
    )

    trainer = AlphaOnlineDPOEnvTrainer(
        model="meta-llama/Llama-3-8B-Instruct",
        env=DummyEnv(),
        reward_funcs=[lambda prompts, completions: [1.0] * len(prompts)],
        args=cfg,
        train_dataset=[{"prompt": [{"role": "user", "content": "Hello"}]}],
    )
    trainer.train()


### Where “batch size” lives in the trainer

| Stage | Symbol in **`DPOConfig`** | What it counts | Default |
|-------|--------------------------|----------------|---------|
| **Training forward-/back-pass** | `per_device_train_batch_size` | **Prompts** processed **per GPU** **per optimizer step** | `8`  ([DPO Trainer - Hugging Face](https://huggingface.co/docs/trl/main/en/dpo_trainer?utm_source=chatgpt.com)) |
| **Gradient accumulation** | `gradient_accumulation_steps` | Number of **optimizer-step delays** that aggregate micro-batches | `1` |
| **Multi-GPU / DDP** | `world_size` (inferred) | # processes = GPUs | – |
| **Online generation** | `num_generations` (= `2`) | **Completions per prompt** (chosen + rejected) | fixed |

Effective prompts per optimisation step  
\[
\textbf{B}_{\text{effective}}
= (\text{per\_device\_train\_batch\_size})
  \times (\text{world\_size})
  \times (\text{gradient\_accumulation\_steps})
\]

Tokens forwarded/back-propagated scale with `2 × B_effective` because every prompt produces **two** completions.

---

### Changing it safely

```python
from trl import DPOConfig
cfg = DPOConfig(
    per_device_train_batch_size = 2,      # ↓ fit into memory …
    gradient_accumulation_steps = 16,     # ↑ keep B_effective = 32 prompts
    alpha = 0.2,
    beta  = 0.1,
    use_vllm = True,
)
```

*Start big, halve the batch size on OOM, and double `gradient_accumulation_steps` to keep the same effective size*—the rule of thumb recommended in HF’s preference-optimisation guide  ([Preference Optimization for Vision Language Models with TRL](https://huggingface.co/blog/dpo_vlm?utm_source=chatgpt.com)).  

Things to watch:

| Leverage | What it buys you |
|----------|------------------|
| **Mixed / bfloat16** (`--bf16` or `--fp16`) | ~2× memory cut with negligible quality loss |
| **Gradient checkpointing** (`gradient_checkpointing=True`) | Saves ~40-60 % activations; adds ~30 % compute |
| **ZeRO-3** (`deepspeed`) | Parameter sharding → near-linear memory scaling across GPUs |
| **Sequence length cap** (`max_prompt_length`, `max_completion_length`) | Memory ∝ tokens × layers, so shortening from 2048→1024 halves activations |
| **vLLM generation micro-batch** (`vllm_max_num_batched_tokens`) | Limits *inference* RAM when sampling long answers; unrelated to training batch |

---

### Quick reference: GPU RAM vs. per-device batch for an 8 B model¹

| fp16 RAM | GC off | GC on |
|----------|--------|-------|
| 24 GB | `1` | `2` |
| 40 GB | `2` | `4` |
| 80 GB | `4` | `8` |

<sub>¹Approximate, 2048-token prompt + 2×512-token completions, AdamW 8-bit, no ZeRO.</sub>

---

#### TL;DR

* **Edit** `per_device_train_batch_size` freely—just respect your GPU memory.  
* **Compensate** with `gradient_accumulation_steps` to keep the same effective batch if you need large updates.  
* Generation is decoupled through vLLM; its micro-batching is tuned separately from the training batch.
"""