

**PRD 2: Trainer Adaptation for Environment Rollouts**

**File:** `trainer_update.md`

**1. Goal/Objective:**
Modify the `GRPOEnvTrainer` to:
*   Consume datasets where each row specifies the environment class, configuration, and task data.
*   Dynamically instantiate and cache environment objects based on the dataset information.
*   Orchestrate multi-environment rollouts by calling the refactored `env.run_trial` method on appropriate environment instances.
*   Process the padded, per-token trajectory data returned by the environment (including rewards and masks).
*   Calculate policy log probabilities (`old`, `ref`, current) over the full trajectories.
*   Compute the GRPO loss using the per-token rewards (as advantages initially) and applying the agent token mask correctly.

**2. Background/Motivation:**
Following the environment refactoring (PRD 1), the environment now handles rollouts and provides rich, padded trajectory data including internal rewards. The trainer needs to adapt to this new interface. It must shift from generating single completions and calling external reward functions to managing environment instances, consuming trajectory data, and calculating the RL loss based on rewards and masks provided by the environment. This enables training on full multi-step interactions and across diverse environments.

**3. Scope:**
*   **In Scope:**
    *   Modifying `GRPOEnvTrainer` (`__init__`, `_generate_and_score_completions`, `compute_loss`).
    *   Implementing dynamic environment instantiation and caching within the trainer.
    *   Implementing logic to group batch items by environment type/config.
    *   Implementing logic to merge results from different environment groups.
    *   Adapting log probability calculation for full trajectories.
    *   Adapting loss calculation to use per-token rewards and agent masks.
    *   Updating trainer unit tests.
*   **Out of Scope:**
    *   Environment implementation details (assumed complete and correct from PRD 1).
    *   Agent implementation details.
    *   Implementing value functions or advanced advantage estimation (GAE) in the trainer.
    *   Large-scale training runs or hyperparameter tuning.

**4. Requirements:**

*   **R1: Trainer `__init__` Update:**
    *   Modify `GRPOEnvTrainer.__init__` signature: Remove `env` and `reward_funcs` parameters. Add required `tokenizer: PreTrainedTokenizerBase` and `agent: Agent` parameters.
    *   Store `self.tokenizer` and `self.agent`.
    *   Initialize an empty environment cache: `self.env_cache = {}`.
    *   Initialize `self.ref_model` based on `self.agent`'s policy model.
*   **R2: `_generate_and_score_completions` Overhaul:**
    *   Process input `List[Dict[{"env_class_path": ..., "env_config": ..., "task_data": ...}]]`.
    *   Gather all inputs to the main process (`gather_object`).
    *   On the main process:
        *   Implement grouping logic: Group inputs by `(env_class_path, env_config)`.
        *   Implement dynamic environment instantiation: For each group, import the class from `env_class_path`, instantiate it using `env_config` and `self.tokenizer`, and store it in `self.env_cache`.
        *   Call the appropriate `env_instance.run_trial(task_data_list=group_task_data, agent=self.agent, num_rollouts=self.num_generations)`.
        *   Store results per group.
        *   Implement merging logic: Combine results from all groups back into a single dictionary (`final_results`) preserving the original batch order. Ensure data remains padded.
    *   Broadcast the merged `final_results` dictionary.
    *   Slice the data for the local process.
    *   Convert received lists/data to `torch.Tensor` on the correct `device` (e.g., `padded_token_ids`, `padded_attn_mask`, `padded_agent_mask`, `padded_rewards`).
    *   Calculate `old_per_token_logps` and `ref_per_token_logps` using the full `padded_token_ids` and `padded_attn_mask`.
    *   Set `advantages = padded_rewards` (initial approach).
    *   Remove all logic related to external `reward_funcs`.
    *   Update logging (`self._metrics`, `self._textual_logs`) based on available trajectory data (e.g., use `final_rewards` from env results).
    *   Return the required dictionary containing padded tensors: `{"prompt_completion_ids": padded_token_ids, "attention_mask": padded_attn_mask, "agent_token_mask": padded_agent_mask, "advantages": advantages, "old_per_token_logps": ..., "ref_per_token_logps": ...}`.
*   **R3: `compute_loss` Adaptation:**
    *   Override `GRPOEnvTrainer.compute_loss`.
    *   Retrieve inputs from the dictionary returned by `_generate_and_score_completions`.
    *   Calculate current policy `per_token_logps` for the full trajectories.
    *   Apply the `agent_token_mask` element-wise when calculating loss components (policy loss, KL divergence) before summing/averaging over the sequence dimension. Ensure only agent tokens contribute to the loss gradients.

**5. Testing Plan:**

*   **T1: Unit Test - `__init__`:** Test `GRPOEnvTrainer.__init__` initializes correctly, stores agent/tokenizer, and creates an empty `env_cache`.
*   **T2: Unit Test - `_generate_and_score_completions` (Mocked Env):**
    *   Create a mock `env` class with a mock `run_trial` that returns correctly structured *padded* data.
    *   Create a sample input batch with multiple environment types/configs.
    *   Test the grouping logic correctly separates tasks.
    *   Test dynamic instantiation (check `EnvClass(...)` is called with correct args) and caching (check it's called only once per unique env).
    *   Test that `env_instance.run_trial` is called with the correct `task_data_list`, `agent`, and `num_rollouts`.
    *   Test the merging logic reconstructs the batch order correctly.
    *   Verify the final returned dictionary structure and tensor dtypes/devices after broadcasting/slicing.
    *   Verify log probability calculations are called with full trajectory data.
    *   Verify advantages are assigned from the mock rewards.
*   **T3: Unit Test - `compute_loss`:**
    *   Provide dummy/mock trajectory data (IDs, masks, rewards/advantages, logps) matching the output of `_generate_and_score_completions`. Include agent vs. env tokens in the masks.
    *   Verify that the loss calculation correctly applies the `agent_token_mask` (e.g., assert gradients are zero for non-agent tokens, or compare loss values with/without masking). Check both policy and KL terms if applicable.
*   **T4: Integration Test - Trainer Step:**
    *   Use a *real*, refactored environment (from PRD 1) and a dummy/simple `agent`.
    *   Create a small dataset (`.hf` format or `Dataset.from_dict`) conforming to the new schema (potentially mixing 1-2 simple env types).
    *   Instantiate `GRPOEnvTrainer` with the `agent` and `tokenizer`.
    *   Run a single training step (`trainer.train(max_steps=1)` or manually call `trainer.training_step`).
    *   Verify the step completes without runtime errors.
    *   Check logs for basic metrics (e.g., loss, rewards).

**6. Success Metrics:**
*   `GRPOEnvTrainer` can be initialized without an `env` object.
*   `_generate_and_score_completions` successfully processes batches with mixed environment types, calls the correct `env.run_trial`, and returns structured tensor data.
*   `compute_loss` correctly applies the agent mask.
*   A single training step (T4) completes successfully using a dataset with the new schema and a refactored environment.
