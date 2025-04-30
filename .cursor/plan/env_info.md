# Consolidated Refactoring Plan: Environment Overhaul

Look at agentsgym/temp_textarena/envs/ClassicalReasoningEvals/env.py
and agentsgym/envs/TwentyQuestions/env.py


**Core Idea:** Centralize interaction logic, tokenization, and reward calculation within the `Environment`. It will return structured, per-token trajectory data (token IDs, masks, rewards) to the `Trainer`, simplifying the Trainer's role to log probability calculation, advantage processing (if needed), and loss computation.

{
  "env_class_path": str,        # e.g., "agentsgym.envs.TwentyQuestions.TwentyQuestionsEnv"
  "env_config": Dict[str, Any], # Config for the env instance (max_steps, episodes_per_trial, etc.)
  "task_data": Dict[str, Any],  # Config for the specific task (solution word, goal state, etc.)
  # Optional: "__index__": int
}

---

**Phase 1: Environment Initialization and Configuration**

1.  **`MultistepEnv` & Subclass `__init__`:**
    *   **Remove `prompt`:** Eliminate the `prompt` argument from `MultistepEnv.__init__`. Environments generate prompts internally based on state.
    *   **Accept `tokenizer`:** Add `tokenizer: PreTrainedTokenizerBase = None` to `MultistepEnv.__init__`. Store as `self.tokenizer`. Make it optional initially for backward compatibility, guarding tokenization logic (`if self.tokenizer:`). *Goal: Make tokenizer mandatory for training.*
    *   **Accept `env_config: Dict[str, Any]`:** Pass a configuration dictionary `env_config` to `MultistepEnv.__init__`. This holds env-specific settings *constant across tasks* for this environment instance (e.g., `max_steps_per_episode`, `error_allowance`, `episodes_per_trial`, `free_shots`). `MultistepEnv` extracts common parameters (e.g., `episodes_per_trial=1`, `free_shots=0` - defaults defined here) and stores the whole `env_config` for potential subclass use. *Subclasses should use `env_config[\'max_steps_per_episode\']` to set the underlying `textarena.State.max_turns` during initialization.* Prefer specific keys like `error_allowance` over generic groups like `game_rules`.
    *   **Remove `train_dataset_size`:** No longer needed in `__init__`; dataset size is determined by the provided dataset.
    *   **Subclass Updates:** Update `TwentyQuestionsEnv.__init__` (and others) to accept `tokenizer` and `env_config`, passing them to `super().__init__`. Extract subclass-specific *configuration* params from `env_config` (e.g., `hardcore` for 20Q). Subclasses should *not* expect task-specific data (like `solution`) in `env_config`.
    *   **Remove `env_id`:** Remove `env_id` management from the constructor; handle via registration or simply by how the env object is named/used.
    *   **Remove `seed`:** Remove the global `seed` parameter from `MultistepEnv.__init__`. If episode-level determinism is needed, include a `seed` key within the `task_data` passed to `run_trial`.
    *   **System Prompts:** Handle system prompts within subclass logic (e.g., `_initialize_episode`, `_format_prompt`) based on `env_config` if needed. Remove `system_prompt` from `MultistepEnv.__init__`.

2.  **Standardize Dataset Schema:**
    *   Each dataset row (input to `run_trial`) should be a dictionary containing only the information needed to initialize *one specific task* or episode.
    *   **Required Key:** `task_data: Dict[str, Any]`. This nested dictionary holds all task-specific info (e.g., `{"solution": "apple"}` for TwentyQuestions, `{"initial_board": "...", "goal": "..."}` for a puzzle env).
    *   *(Optional Key for Mixed Datasets):* `env_type: str`. Could be used by a higher-level runner to dispatch to the correct environment type if multiple envs share a dataset. *This refactoring assumes `run_trial` is called on an already instantiated env of the correct type.*
    *   **Example `TwentyQuestions` row:** `{"task_data": {"solution": "banana"}}`
    *   **Update `get_train_dataset` Methods:** Subclasses' `get_train_dataset` methods should generate datasets conforming to this schema (e.g., `Dataset.from_dict({"task_data": [{"solution": w} for w in train_words]})`). Remove the old `"prompt"` and `"solution"` top-level columns.

---

**Phase 2: Interaction Logic (`run_trial`, `_run_complete_episode`) & Tokenization**

1.  **Modify `MultistepEnv.run_trial`:**
    *   **Signature:** `run_trial(self, inputs: List[Dict[str, Any]], agent: Agent, **kwargs)`.
        *   `inputs`: A batch of dataset rows, where each row is a dict like `{"task_data": {...}}`.
        *   Remove `llm` and `sampling_params` arguments; the agent handles interaction.
    *   **Logic:**
        *   Iterate through `inputs`. For each `input_row`:
            *   Extract `task_data = input_row['task_data']`.
            *   Generate a unique `session_id`.
            *   Call `initial_state = self._initialize_episode(session_id, task_data)`. This uses the task-specific data to set up the environment.
            *   Store `initial_state` in `self.active_states[session_id]`.
            *   Store `task_data` associated with the `session_id` if needed for re-initialization between episodes within a trial.
        *   Initialize `agent_states` for all new `session_id`s.
        *   Loop through `session_id`s:
            *   Retrieve associated `task_data`.
            *   Loop for `self.episodes_per_trial` (from `self.env_config`).
                *   If not the first episode, re-initialize: `current_state = self._initialize_episode(session_id, task_data)`. Store in `self.active_states`.
                *   Call `ep_data = self._run_complete_episode(session_id, agent)`.
                *   Collect episode results. Apply `free_shots` logic (zeroing reward if applicable).
        *   Aggregate results, specifically collecting the per-token trajectory data from the *final* episode of each trial (or handle multi-episode returns if needed later).
    *   **Return Value (Per-Token Information from Final Episode):**
        ```python
        return {
            "full_token_ids": List[List[int]],        # Token IDs for the entire final episode interaction
            "full_attention_mask": List[List[int]],   # Attention mask for the full sequence
            "agent_token_mask": List[List[int]],      # Mask: 1 for agent tokens, 0 otherwise
            "per_token_rewards": List[List[float]],   # Reward signal assigned to each token
            "final_rewards": List[float],             # Final cumulative reward for the episode
            "final_completion_messages": List[List[Dict[str, Any]]], # Optional: For logging/complex rewards
            "session_ids": List[str]                  # Pass session IDs for potential debugging/linking
        }
        ```

2.  **Modify `MultistepEnv._run_complete_episode`:**
    *   **Signature:** `_run_complete_episode(self, session_id, agent: Agent)`. (No change needed here based on this review).
    *   **Logic:** (Largely as before, but ensure it correctly uses `self.tokenizer` and `self.active_states`)
        *   Initialize `state`, `messages`, `current_episode_token_ids`, `current_episode_attention_mask`, `current_agent_token_mask`, `current_per_token_rewards`.
        *   **Initial Prompt:**
            *   Get `prompt_text = _format_prompt(...)`.
            *   Tokenize: `initial_tokens = self.tokenizer(...)` (if `self.tokenizer`).
            *   Append IDs and attention mask.
            *   Append `0`s to `current_agent_token_mask` (length of prompt).
            *   Append `0`s to `current_per_token_rewards` (length of prompt).
            *   Add user message to `messages`.
        *   **Interaction Loop (`while not state["done"]...`):**
            *   Get `action_text, agent_state` from `agent.get_action(...)`.
            *   Tokenize: `response_tokens = self.tokenizer(action_text, ...)` (if `self.tokenizer`).
            *   Append IDs and attention mask.
            *   Append `1`s to `current_agent_token_mask` (length of response).
            *   *Reward Assignment:* Append reward values to `current_per_token_rewards` for the agent's tokens (See Phase 3).
            *   Add assistant message to `messages`.
            *   Call `next_state, reward, done, info = self._step_episode(...)`. `reward` is the step reward.
            *   If not `done`:
                *   Get `next_prompt_text = _format_prompt(...)`.
                *   Tokenize: `next_prompt_tokens = self.tokenizer(...)` (if `self.tokenizer`).
                *   Append IDs and attention mask.
                *   Append `0`s to `current_agent_token_mask`.
                *   Append `0`s to `current_per_token_rewards`.
                *   Add user message to `messages`.
        *   **After Loop:**
            *   Calculate `final_reward` (e.g., from step rewards or terminal state).
            *   *Reward Post-processing (Optional):* Adjust `current_per_token_rewards` based on `final_reward`.
            *   **Return:** Dictionary with `full_token_ids`, `full_attention_mask`, `agent_token_mask`, `per_token_rewards`, `final_reward`, `final_completion_messages`.

3.  **Modify Agent:**
    *   Ensure `agent.get_action` returns `(action_text: str, agent_state: Any)`. No tokenization in the agent.

---

**Phase 3: Reward Calculation Strategy (Inside Environment)**

1.  **Define Reward Calculation within `Environment`:**
    *   The `Environment` calculates per-token rewards.
    *   **Remove `get_rubric`:** Method likely obsolete.
    *   **Implement `_calculate_rewards` (or integrate into loop):** Assign rewards during/after episode generation.
    *   **Strategies (Start Simple):**
        *   **Distribute Step Reward:** Distribute the `reward` from `_step_episode` evenly across tokens of the *preceding* agent action. Assign 0 to env tokens.
        *   **Last Token Reward:** Assign step reward only to the last token of the preceding agent action.
        *   **Terminal Reward Distribution:** Distribute `final_reward` across all agent tokens after the episode ends.
    *   **Initial Approach:** Distribute step reward evenly across the preceding agent action's tokens.

---

**Phase 4: Trainer Adaptation**

1.  **Modify `GRPOEnvTrainer._generate_and_score_completions`:**
    *   **Prepare Inputs:** Adapt the input processing to extract the `task_data` dictionaries from the incoming dataset batch to form the `inputs` list for `env.run_trial`. The trainer no longer needs to format prompts itself.
    *   **`env.run_trial` Call:** Use the new signature: `env_result = self.env.run_trial(inputs=batch_task_data, agent=self.agent)`.
    *   **Data Handling:** Broadcast and slice `env_result` keys (`full_token_ids`, `full_attention_mask`, `agent_token_mask`, `per_token_rewards`).
    *   **Remove `reward_func` Call:** Rewards come from `env_result["per_token_rewards"]`.
    *   **Prepare Tensors:** Pad all relevant sequences (`full_token_ids`, `full_attention_mask`, `agent_token_mask`, `per_token_rewards`). Use `self.tokenizer.pad_token_id` where appropriate.
    *   **LogP Calculation:** Calculate `old_per_token_logps` and `ref_per_token_logps` using padded *full sequences*.
    *   **Advantage Calculation:**
        *   **Initial Approach:** Use `per_token_rewards` directly as advantages. `advantages = padded_per_token_rewards`. Acknowledge this simplification (baseline=0).
        *   *(Future Work):* Implement value head training and GAE within the Trainer if needed.
    *   **Return Dictionary:**
        ```python
        return {
            "prompt_completion_ids": padded_full_token_ids,
            "attention_mask": padded_full_attention_mask,
            "agent_token_mask": padded_agent_token_mask, # Mask for loss targets
            "advantages": advantages,                   # Using per-token rewards as advantages
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
        ```

2.  **Override `GRPOTrainer.compute_loss` (in `GRPOEnvTrainer`):**
    *   Retrieve inputs: `ids`, `attention_mask`, `agent_mask`, `advantages`, `old_logps`, `ref_logps`.
    *   Calculate model's current `per_token_logps`.
    *   **Apply Agent Mask:** Multiply `per_token_logps`, `old_logps`, `ref_logps` element-wise by `agent_mask` *before* summing.
        *   `logprob = (per_token_logps * agent_mask).sum(-1)`
        *   `old_logprob = (old_logps * agent_mask).sum(-1)`
        *   `ref_logprob = (ref_logps * agent_mask).sum(-1)` (if applicable)
    *   Calculate GRPO loss using masked log probabilities and `advantages`.

---

**Phase 5: Testing and Compatibility**

1.  **Test Updates:**
    *   Update `tests/envs/test_multistep_env.py` and `tests/envs/TwentyQuestions/test_env.py` for new `__init__` (pass dummy tokenizer/config) and `run_trial` return structure.
    *   Add tests for tokenization, masking, and reward calculation within the env.
    *   Backward compatibility shims mentioned in the *original* `env_info.md` are **not needed**; update tests to use new keys.

2.  **Example Script Updates:**
    *   Update `examples/twenty_questions.py` to instantiate env with `tokenizer` and `env_config`. Adapt training loop/data loading if necessary.

---

**Questions & Decisions:**

1.  **`env_config` Common Keys:** Define standard keys `MultistepEnv` expects (e.g., `max_steps_per_episode`, `episodes_per_trial`, `free_shots`). Subclasses define their own keys (e.g., `hardcore`, `error_allowance` if not standard).
2.  **Dataset Mixing:** This structure supports mixing data if a runner manages multiple environment instances. The runner would read `env_type` (if present) or use other metadata to route the `task_data` row to the correct `env.run_trial`. The current refactor focuses on the Env/Trainer interaction, assuming the correct env instance is used.
3.  **Episode Seeding:** Deterministic episode initialization should be handled by including a `seed` key within the `task_data` dictionary passed to `run_trial`, rather than a global seed in `MultistepEnv.__init__`.
