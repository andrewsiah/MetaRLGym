### Testing & Compatibility Considerations

To ensure existing tests and downstream code continue to pass, we must address the following:

1. Backward-compatible return keys
   - Existing tests use `env.run_trial` and expect top-level keys: `ids`, `messages`, `mask`, `session_ids`.
   - We should alias these in `MultistepEnv.run_trial` to map:
     - `ids` → `full_token_ids`
     - `messages` → final agent messages (`final_completion_messages`)
     - `mask` → `completion_only_mask`
     - `session_ids` remains unchanged.
   - Alternatively, update all tests to use the new key names.

2. Constructor signature defaults
   - `tokenizer` should default to `None` so tests instantiating environments without a tokenizer still work.
   - Ensure tokenization branches skip if `self.tokenizer is None`.
   - Expose `episodes_per_trial: int = 1` and `free_shots: int = 0` in all env subclass constructors (e.g., `TwentyQuestionsEnv.__init__`).

3. Test impacts
   - `tests/envs/test_multistep_env.py` and `tests/envs/TwentyQuestions/*` rely on the original `generate` return schema and constructor defaults.
   - The smoke test `test_generate_structure` is currently xfailed; once compatibility is provided, we can un-xfail it or update it to the new schema.

4. Action items
   - Implement return-key aliases in `MultistepEnv.run_trial` to maintain `ids`, `messages`, `mask`, `session_ids`.
   - Update environment constructors to accept (and default) the new parameters (`tokenizer`, `episodes_per_trial`, `free_shots`).
   - Add guard clauses if `self.tokenizer is None` to bypass tokenization.
   - Review and update example scripts (e.g., `examples/twenty_questions.py`) to pass a tokenizer where needed.
# Refactoring Plan: Environment Data Handling & Tokenization

**Core Idea:** The `Environment` will manage the entire interaction flow, including tokenization of both environment prompts and agent responses. It will return the complete token sequences and necessary masks to the `Trainer`, which will then focus on computing log probabilities and the loss. This centralizes responsibility and simplifies interfaces.

---

**Revised Plan:**

**Phase 1: Environment Setup & Tokenization**

1.  **Modify `MultistepEnv` and Subclass Initializers:**
    *   Update `MultistepEnv.__init__` to accept `tokenizer: PreTrainedTokenizerBase` as an argument. Store it as `self.tokenizer`.
    *   Update `TwentyQuestionsEnv.__init__` (and any other subclasses) to accept `tokenizer` and pass it to the `super().__init__` call.
    *   In the main script (`examples/twenty_questions.py`), pass the loaded `tokenizer` when creating the `TwentyQuestionsEnv` instance.

2.  **Simplify Dataset Schema:**
    *   The `datasets.Dataset` now only needs columns required for environment initialization by `_initialize_episode`.
    *   **Example (`TwentyQuestionsEnv`):** `Dataset({"solution": datasets.Value("string")})`.
    *   Update `TwentyQuestionsEnv.get_train_dataset` to generate this simpler structure: `{"solution": train_words_list}` and `{"solution": eval_words_list}`. Remove any `agent_prompts` column.

**Phase 2: Environment Interaction Logic (`run_trial`, `_run_complete_episode`)**

3.  **Modify `MultistepEnv.run_trial`:**
    *   **Input Signature:** Change to accept the raw batch from the dataloader: `run_trial(self, inputs: List[Dict[str, Any]], llm: LLM, sampling_params: SamplingParams, agent: Agent, **kwargs)`.
    *   **Logic:**
        *   Iterate through the `inputs` list (each element is data for one trial).
        *   For each `input_data`:
            *   Extract `task_info` needed for initialization (e.g., `task_info = {"solution": input_data["solution"]}`).
            *   Generate `session_id`.
            *   Call `initial_state = self._initialize_episode(session_id, task_info)`.
            *   Handle `episodes_per_trial` loop:
                *   Call `ep_data = self._run_complete_episode(session_id, llm, sampling_params, agent)`. This function returns detailed token/mask info.
                *   Re-initialize state for subsequent episodes if needed.
            *   Keep track of the `final_ep = episodes[-1]` data. Store its results (`full_token_ids`, `full_attention_mask`, `completion_only_mask`, `final_completion_messages`, `session_id`).
    *   **Return Value:** Structure the output dictionary to include the detailed token/mask information from the final episodes:
        ```python
        return {
            "full_token_ids": List[List[int]],           # Token IDs for the entire final episode interaction
            "full_attention_mask": List[List[int]],      # Attention mask for the full sequence
            "completion_only_mask": List[List[int]],     # Mask identifying only the final agent completion tokens to train on
            "final_completion_messages": List[List[Dict[str, Any]]], # Structured messages for reward func
            "session_ids": List[str]
        }
        ```

4.  **Modify `MultistepEnv._run_complete_episode`:**
    *   **Input Signature:** `_run_complete_episode(self, session_id, llm, sampling_params, agent: Agent)`.
    *   **Logic:**
        *   Get `state = self.active_states[session_id]`.
        *   Initialize `messages = []` (for agent input).
        *   Initialize `current_episode_token_ids = []` and `current_episode_attention_mask = []`.
        *   **Generate & Tokenize Initial Prompt:** Use `_format_prompt(state, 0)` to get the initial `prompt_text`. Tokenize it using `self.tokenizer`: `initial_tokens = self.tokenizer(prompt_text, add_special_tokens=False)`. Append `initial_tokens["input_ids"]` to `current_episode_token_ids` and `initial_tokens["attention_mask"]` to `current_episode_attention_mask`. Add the initial user message `{"role": "user", "content": prompt_text}` to `messages`.
        *   **Interaction Loop (`while not state["done"]...`):**
            *   Call `action_text, agent_state = agent.get_action(messages, agent_state)`. (Agent returns only text).
            *   **Tokenize Agent Response:** `response_tokens = self.tokenizer(action_text, add_special_tokens=False)`.
            *   Append `response_tokens["input_ids"]` to `current_episode_token_ids`.
            *   Append `response_tokens["attention_mask"]` (or `[1]*len(...)`) to `current_episode_attention_mask`.
            *   Store `action_text` (e.g., in `episode_data["llm_responses"]`) for logging/reward.
            *   Add assistant message to `messages`: `messages.append({"role": "assistant", "content": action_text})`.
            *   Call `next_state, reward, done, info = self._step_episode(...)`. Update `state`.
            *   If not `done`:
                *   Get `next_prompt_text = _format_prompt(state, current_step)`.
                *   Tokenize: `next_prompt_tokens = self.tokenizer(next_prompt_text, add_special_tokens=False)`.
                *   Append `next_prompt_tokens["input_ids"]` to `current_episode_token_ids` and `next_prompt_tokens["attention_mask"]` to `current_episode_attention_mask`.
                *   Add user message to `messages`: `messages.append({"role": "user", "content": next_prompt_text})`.
        *   **After Loop:**
            *   Calculate `final_reward`.
            *   **Create `completion_only_mask`:** Initialize a list of `0`s with the same length as `current_episode_token_ids`. Identify the token indices corresponding to *all* agent responses (all `response_tokens["input_ids"]` appended during the loop). Set the corresponding elements in the mask to `1`. *Initially, this might just mask the final agent response, but masking all agent responses could be an alternative training strategy.*
            *   **Return:** Dictionary containing `full_token_ids=current_episode_token_ids`, `full_attention_mask=current_episode_attention_mask`, `completion_only_mask=completion_only_mask`, `final_completion_messages=[messages[-1]]` (or relevant messages for reward), `reward=final_reward`, etc.

5.  **Modify `DirectOutputAgent` (or Agent Wrapper):**
    *   The Agent does **not** need the tokenizer.
    *   `get_action` should generate the response text using its internal LLM or logic.
    *   `get_action` should return `(action_text: str, agent_state: Any)`.

**Phase 3: Trainer Adaptation**

6.  **Modify `GRPOEnvTrainer._generate_and_score_completions`:**
    *   **Input Processing:** `inputs` is the raw batch dictionary (e.g., `{"solution": [...], ...}`).
    *   **Gathering:** `all_inputs = gather_object(inputs)`.
    *   **`env.run_trial` Call (Main Process):** `env_result = self.env.run_trial(inputs=all_inputs, llm=self.vllm_client, sampling_params=self.sampling_params, agent=self.agent)`.
    *   **Broadcasting & Slicing:** Broadcast `env_result` contents (`full_token_ids`, etc.). Slice them per process (e.g., `full_token_ids_local`, `completion_only_mask_local`, etc.).
    *   **Prepare Tensors for LogPs:**
        *   Convert `full_token_ids_local` and `full_attention_mask_local` to tensors.
        *   Pad them: `prompt_completion_ids = pad(...)`, `attention_mask = pad(...)`. Use `self.tokenizer.pad_token_id` (accessing via `self.processing_class` might also work if it's the same tokenizer).
    *   **LogP Calculation:** Calculate `old_per_token_logps` and `ref_per_token_logps` using the *padded full sequences* (`prompt_completion_ids`, `attention_mask`). These logps cover the whole interaction.
    *   **Reward Calculation:** Call `reward_func(completions=final_completion_messages_local, session_ids=session_ids_local, **reward_kwargs)`. Ensure `MultistepEnv.get_rubric` is adapted to accept `session_ids`.
    *   **Advantage Calculation:** Compute advantages based on rewards as before.
    *   **Return Dictionary:** Include the new mask:
        ```python
        return {
            "prompt_completion_ids": prompt_completion_ids, # Padded full sequence IDs
            "attention_mask": attention_mask,             # Padded full sequence mask
            "completion_only_mask": pad(completion_only_mask_local, padding_value=0), # Padded mask for loss targets
            "old_per_token_logps": old_per_token_logps,   # Logps for full sequence
            "ref_per_token_logps": ref_per_token_logps,   # Logps for full sequence
            "advantages": advantages,
        }
        ```
        *Note:* Ensure `completion_only_mask` is also converted to a tensor and padded consistently.

7.  **Override `GRPOTrainer.compute_loss`:**
    *   Create a method `compute_loss` within `GRPOEnvTrainer`.
    *   Inside this method:
        *   Get the model's logits for the `prompt_completion_ids`.
        *   Calculate `per_token_logps` from logits and labels (`prompt_completion_ids`).
        *   Retrieve `completion_mask = inputs["completion_only_mask"]`.
        *   Retrieve `old_logps = inputs["old_per_token_logps"]`, `ref_logps = inputs["ref_per_token_logps"]`.
        *   **Apply Mask:** Before summing or using log probabilities in the loss formula, multiply them element-wise by the `completion_mask`.
            *   `logprob = (per_token_logps * completion_mask).sum(-1)`
            *   `old_logprob = (old_logps * completion_mask).sum(-1)`
            *   `ref_logprob = (ref_logps * completion_mask).sum(-1)` (if `ref_logps` exists)
        *   Retrieve `advantages = inputs["advantages"]`.
        *   Calculate the standard GRPO loss terms (policy loss, reference loss/KL penalty) using these *masked* log probabilities and the advantages. Handle potential division by zero if the mask sums to zero.
    *   Return the computed loss.

---

This revised plan centralizes the complex interaction and tokenization logic within the environment, making the trainer's role clearer (orchestration, logp calculation, loss computation) and arguably leading to a cleaner design for multi-step RL tasks within this framework.
