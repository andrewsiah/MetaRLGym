## Plan: Enable Per-Step Masking and Training on Intermediate Agent Outputs

Objective: Allow GRPOEnvTrainer and MultistepEnv to train on every agent response in a multi-turn episode (rather than only the final action), by splitting prompt vs. completion masks per step and flattening multi-step outputs into individual RL transitions.

### 1. Extend MultistepEnv.generate() to return all steps
  - Add a new flag/parameter `return_all_steps: bool` to `MultistepEnv.generate()` (default: `False` for backward compatibility).
  - Inside `generate()`, for each session:
    - During `_run_complete_episode()`, accumulate per-step:  
      • `observations[i]` (prompt text),  
      • `llm_responses[i]` (raw text),  
      • `response_ids[i]` (token IDs),  
      • `messages[i]` (full message history up to and including the response).
    - After running all steps, if `return_all_steps` is `True`, structure the return dict so that for N prompts and up to K steps per prompt, we produce:
      ```yaml
      ids: [[ids_{p,1}, ids_{p,2}, ..., ids_{p,S_p}] for p in 1..N]
      messages: [[messages_{p,1}, ..., messages_{p,S_p}] for p in 1..N]
      mask: [[ [1]*len(ids_{p,k}) for k in 1..S_p ] for p in 1..N]
      session_ids: [[session_id]*S_p for p in 1..N]
      ```
    - If `return_all_steps` is `False`, keep existing behavior (return only final step as single-element lists).

### 2. Adapt GRPOEnvTrainer._generate_and_score_completions()
  - Detect nested completion lists: if `completion_ids` is a list of lists (instead of flat list):
    1. **Flatten**:
       - Build new lists `flat_prompts`, `flat_completion_ids`, `flat_masks`, `flat_session_ids`, and, if provided, `flat_rewards`, plus map back indices `episode_idx_map` so we know which episode each flat sample came from.
       - For each prompt index `p` in batch and each step index `k`:
         ```python
         flat_prompts.append(all_prompts[p])
         flat_completion_ids.append(completion_ids[p][k])
         flat_masks.append(mask[p][k])
         flat_session_ids.append(session_ids[p])
         # If per-step rewards were returned by env.generate():
         flat_rewards.append(rewards[p][k])
         episode_idx_map.append(p)
         ```
    2. **Broadcast & Pad** as usual over the flat lists.
    3. **Compute per-token log‑probs and advantages** on each flat sample; train on every step.
    4. **Group metrics**: to compute episode-level statistics (e.g., final reward), aggregate flat rewards back to episodes using `episode_idx_map`.
  - If not nested, preserve original code path.

### 3. Modify Reward Handling
  - Decide reward assignment strategy for episodes with multiple steps or episodes:
    1. **Dense per-step rewards**:
       - Extend `MultistepEnv.generate(return_all_steps=True)` to also return a nested `rewards: List[List[float]]`,
         providing a scalar reward for each step (or token if finer granularity is desired).
       - Environment implementations should compute and emit these per-step rewards from `_step_episode()` or `_calculate_reward()`.
    2. **Per-token credit assignment**:
       - In the trainer, align each `completion_ids[p][k]` (the k-th step token IDs for prompt p) with `rewards[p][k]`.
       - Flatten both lists in the same order and repeat or interpolate the step reward across the tokens of that step.
       - Use these per-token or per-step rewards when computing policy gradients (i.e., multiply log‑probs by the per-step reward instead of a final episodic return).
    3. **Final vs intermediate rewards**:
       - For settings where only a final reward is available, repeat the final reward across all steps.
       - For pure episodic RL tasks, default to using the final episode reward on the last step only (zero reward on earlier steps).
  - **Integrate into GRPOEnvTrainer**:
    - Modify `_generate_and_score_completions()` to accept a third flattened list `flat_rewards`.
    - When computing advantages, use `flat_rewards` instead of the current aggregated-per-episode reward.
  - Update `get_rubric()` (or a new calibration function) to retrieve or compute these per-step rewards for each flat sample.

### 4. Refactor MultistepEnv._run_complete_episode()
  - Remove concatenated generation logic; instead rely on agent calls already instrumented per step.
  - Ensure `episode_data` collects full per-step lists.
  - Pass `return_all_steps` through to `generate()` and onto `_run_complete_episode()` if needed.

### 5. Tests and Validation
  - Write unit tests for `MultistepEnv.generate(return_all_steps=True)`:
    - Mock a two-step env to return predictable `ids` and `messages`; assert nested structure.
  - Write integration tests in GRPOEnvTrainer:
    - Simulate flattening on a small batch of multi-step episodes; assert that the final input passed to the model for training contains the correct number of flat prompt+completion pairs and that padding/masking aligns.
  - Ensure backward compatibility: existing single-step envs and `return_all_steps=False` should behave exactly as before.

### 6. Documentation
  - Update `MultistepEnv.generate()` docstring to describe `return_all_steps` behavior and output format.
  - Add examples in README explaining how to enable multi-turn training.
  - Document new Trainer behavior in GRPOEnvTrainer docstrings.

### 7. Milestones
  1. Skeleton patch: `return_all_steps` parameter and pass-through in `generate()`
  2. Collect per-step data in `episode_data` and implement nested return format
  3. Flattening logic in `GRPOEnvTrainer._generate_and_score_completions`
  4. Reward propagation strategy implementation
  5. Comprehensive tests and CI
  6. Documentation updates and examples
  7. Code review and merge

By following this plan, we enable fine‑grained policy learning on each reasoning step while preserving existing single‑turn behavior by default.