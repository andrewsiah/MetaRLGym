---

**PRD 1: Environment Refactoring for Rollouts and Internal Rewards**

**File:** `env_update.md`

**1. Goal/Objective:**
Refactor the `MultistepEnv` base class and its subclasses to:
*   Be initializable based on a configuration dictionary (`env_config`).
*   Accept task-specific data (`task_data`) to initialize individual episodes/trials.
*   Perform multiple full rollouts (`num_rollouts`) for a given task configuration via a revamped `run_trial` method.
*   Calculate per-token rewards internally during episode execution (`_run_complete_episode`).
*   Return detailed, padded, per-token trajectory data from `run_trial`, ready for trainer consumption.
*   Standardize the dataset schema required for instantiating environments and tasks.

**2. Background/Motivation:**
The current environment implementation is coupled with simple prompt inputs and relies on external reward functions via `get_rubric`. The trainer handles generation and reward calculation. To enable true multi-step RL training (like GRPO on full trajectories) and support diverse environments, the environment needs to manage its own rollouts using an agent, calculate rewards internally based on its state transitions, and provide structured trajectory data (including token-level rewards and masks) to the trainer. This decoupling simplifies the trainer's role.

**3. Scope:**
*   **In Scope:**
    *   Modifying `MultistepEnv` (`__init__`, `run_trial`, `_run_complete_episode`).
    *   Modifying specific environment subclasses (e.g., `TwentyQuestionsEnv`) for `__init__`, `_initialize_episode`, `_step_episode`, `_format_prompt`, reward calculation logic, and `get_train_dataset`.
    *   Defining the new standard dataset schema.
    *   Implementing per-token reward calculation logic within environments.
    *   Implementing padding logic within `run_trial`.
    *   Removing `get_rubric`.
    *   Updating unit tests for environments.
*   **Out of Scope:**
    *   Any modifications to `GRPOEnvTrainer` or other trainer classes.
    *   Modifications to the `Agent` class interface (assuming `agent.get_action` returning `(action_text, agent_state)` is sufficient).
    *   Implementing complex advantage estimation (e.g., GAE) within the environment.
    *   End-to-end training loop tests.

**4. Requirements:**

*   **R1: Dataset Schema Definition:**
    *   Define the standard dataset row schema required by the refactored system:
      ```python
      {
        "env_class_path": str,        # Full Python path to Env class
        "env_config": Dict[str, Any], # Config dict for Env.__init__
        "task_data": Dict[str, Any],  # Config dict for Env._initialize_episode
      }
      ```
*   **R2: Environment `__init__` Refactoring:**
    *   Modify `MultistepEnv.__init__` signature to `(self, tokenizer: PreTrainedTokenizerBase, env_config: Dict[str, Any])`.
    *   Store `self.tokenizer` and `self.env_config`.
    *   Remove parameters: `prompt`, `train_dataset_size`, `env_id`, global `seed`, `system_prompt`.
    *   Update all subclasses' `__init__` methods to match this signature, call `super().__init__`, and parse necessary configuration from `env_config`.
*   **R3: Dataset Generation Update:**
    *   Modify `get_train_dataset` (and similar methods like `get_eval_dataset`) in all subclasses to generate and return a `datasets.Dataset` where each row conforms to the schema defined in R1.
*   **R4: `_run_complete_episode` Enhancement:**
    *   Modify `MultistepEnv._run_complete_episode(self, session_id, agent)`:
        *   Track token origins (agent vs. environment/prompt).
        *   Implement per-token reward calculation logic (e.g., distribute step reward to preceding agent tokens).
        *   Return a dictionary containing *unpadded* lists for the single completed episode: `{"full_token_ids": List[int], "full_attention_mask": List[int], "agent_token_mask": List[int], "per_token_rewards": List[float], "final_reward": float}`.
*   **R5: `run_trial` Overhaul:**
    *   Modify `MultistepEnv.run_trial` signature to `(self, task_data_list: List[Dict[str, Any]], agent: Agent, num_rollouts: int, **kwargs)`.
    *   Iterate through each `task_data` in `task_data_list`.
    *   For each `task_data`, loop `num_rollouts` times, calling `_run_complete_episode` for each rollout.
    *   Collect the unpadded results from all rollouts for the *current task*.
    *   Implement padding logic: Pad all collected sequences (`full_token_ids`, `full_attention_mask`, `agent_token_mask`, `per_token_rewards`) for the current task's rollouts to the maximum sequence length observed *within that task's rollouts*. Use `self.tokenizer.pad_token_id` or 0 as appropriate.
    *   Aggregate the padded results (from all tasks in `task_data_list`) into overall lists/tensors.
    *   Return a dictionary containing the aggregated, padded trajectory data for the entire input batch: `{"padded_full_token_ids": List[List[int]], "padded_full_attention_mask": ..., "padded_agent_token_mask": ..., "padded_per_token_rewards": ..., "final_rewards": List[float]}`.
*   **R6: Remove `get_rubric`:** Delete the `get_rubric` method from `MultistepEnv` and any overrides in subclasses.

**5. Testing Plan:**

*   **T1: Unit Test - Dataset Generation:**
    *   Test `TwentyQuestionsEnv.get_train_dataset` to ensure it returns a `Dataset` object where rows strictly adhere to the schema in R1 (`env_class_path`, `env_config`, `task_data`). Verify the content of these fields is correct for 20 Questions.
    *   *(Apply similarly to other env subclasses as they are refactored).*
*   **T2: Unit Test - `__init__`:**
    *   Test `MultistepEnv.__init__` stores `tokenizer` and `env_config`.
    *   Test `TwentyQuestionsEnv.__init__` correctly calls super, stores `tokenizer`, `env_config`, and parses specific config values (e.g., `hardcore` if added to `env_config`).
    *   *(Apply similarly to other env subclasses).*
*   **T3: Unit Test - `_run_complete_episode`:**
    *   Use `TwentyQuestionsEnv` instance.
    *   Use a mock `agent` (returning predefined questions/guesses) and a real `tokenizer`.
    *   Provide sample `task_data` (e.g., `{"solution": "testword"}`).
    *   Call `env._run_complete_episode(...)` for the sample task.
    *   Verify the returned dictionary contains the correct keys (`full_token_ids`, etc.).
    *   Verify the lengths of all returned lists are consistent for the single trajectory.
    *   Verify `agent_token_mask` correctly identifies mock agent tokens based on the mock agent's responses.
    *   Verify `per_token_rewards` assigns non-zero values only to agent tokens based on the reward logic implemented in `TwentyQuestionsEnv._step_episode` (or wrapper logic).
    *   Verify the return values are *unpadded*.
*   **T4: Unit Test - `run_trial`:**
    *   Use `TwentyQuestionsEnv` instance.
    *   Use a mock `agent` and a real `tokenizer`.
    *   Can mock `_run_complete_episode` (returning data matching T3's verified output structure) *or* use the real one tested in T3.
    *   Call `env.run_trial` with a `task_data_list` (e.g., `[{"solution": "apple"}, {"solution": "banana"}]`, size >= 1) and `num_rollouts` > 1.
    *   Verify `_run_complete_episode` is called `len(task_data_list) * num_rollouts` times.
    *   Verify the returned dictionary contains the correct keys (`padded_full_token_ids`, etc.).
    *   Verify all sequences within `padded_full_token_ids` (and other sequence keys) for the *same input task* (e.g., all rollouts for "apple") have the *same length* after padding. Lengths might differ between different tasks (e.g., "apple" rollouts vs. "banana" rollouts).
    *   Verify the total number of trajectories returned (e.g., number of elements in `padded_full_token_ids`) is `len(task_data_list) * num_rollouts`.
    *   Check padding values match `tokenizer.pad_token_id` or 0 as appropriate.
*   **T5: Integration Test - Single Env Run:**
    *   Instantiate a *real*, refactored `TwentyQuestionsEnv` with a real `tokenizer` and a dummy `agent` (e.g., one that repeats a simple question/guess).
    *   Create a small sample `task_data_list` for 20 Questions (e.g., `[{"solution": "clock"}, {"solution": "chair"}]`).
    *   Call `env.run_trial(task_data_list=..., agent=dummy_agent, num_rollouts=2)`.
    *   Check for runtime errors during initialization and execution.
    *   Perform basic sanity checks on the structure and dimensions of the returned padded data dictionary (e.g., number of elements, consistent padded lengths per task group).

**6. Success Metrics:**
*   All environment subclasses are successfully refactored according to the requirements.
*   All unit tests (T1-T4) pass for `MultistepEnv` and at least one representative subclass.
*   The integration test (T5) runs without errors and produces data in the expected padded format.
*   The `get_rubric` method is removed.

---