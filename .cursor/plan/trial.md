<!-- Trial design plan for TwentyQuestions and general trial implementation -->
# Trial Plan

A 1. Trial Definition
A **trial** `𝒯` consists of:
- **Context** `ℐ`: initial info (e.g., game rules, number of episodes)
- **Task** `𝓜`: the specific hidden word for TwentyQuestions
- **Episodes**: `K` sequential games on the same word

For TwentyQuestions:
For TwentyQuestions:
- Default `episodes_per_trial = 1` (one game per trial)
- Default `free_shots = 0` (no free-shots/unscored episodes by default)
- Allow optional overrides: `episodes_per_trial: int >= 1`, `free_shots: int >= 0`, with constraint
  ```
  free_shots < episodes_per_trial
  ```
  so that at least one episode is scored per trial.
  ```
  exploration_episodes < episodes_per_trial
  ```
  to ensure final episode is scored.

## 2. Dataset Creation
- **Training set** size = user‐defined `task_dataset_size`
- Build `task_dataset` by sampling (`theme`, `word`) pairs:
  - Each row:
    - `prompt`: initial player prompt via `_generate_player_prompt(theme)`
    - `solution`: the chosen word
- **Evaluation set**: disjoint held‐out subset of words (random split)

## 3. Environment Hooks
1. `_create_task_dataset(self)`:
   - Sample words, split into train vs. eval
   - Populate `self.task_dataset` / `self.eval_task_dataset`
2. `_initialize_episode(self, session_id, task_info)`:
   - Set `self.game_theme`, `self.game_word = task_info["solution"]`
   - Seed & reset `self.state` with `seed`
   - Return state dict for `_format_prompt`
3. `_format_prompt(self, state, step)`: render history + context
4. `_step_episode(self, session_id, state, llm_action)`:
   - Call existing `step(action)` → `(next_state, reward, done, info)`
   - Return for RL loop
5. `_calculate_reward(self, state, llm_actions, final_action)`:
   - Return numeric reward (e.g., +1 for correct guess, else 0)

## 4. generate() Changes
- Loop `ep_idx` over `0 .. episodes_per_trial-1` per trial
  - For `ep_idx < free_shots`, run episodes but zero-out or ignore their rewards
- Ensure `exploration_episodes < episodes_per_trial` at call/initialization time
- Default `exploration_episodes = 0`, `episodes_per_trial = 1`
- Final episode’s LLM response is used as the trainer “completion”

## 5. Reward & Evaluation
- No question‐count penalties (use default TextArena rewards)
- Use disjoint eval set for held‐out words

## 6. Testing
- Unit tests for:
  - `get_dataset()` outputs correct prompts & solutions
  - End‐to‐end `generate()` for single & multi‐episode trials
  - Reward lookup via `get_rubric()` etc.

Implementing this plan will provide a clean “trial” abstraction (one seed per trial, multiple episodes, unique tasks, and per-trial datasets of episode-lists).
