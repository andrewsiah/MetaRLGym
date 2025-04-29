# AgentsGym Core Package (`agentsgym`)

## Directory Structure

-   **`envs/`**: Defines the various reinforcement learning environments available in AgentsGym. This includes base environment classes and specific implementations like `math_env`, `tool_env`, `textarena_env`, etc. 
-   **`curriculum/`**: How to do active learning by structuring the tasks & environments that our policy learns over.
-   **`eval/`**: Contains scripts and utilities for evaluating trained agents. (Currently seems empty).
-   **`inner_loop/`**: For the inner loop of Agents, i.e. special prompting methods or other adaptive methods.
-   **`parsers/`**: Includes modules for parsing specific data formats, such as the `xml_parser.py`.
-   **`prompts/`**: Stores prompt templates, few-shot examples (`few_shots.py`), and system prompts (`system_prompts.py`) used for interacting with language models within the environments or agents.
-   **`rubrics/`**: Defines rubrics used for evaluating agent performance in different tasks (e.g., `math_rubric.py`, `code_rubric.py`, `tool_rubric.py`). Contains the base `rubric.py`.
-   **`tools/`**: Implements tools that agents can use within environments, such as a `calculator.py` and `search.py`.
-   **`trainers/`**: Contains training algorithms and scripts for training agents, like `grpo_env_trainer.py`.
-   **`utils/`**: Provides various utility functions supporting the framework, including configuration (`config_utils.py`), data handling (`data_utils.py`), logging (`logging_utils.py`), and model interactions (`model_utils.py`).
-   **`temp_textarena/`**: Temporary here so that my AI coding agents can easily grab context.
