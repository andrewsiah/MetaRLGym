# MetaRLGym Environments

This directory contains various interactive environments designed for training and evaluating Large Language Models (LLMs) within the MetaRLGym framework.

## Core Concept: The `Environment` Base Class

All environments in MetaRLGym inherit from the abstract base class `Environment` defined in `environment.py`. This class provides a standardized interface for interacting with different types of tasks and scenarios.

To create a new environment, you need to subclass `Environment` and implement the following abstract methods:

1.  **`__init__(self, **kwargs)`**:
    *   Initialize your environment-specific attributes.
    *   It's recommended to call `super().__init__(**kwargs)` to handle common initialization.
    *   Set up logging using `self.logger`.

2.  **`get_dataset(self, **kwargs) -> Dataset | None`**:
    *   Load or generate the training dataset for your environment.
    *   This dataset should typically contain the prompts or initial states for the tasks.
    *   Return a `datasets.Dataset` object or `None` if no specific training dataset is needed directly from the environment (e.g., if handled by a trainer).

3.  **`get_eval_dataset(self, **kwargs) -> Dataset | None`**:
    *   Load or generate the evaluation dataset.
    *   Similar structure to `get_dataset`, but for evaluation purposes.
    *   Return a `datasets.Dataset` object or `None`.

4.  **`get_rubric(self, **kwargs) -> List[RewardFunc]`**:
    *   Define the reward functions (rubrics) used to evaluate the LLM's performance in the environment.
    *   This is crucial for reinforcement learning algorithms like GRPO.
    *   Return a list of `trl.trainer.grpo_trainer.RewardFunc` objects.

5.  **`generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams, **kwargs) -> Dict[...]`**:
    *   This method handles the interaction between the LLM and the environment.
    *   It takes a batch of prompts (often conversation histories), an LLM instance, and sampling parameters.
    *   It should generate the LLM's responses/actions based on the prompts.
    *   The exact return type depends on the specific needs of the environment and how it integrates with trainers, but it typically includes the generated sequences or actions.

## Creating a Custom Environment: Template

Here's a basic template for creating your own environment:

```python
from metarlgym.envs import Environment
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Sequence

class MyCustomEnv(Environment):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Your custom initialization logic here
        self.logger.info("Initializing MyCustomEnv")
        # Example: Load resources, define parameters, etc.

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        # Load or create your training dataset
        self.logger.info("Loading/Creating training dataset...")
        # Replace with your actual dataset loading logic
        # Example: return Dataset.from_dict({"prompt": ["Task 1", "Task 2"]})
        return None

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        # Load or create your evaluation dataset
        self.logger.info("Loading/Creating evaluation dataset...")
        # Replace with your actual dataset loading logic
        return None

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        # Define your reward functions
        self.logger.info("Defining rubric...")
        # Replace with your actual RewardFunc definitions
        rubrics: List[RewardFunc] = []
        # Example: rubrics.append(RewardFunc(...))
        return rubrics

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        # Implement the LLM interaction logic
        self.logger.info(f"Generating responses for {len(prompts)} prompts...")

        # Use the llm object to generate responses based on the prompts
        # Example: llm_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

        # Process llm_outputs and return them in the required format
        # The exact format might depend on the trainer or evaluation loop using this env
        results = {
            "generated_texts": [], # Placeholder for actual results
            # Add other relevant outputs
        }
        return results

```

## Existing Environments

This directory includes several pre-built environments:

*   **`MultistepEnv`**: A base class for environments involving multiple interaction steps (see `multistep_env.py` and its specific README).
*   **`MathEnv`**: An environment for solving math problems (GSM8k).
*   **`TextArenaMultistepEnv`**: Integrates environments from the TextArena library.
*   **`ToolEnv`**: An environment focused on LLM tool usage.
*   **`SimpleEnv`**: A basic example environment.

Explore these implementations for more detailed examples of how to build environments within the MetaRLGym framework.
