# MultistepEnv - A Framework for Multi-Step LLM Environments

`MultistepEnv` provides a standardized framework for creating multi-step interactive environments for LLM agents. This framework is designed to be compatible with both MetaRLGym's Environment API and TextArena's Env API, enabling a wide range of applications from multi-step reasoning to interactive games.

## Key Features

- **Multi-Step Reasoning**: Enables environments where LLMs can take multiple reasoning steps before reaching a final answer.
- **TextArena Compatibility**: Seamlessly integrates with TextArena environments for game-based interactions.
- **Flexible Action Space**: Supports various types of actions, from structured game moves to free-form reasoning.
- **Standardized API**: Consistent interface for diverse environment types.

## Core Components

The framework consists of the following key components:

1. **MultistepEnv** (Base Class): The foundation for all multi-step environments, handling the core logic for multi-step interactions.

2. **MathEnv**: An environment for multi-step math problem solving with optional hints.

3. **TextArenaMultistepEnv**: A bridge to TextArena environments, allowing for game-based interactions.

## Usage Examples

### Math Environment

```python
from metarlgym.envs import MathEnv
from vllm import LLM, SamplingParams

# Create the environment
env = MathEnv(
    dataset_name="gsm8k",
    max_steps_per_episode=3,
    system_prompt="You are a helpful math assistant. Solve the problem step by step."
)

# Create the LLM
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Evaluate the model
metrics = env.evaluate_llm(
    llm_model=llm,
    num_episodes=5,
    verbose=True
)

print(metrics)
```

### TextArena Environment

```python
from metarlgym.envs import TextArenaMultistepEnv
from vllm import LLM, SamplingParams

# Create the environment
env = TextArenaMultistepEnv(
    env_id="Sudoku-v0",
    max_steps_per_episode=10,
    system_prompt="You are playing Sudoku. Follow the game rules and try to win."
)

# Create the LLM
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Evaluate the model
metrics = env.evaluate_llm(
    llm_model=llm,
    num_episodes=3,
    verbose=True
)

print(metrics)
```

## Creating Custom Environments

You can easily create custom environments by extending the `MultistepEnv` class:

```python
from metarlgym.envs import MultistepEnv

class MyCustomEnv(MultistepEnv):
    def __init__(self, **kwargs):
        super().__init__(
            env_id="MyCustom-v0",
            **kwargs
        )
    
    def _create_task_dataset(self):
        # Create your task dataset here
        pass
    
    def _initialize_episode(self, session_id, task_info):
        # Initialize the episode state
        pass
    
    def _step_episode(self, session_id, state, llm_action):
        # Define how to take a step in the environment
        pass
    
    def _calculate_reward(self, state, llm_actions, final_action):
        # Calculate the final reward
        pass
```

## Running the Examples

We provide an example script to demonstrate the usage of the framework:

```bash
python examples/multistep_example.py --model meta-llama/Llama-2-7b-chat-hf --env both --episodes 2 --verbose
```

## Integration with GRPO Training

The `MultistepEnv` framework is designed to work seamlessly with the GRPO training pipeline. To use it for training:

```python
from trl import GRPOConfig, GRPOTrainer
from metarlgym.envs import MathEnv

# Create the environment
env = MathEnv(
    dataset_name="gsm8k",
    max_steps_per_episode=3
)

# Create the trainer
trainer = GRPOTrainer(
    model="meta-llama/Llama-2-7b-chat-hf",
    rubric=env.get_rubric(),
    dataset=env.get_dataset(),
    environment=env,
    # Other GRPO parameters...
)

# Start training
trainer.train()
```

## Contributing

To contribute to the framework, you can:

1. Create new environment types extending `MultistepEnv`
2. Enhance the existing environments with new features
3. Improve the core framework functionality

## License

This framework is released under the MIT License. 