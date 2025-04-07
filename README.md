# MetaRLGym

This is a training repo for Meta-Reinforcement Learning with LLMs. 
Meta-Reinforcement Learning (MetaRL) is a paradigm that trains Large Language Models (LLMs) to "learn how to learn." This is achieved by exposing the model to diverse tasks across multiple environments, enabling it to develop _generalizable_ learning strategies. 
Such that the model can quickly adapt and perform effectively with minimal additional training.

## Getting Started

### Prerequisites

*   **UV**: This project uses `uv` for package management. Install it if you haven't already: [UV Installation Guide](https://github.com/astral-sh/uv#installation).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/namkoong-lab/MetaRLGym
    cd MetaRLGym
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # Create and activate a virtual environment
    uv venv
    source .venv/bin/activate

    # Sync dependencies from pyproject.toml
    uv sync

    # Install flash-attn (requires build isolation disabled for specific setups)
    uv pip install flash-attn --no-build-isolation
    ```

### Running the Example

Training requires two separate terminal sessions within the activated virtual environment (`source .venv/bin/activate` in both).

**Terminal 1: Start the VLLM Server**

This terminal runs the language model that the training process will interact with.

```bash
# Start the TRL VLLM server. Replace `"7"` with the GPU you want to use.
CUDA_VISIBLE_DEVICES=7 trl vllm-serve --model "Qwen/Qwen2.5-Math-1.5B"
```
*Note: Replace `"7"` with the ID of the GPU you want to use.*

**Terminal 2: Launch the Training Script**

This terminal runs the main training script using `accelerate` for distributed training.

```bash
# Launch the GSM8k example using accelerate
# Adjust --num-processes based on the number of GPUs you want to use for training (excluding the one for the VLLM server)
accelerate launch --config-file configs/zero3.yaml --num-processes 7 examples/gsm8k_simple.py
```

### Troubleshooting

*   **OpenCV `libGL` error**: If you encounter errors related to `libGL.so.1`, it might mean OpenCV cannot find the necessary OpenGL library. If using Conda, try:
    ```bash
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    ```
    Otherwise, ensure `libgl1-mesa-glx` (or equivalent) is installed via your system's package manager. On the grid, since we don't have permissions, install via
    ```bash
    conda activate uv
    conda install conda-forge::libgl
    ```

## To build an Environment

See the README.md in the `metarlgym/envs` directory for instructions on how to create an environment.

## Background

Paper:
[A Survey of Meta Reinforcement Learning](https://arxiv.org/abs/2301.08028)

Our design constraint involves:
- Use training code from TRL, currently prioritizing GRPO.
- We use API designs from OpenAI's Gym Library (step, reset, etc.), inheriting it from [TextArena](https://github.com/LeonGuertler/TextArena/tree/main).
- Our implementation takes heavy inspiration from [Verifiers](https://github.com/willccbb/verifiers/tree/main). Thank you Willcbcb!

## Glossary

- **Outer Loop (Slow RL)**: LLM learning across multiple environments and tasks
- **Inner Loop (Fast RL)**: Policy deployed in a single task, potentially adapting across episodes
- **Environment/Task Group**: A distribution of tasks (e.g., Wordle, Twenty Questions)
- **Task**: A specific MDP instance (e.g., the word "Apple" in Wordle)
- **Episode/Trajectory**: Complete sequence $(s_0, a_0, r_0, s_1, ..., s_T)$ in a task
- **Experience**: Single interaction $(s_t, a_t, r_t, s_{t+1})$ with environment
- **Step**: Taking action $a$ to transition from $s_t$ to $s_{t+1}$
- **Self-Play/Imagination Rollouts**: LLM simulating steps without environment interaction
- **Lifetime/Trial**: Multiple episodes of the same task with environment interaction
- **Meta-Trajectory**: Collection of episode data $D = \{\tau_1, \tau_2, ..., \tau_H\}$ from a trial
- **Free Shots**: Initial exploration episodes with zero rewards to prevent under-exploration
- **POMDP** (Partially Observable Markov Decision Process): A generalization of MDPs where the agent receives observations instead of full states, requiring inference over hidden state.
- **BAMDP** (Bayes-Adaptive MDP): An MDP where uncertainty over the environment’s dynamics is modeled via a belief, turning the problem into a fully observable MDP over belief states.
