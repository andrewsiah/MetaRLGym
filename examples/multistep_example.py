#!/usr/bin/env python
"""Example script demonstrating how to use MultistepEnv-based environments.

This script shows how to use both the MathEnv and TextArenaMultistepEnv classes 
for multi-step reasoning tasks.
"""

import logging
import argparse
import sys
from vllm import LLM, SamplingParams

from metarlgym.envs import MathEnv, TextArenaMultistepEnv


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def math_example(model_name, num_episodes=3, verbose=True):
    """Run a simple example with the MathEnv."""
    print("\n=== Math Environment Example ===\n")
    
    # Create the environment
    env = MathEnv(
        dataset_name="gsm8k",
        max_steps_per_episode=3,
        task_dataset_size=100,  # Small dataset for quick demo
        system_prompt="You are a helpful math assistant. Solve the math problem step by step.",
    )
    
    # Create the LLM
    try:
        llm = LLM(model=model_name, max_model_len=4096)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using mock LLM instead")
        # Create a mock LLM for testing
        from vllm.sampling_params import SamplingParams as SP
        class MockLLM:
            def generate(self, prompts, n=1, **kwargs):
                print(f"Mock LLM generating answer for: {prompts[0][:100]}...")
                # Return mock completion IDs (10 tokens)
                return [[i for i in range(10)] for _ in range(len(prompts))]
        llm = MockLLM()
    
    # Evaluate the model on the environment
    metrics = env.evaluate_llm(
        llm_model=llm,
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def textarena_example(model_name, env_id="Sudoku-v0", num_episodes=2, verbose=True):
    """Run a simple example with the TextArenaMultistepEnv."""
    print(f"\n=== TextArena Environment Example ({env_id}) ===\n")
    
    # Create the environment
    env = TextArenaMultistepEnv(
        env_id=env_id,
        task_dataset_size=5,  # Small dataset for quick demo
        max_steps_per_episode=10,
        system_prompt=f"You are playing {env_id}. Follow the game rules and try to win.",
    )
    
    # Create the LLM
    try:
        llm = LLM(model=model_name, max_model_len=4096)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using mock LLM instead")
        # Create a mock LLM for testing
        from vllm.sampling_params import SamplingParams as SP
        class MockLLM:
            def generate(self, prompts, n=1, **kwargs):
                print(f"Mock LLM generating answer for: {prompts[0][:100]}...")
                # Return mock completion IDs (10 tokens)
                return [[i for i in range(10)] for _ in range(len(prompts))]
        llm = MockLLM()
    
    # Evaluate the model on the environment
    metrics = env.evaluate_llm(
        llm_model=llm,
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MultistepEnv examples")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", 
                        help="Model name to use for evaluation")
    parser.add_argument("--env", type=str, choices=["math", "textarena", "both"], default="both",
                        help="Which environment to demonstrate")
    parser.add_argument("--textarena-env", type=str, default="Sudoku-v0",
                        help="TextArena environment ID")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of evaluation episodes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Run the selected examples
    if args.env in ["math", "both"]:
        math_example(args.model, args.episodes, args.verbose)
    
    if args.env in ["textarena", "both"]:
        textarena_example(args.model, args.textarena_env, args.episodes, args.verbose)
    

if __name__ == "__main__":
    main() 