#!/usr/bin/env python
"""Example script demonstrating how to use MultistepEnv-based environments.

This script shows how to use both the MathEnv and TextArenaMultistepEnv classes 
for multi-step reasoning tasks.
"""

import logging
import argparse
import sys
import os
from vllm import LLM, SamplingParams

from metarlgym.envs import MathEnv, TextArenaMultistepEnv
from metarlgym.utils.logging_utils import setup_logging, ensure_logs_directory


def setup_example_logging(example_name, level="INFO"):
    """Set up comprehensive logging configuration for examples.
    
    Args:
        example_name: Name of the example for log file naming
        level: Logging level to use
        
    Returns:
        Logger instance
    """
    # Create a timestamp-based run name
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{example_name}_{timestamp}"
    
    # Setup logging with proper directory structure
    logging_result = setup_logging(
        level=level,
        log_to_file=True,
        run_name=run_name
    )
    
    logger = logging_result["logger"]
    log_file_path = logging_result["log_file_path"]
    
    logger.info(f"Starting {example_name} example")
    logger.info(f"Logging to: {log_file_path}")
    
    return logger


def math_example(model_name, num_episodes=3, verbose=True):
    """Run a simple example with the MathEnv."""
    logger = setup_example_logging("math_env", level="DEBUG" if verbose else "INFO")
    logger.info(f"Starting math environment example with model: {model_name}")
    
    print("\n=== Math Environment Example ===\n")
    
    # Create the environment
    logger.info("Creating MathEnv with gsm8k dataset")
    env = MathEnv(
        dataset_name="gsm8k",
        max_steps_per_episode=3,
        task_dataset_size=100,  # Small dataset for quick demo
        system_prompt="You are a helpful math assistant. Solve the math problem step by step.",
    )
    
    # Create the LLM
    logger.info(f"Initializing LLM with model: {model_name}")
    try:
        llm = LLM(model=model_name, max_model_len=4096)
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to mock LLM")
        print(f"Error loading model: {e}")
        print("Using mock LLM instead")
        # Create a mock LLM for testing
        from vllm.sampling_params import SamplingParams as SP
        class MockLLM:
            def generate(self, prompts, n=1, **kwargs):
                logger.debug(f"Mock LLM generating for prompt: {prompts[0][:100]}...")
                print(f"Mock LLM generating answer for: {prompts[0][:100]}...")
                # Return mock completion IDs (10 tokens)
                return [[i for i in range(10)] for _ in range(len(prompts))]
        llm = MockLLM()
    
    # Evaluate the model on the environment
    logger.info(f"Starting evaluation with {num_episodes} episodes")
    metrics = env.evaluate_llm(
        llm_model=llm,
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    # Log and print results
    logger.info(f"Evaluation complete. Results: {metrics}")
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    logger.info("Math environment example completed")


def textarena_example(model_name, env_id="Sudoku-v0", num_episodes=2, verbose=True):
    """Run a simple example with the TextArenaMultistepEnv."""
    logger = setup_example_logging(f"textarena_{env_id.lower().replace('-', '_')}", level="DEBUG" if verbose else "INFO")
    logger.info(f"Starting TextArena environment example with env: {env_id}, model: {model_name}")
    
    print(f"\n=== TextArena Environment Example ({env_id}) ===\n")
    
    # Create the environment
    logger.info(f"Creating TextArenaMultistepEnv with {env_id}")
    env = TextArenaMultistepEnv(
        env_id=env_id,
        task_dataset_size=5,  # Small dataset for quick demo
        max_steps_per_episode=10,
        system_prompt=f"You are playing {env_id}. Follow the game rules and try to win.",
    )
    
    # Create the LLM
    logger.info(f"Initializing LLM with model: {model_name}")
    try:
        llm = LLM(model=model_name, max_model_len=4096)
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to mock LLM")
        print(f"Error loading model: {e}")
        print("Using mock LLM instead")
        # Create a mock LLM for testing
        from vllm.sampling_params import SamplingParams as SP
        class MockLLM:
            def generate(self, prompts, n=1, **kwargs):
                logger.debug(f"Mock LLM generating for prompt: {prompts[0][:100]}...")
                print(f"Mock LLM generating answer for: {prompts[0][:100]}...")
                # Return mock completion IDs (10 tokens)
                return [[i for i in range(10)] for _ in range(len(prompts))]
        llm = MockLLM()
    
    # Evaluate the model on the environment
    logger.info(f"Starting evaluation with {num_episodes} episodes")
    metrics = env.evaluate_llm(
        llm_model=llm,
        num_episodes=num_episodes,
        verbose=verbose
    )
    
    # Log and print results
    logger.info(f"Evaluation complete. Results: {metrics}")
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    logger.info("TextArena environment example completed")


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
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Optional directory name for logs (defaults to date-based)")
    
    args = parser.parse_args()
    
    # Set up base log directory if specified
    if args.log_dir:
        log_dir = ensure_logs_directory(args.log_dir)
        print(f"Logs will be stored in: {log_dir}")
    
    # Run the selected examples
    if args.env in ["math", "both"]:
        math_example(args.model, args.episodes, args.verbose)
    
    if args.env in ["textarena", "both"]:
        textarena_example(args.model, args.textarena_env, args.episodes, args.verbose)
    

if __name__ == "__main__":
    main() 