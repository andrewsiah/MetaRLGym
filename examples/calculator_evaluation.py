"""
Example script for evaluating a model on mathematical reasoning with a calculator tool.

This example demonstrates how to set up and run evaluation of OpenAI API models
with a ToolEnv that uses a calculator tool to solve math problems from GSM8K.
"""

import argparse
import logging
import os
from pathlib import Path
import time

from openai import OpenAI
from transformers import AutoTokenizer

from metarlgym.envs.tool_env import ToolEnv
from metarlgym.prompts.few_shots import CALCULATOR_FEW_SHOT
from metarlgym.prompts.templates import TOOL_SYSTEM_PROMPT_TEMPLATE
from metarlgym.tools.calculator import calculator
from metarlgym.utils.logging_utils import ensure_logs_directory, setup_logging

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K with calculator tool")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to evaluate")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum concurrent requests")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum steps per episode")
    args = parser.parse_args()

    # Setup logging
    run_name = f"eval-{args.model.replace('/', '-')}-gsm8k-calc"
    log_dir = ensure_logs_directory(run_name)
    setup_logging(
        log_dir=log_dir,
        console_level=logging.INFO,
        file_level=logging.DEBUG
    )
    logger = logging.getLogger("calculator_evaluation")
    logger.info(f"Starting evaluation with model: {args.model}, run name: {run_name}")
    logger.info(f"Arguments: {args}")

    # Load tokenizer for tokenizing if not using OpenAI API
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except:
        logger.warning("Could not load tokenizer, using None")
    
    # Configure tool environment
    logger.info("Setting up tool environment...")
    env = ToolEnv(
        env_id="gsm8k-calculator-eval",
        dataset_name="gsm8k",
        tools=[calculator],
        system_prompt_template=TOOL_SYSTEM_PROMPT_TEMPLATE,
        few_shot=CALCULATOR_FEW_SHOT[0],
        max_steps_per_episode=args.max_steps,
        sampling_args={
            "stop": ["</tool>", "</answer>"],
            "include_stop_str_in_output": True
        },
        tokenizer=tokenizer
    )
    
    # Initialize OpenAI client
    logger.info("Initializing OpenAI client...")
    client = OpenAI()
    
    # Start timing
    start_time = time.time()
    
    # Run evaluation
    logger.info(f"Running evaluation on {args.num_examples} examples...")
    metrics = env.eval_api(
        client=client,
        model=args.model,
        n=args.num_examples,
        max_concurrent=args.max_concurrent
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Log results
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results to a file
    results_path = os.path.join(log_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: gsm8k\n")
        f.write(f"Examples: {args.num_examples}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"Time: {elapsed_time:.2f} seconds\n\n")
        f.write("Results:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()