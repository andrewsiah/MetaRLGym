"""
Example script for training a model on mathematical reasoning with a calculator tool.

This example demonstrates how to set up and run training with a ToolEnv
that uses a calculator tool to solve math problems from GSM8K.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM

from metarlgym.envs.tool_env import ToolEnv
from metarlgym.prompts.few_shots import CALCULATOR_FEW_SHOT
from metarlgym.prompts.templates import TOOL_SYSTEM_PROMPT_TEMPLATE
from metarlgym.tools.calculator import calculator
from metarlgym.trainers.grpo_env_trainer import GRPOEnvTrainer
from metarlgym.utils.logging_utils import ensure_logs_directory, get_model_name, setup_logging
from metarlgym.utils.model_utils import get_default_grpo_config

def main():
    parser = argparse.ArgumentParser(description="Train a model on GSM8K with calculator tool")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model to use")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_rollouts", type=int, default=7, help="Number of rollouts per prompt")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum steps per episode")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    parser.add_argument("--use_eval", action="store_true", help="Run evaluation during training")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    args = parser.parse_args()

    # Setup logging
    model_name = get_model_name(args.model)
    run_name = f"gsm8k-calc_{model_name}"
    log_dir = ensure_logs_directory(run_name)
    setup_logging(
        log_dir=log_dir,
        console_level=logging.INFO,
        file_level=logging.DEBUG
    )
    logger = logging.getLogger("calculator_training")
    logger.info(f"Starting training with model: {args.model}, run name: {run_name}")
    logger.info(f"Arguments: {args}")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure tool environment
    logger.info("Setting up tool environment...")
    env = ToolEnv(
        env_id="gsm8k-calculator",
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

    # Get dataset
    dataset = env.get_dataset()
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Setup evaluation dataset if needed
    eval_dataset = None
    if args.use_eval:
        eval_dataset = env.get_eval_dataset(n=100)
        logger.info(f"Evaluation dataset loaded with {len(eval_dataset)} examples")

    # Get reward functions
    rubric = env.get_rubric()
    logger.info(f"Reward functions: {[func.__name__ for func in rubric]}")

    # Training configuration
    training_args = get_default_grpo_config(
        run_name=run_name,
        num_gpus=args.num_gpus,
        logging_dir=log_dir
    )
    
    # Update training args
    training_args.num_generations = args.num_rollouts
    training_args.per_device_train_batch_size = args.batch_size
    training_args.gradient_accumulation_steps = args.grad_accum
    training_args.num_iterations = 2  # 1 on-policy, 1 off-policy
    training_args.max_steps = args.max_train_steps
    training_args.learning_rate = args.lr
    training_args.beta = args.beta
    
    # Configure evaluation
    if args.use_eval:
        training_args.eval_strategy = "steps"
        training_args.eval_steps = args.eval_steps
        training_args.per_device_eval_batch_size = args.batch_size
        training_args.eval_accumulation_steps = 1

    # Configure trainer
    logger.info("Initializing trainer...")
    trainer = GRPOEnvTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=rubric,
        env=env,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset if args.use_eval else None,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")

if __name__ == "__main__":
    main()