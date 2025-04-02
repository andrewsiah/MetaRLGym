#!/usr/bin/env python
"""Minimal integration test for TextArenaEnv with GRPO training."""

import os
import sys
import argparse
import logging
import tempfile
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test TextArenaEnv with GRPO training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Small model to use for testing"
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="SpellingBee-v0",
        help="TextArena environment ID to test"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of tasks to initialize"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="localhost",
        help="vLLM server host"
    )
    parser.add_argument(
        "--vllm_server_port",
        type=int,
        default=8000,
        help="vLLM server port"
    )
    parser.add_argument(
        "--mock_vllm",
        action="store_true",
        help="Use mock vLLM instead of real vLLM server"
    )
    return parser.parse_args()

class MockVLLM:
    """Mock vLLM client for testing."""
    
    def __init__(self):
        self.calls = 0
    
    def generate(self, prompts, **kwargs):
        """Mock generate method that returns some token IDs."""
        self.calls += 1
        logger.info(f"Mock vLLM called with {len(prompts)} prompts (call #{self.calls})")
        return [[i+1 for i in range(10)] for _ in prompts]
    
    def update_named_param(self, name, data):
        """Mock method that pretends to update model parameters."""
        logger.info(f"Mock vLLM would update parameter: {name}")
        return True
    
    def reset_prefix_cache(self):
        """Mock method that pretends to reset cache."""
        logger.info("Mock vLLM would reset prefix cache")
        return True

def main():
    """Main test function."""
    args = parse_args()
    
    try:
        # Import necessary modules
        from metarlgym.envs.textarena_env import TextArenaEnv
        from trl import GRPOConfig
        from metarlgym.trainers.grpo_env_trainer import GRPOEnvTrainer
        
        # Create a temporary output directory
        output_dir = tempfile.mkdtemp(prefix="textarena_grpo_test_")
        logger.info(f"Using temporary output directory: {output_dir}")
        
        # Set up tokenizer
        logger.info(f"Loading tokenizer for {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Create TextArena environment
        logger.info(f"Creating TextArenaEnv with env_id={args.env_id}")
        env = TextArenaEnv(
            env_id=args.env_id,
            task_dataset_size=args.num_tasks,
            max_steps_per_episode=args.max_steps,
            tokenizer=tokenizer,
            seed=42
        )
        
        # Configure minimal GRPO for testing
        logger.info("Configuring minimal GRPO for testing")
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=1,            # Just 1 epoch for testing
            max_steps=2,                   # Only 2 training steps
            per_device_train_batch_size=2, # Small batch size
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            use_vllm=True,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            logging_steps=1,
            save_steps=10,                # Don't save during test
            eval_steps=10,                # Don't evaluate during test
            num_generations=2,            # Minimal number of completions
            temperature=0.7,
            max_prompt_length=128,
            max_completion_length=32,
            scale_rewards=True,
            seed=42,
            report_to="none",             # Disable reporting during test
        )
        
        # Set up mock vLLM if requested
        if args.mock_vllm:
            logger.info("Using mock vLLM for testing")
            # Monkey patch the VLLMClient import in GRPOTrainer
            import importlib
            import sys
            
            class MockVLLMClient:
                @staticmethod
                def from_pretrained(*args, **kwargs):
                    return MockVLLM()
            
            # Add mock module to sys.modules
            sys.modules['vllm'] = type('vllm', (), {'LLM': MockVLLMClient})
            
            # Also patch the vllm in trl
            sys.modules['trl.extras.vllm_client'] = type('vllm_client', (), {'VLLMClient': MockVLLM})
            
            # Set is_vllm_available to return True
            def mock_is_vllm_available():
                return True
            
            import trl.import_utils
            trl.import_utils.is_vllm_available = mock_is_vllm_available
        
        # Create GRPO trainer
        logger.info("Creating GRPOEnvTrainer")
        try:
            trainer = GRPOEnvTrainer(
                model=args.model_name,
                env=env,
                reward_funcs=env.get_rubric(),
                args=grpo_config,
                train_dataset=env.get_dataset(),
                eval_dataset=env.get_eval_dataset(),
                processing_class=tokenizer,
            )
            
            # Run minimal training
            logger.info("Starting minimal training (2 steps)")
            trainer.train()
            
            logger.info("Integration test completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you've installed the required packages")
        return 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 