#!/usr/bin/env python
"""Minimal GRPO integration test with mocked dependencies."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import numpy as np

# Set environment variables to disable rendering
os.environ["TEXTARENA_DISABLE_RENDER"] = "1"  # Disable TextArena's rendering
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Hide pygame support message

# Mock the vLLM modules
class MockVLLM:
    """Mock vLLM client."""
    def generate(self, *args, **kwargs):
        return [[i+1 for i in range(10)] for _ in range(len(args[0]))]
    
    def update_named_param(self, *args, **kwargs):
        return True
    
    def reset_prefix_cache(self):
        return True

# Create mocks for import
sys.modules['vllm'] = MagicMock()
sys.modules['vllm'].LLM = MockVLLM

# Create mock for trl.extras.vllm_client
mock_vllm_client = MagicMock()
mock_vllm_client.VLLMClient = MagicMock()
mock_vllm_client.VLLMClient.return_value = MockVLLM()
sys.modules['trl.extras.vllm_client'] = mock_vllm_client

# Mock is_vllm_available to return True
sys.modules['trl.import_utils'] = MagicMock()
sys.modules['trl.import_utils'].is_vllm_available = MagicMock(return_value=True)

# Create a mock tokenizer
class MockTokenizer:
    """Mock tokenizer."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 0
    
    def batch_decode(self, *args, **kwargs):
        return ["test response"] * len(args[0])
    
    def __call__(self, *args, **kwargs):
        result = MagicMock()
        result.input_ids = torch.ones((len(args[0]), 5), dtype=torch.long)
        result.attention_mask = torch.ones((len(args[0]), 5), dtype=torch.long)
        return result

# Import after mocking
import torch
from transformers import TrainingArguments

class TestMinimalGRPO(unittest.TestCase):
    """Minimal GRPO integration test."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once."""
        try:
            from metarlgym.envs.textarena_env import TextArenaEnv
            cls.TextArenaEnv = TextArenaEnv
            
            from trl import GRPOConfig
            cls.GRPOConfig = GRPOConfig
            
            from metarlgym.trainers.grpo_env_trainer import GRPOEnvTrainer
            cls.GRPOEnvTrainer = GRPOEnvTrainer
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import required modules: {e}")
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for output
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('textarena.make')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_minimal_grpo_integration(self, mock_tokenizer, mock_model, mock_make):
        """Test minimal GRPO integration."""
        # Set up mocks
        mock_tokenizer.return_value = MockTokenizer()
        
        mock_model_obj = MagicMock()
        mock_model_obj.config = MagicMock()
        mock_model_obj.config._name_or_path = "test_model"
        mock_model_obj.warnings_issued = {"estimate_tokens": True}
        mock_model.return_value = mock_model_obj
        
        # Mock TextArena environment
        mock_env = MagicMock()
        mock_env.reset.return_value = None
        mock_env.get_observation.return_value = (0, "Test observation")
        mock_env.step.return_value = (True, {})  # Episode ends immediately
        mock_env.close.return_value = {0: 1.0}  # Mock reward
        mock_make.return_value = mock_env
        
        # Create the environment
        env = self.TextArenaEnv(
            env_id="TicTacToe-v0",
            task_dataset_size=2,
            max_steps_per_episode=2
        )
        
        # Configure minimal GRPO
        grpo_config = self.GRPOConfig(
            output_dir=self.temp_dir,
            num_train_epochs=1,
            max_steps=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            use_vllm=True,
            vllm_server_host="localhost",
            vllm_server_port=8000,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            num_generations=1,
            temperature=0.7,
            max_prompt_length=5,
            max_completion_length=5,
            report_to="none",
        )
        
        # Create trainer with minimal training
        with patch.object(self.GRPOEnvTrainer, 'train', return_value=None):
            trainer = self.GRPOEnvTrainer(
                model="test_model",
                env=env,
                reward_funcs=env.get_rubric(),
                args=grpo_config,
                train_dataset=env.get_dataset(),
                eval_dataset=env.get_eval_dataset(),
                processing_class=mock_tokenizer.return_value,
            )
            
            # Just check that trainer was created successfully
            self.assertIsNotNone(trainer)
            self.assertEqual(trainer.env, env)

if __name__ == "__main__":
    unittest.main() 