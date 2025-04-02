#!/usr/bin/env python
"""Unit tests for TextArenaEnv."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Set environment variables to disable rendering
os.environ["TEXTARENA_DISABLE_RENDER"] = "1"  # Disable TextArena's rendering
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Hide pygame support message

class TestTextArenaEnv(unittest.TestCase):
    """Tests for TextArenaEnv."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Import here to avoid import errors affecting test discovery
        try:
            from metarlgym.envs.textarena_env import TextArenaEnv
            cls.TextArenaEnv = TextArenaEnv
        except ImportError as e:
            raise unittest.SkipTest(f"Could not import TextArenaEnv: {e}")
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create mock objects
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = [np.array([1, 2, 3, 4, 5])]
        
        self.mock_sp = MagicMock()
        self.mock_sp.clone.return_value = self.mock_sp
        for attr in ["repetition_penalty", "temperature", "top_p", 
                    "top_k", "min_p", "max_tokens"]:
            setattr(self.mock_sp, attr, 1.0)
    
    def test_environment_creation(self):
        """Test that the environment can be created."""
        # This test might still fail if the actual TicTacToe env has issues,
        # but the goal is to check TextArenaEnv initialization logic.
        # We rely on the mocked tests below for functional checks.
        try:
            env = self.TextArenaEnv(
                env_id="TicTacToe-v0", # Using a real (but simple) env ID
                task_dataset_size=10,  # Increased from 1
                max_steps_per_episode=2,
                num_players=2
            )
            self.assertIsNotNone(env)
            self.assertGreater(len(env.task_dataset), 0)
            self.assertGreater(len(env.eval_task_dataset), 0) # Check eval set too
        except ImportError as e:
             # If textarena or dependencies are missing
            raise unittest.SkipTest(f"Skipping test_environment_creation due to import error: {e}")
        except Exception as e:
            # Catch other potential errors during real env init
             self.fail(f"Environment creation failed with real components: {e}")
    
    @patch('textarena.wrappers.LLMObservationWrapper') # Patch the wrapper
    @patch('textarena.make') # Patch make
    def test_task_dataset_creation(self, mock_make, mock_wrapper_class):
        """Test task dataset creation with mocked TextArena and Wrapper."""
        # Mock the base env returned by ta.make
        mock_base_env = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_make.return_value = mock_base_env

        # Mock the wrapper instance returned by LLMObservationWrapper()
        mock_wrapped_env = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_wrapped_env.reset.return_value = None
        mock_wrapped_env.get_observation.return_value = (0, "Test observation during init")
        mock_wrapped_env.close.return_value = None
        mock_wrapper_class.return_value = mock_wrapped_env

        # Create the environment - this will call _create_task_dataset
        # which uses the mocked make and wrapper
        env = self.TextArenaEnv(
            env_id="mocked-id", # ID doesn't matter now
            task_dataset_size=2,
            max_steps_per_episode=2
        )
        
        # Check mocks used during init
        self.assertEqual(mock_make.call_count, 2) # Called twice in create_task_dataset
        self.assertEqual(mock_wrapper_class.call_count, 2)
        self.assertEqual(mock_wrapped_env.reset.call_count, 2)
        self.assertEqual(mock_wrapped_env.get_observation.call_count, 2)
        self.assertEqual(mock_wrapped_env.close.call_count, 2)

        # Check resulting dataset
        self.assertEqual(len(env.task_dataset), 1)
        self.assertEqual(len(env.eval_task_dataset), 1)
        self.assertTrue("prompt" in env.task_dataset[0])
        self.assertTrue("state" in env.task_dataset[0])
        self.assertIn("Test observation during init", env.task_dataset[0]['prompt'][0]['content'])
    
    @patch('textarena.wrappers.LLMObservationWrapper') # Patch the wrapper
    @patch('textarena.make') # Patch make
    def test_generation(self, mock_make, mock_wrapper_class):
        """Test generate method with mocked TextArena and Wrapper."""
        # --- Mock setup for _create_task_dataset phase ---
        mock_base_env_init = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_wrapped_env_init = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_wrapped_env_init.reset.return_value = None
        mock_wrapped_env_init.get_observation.return_value = (0, "Observation during init")
        mock_wrapped_env_init.close.return_value = None

        # --- Mock setup for generate phase ---
        mock_base_env_gen = MagicMock(spec=['reset', 'get_observation', 'step', 'close'])
        mock_wrapped_env_gen = MagicMock(spec=['reset', 'get_observation', 'step', 'close'])
        mock_wrapped_env_gen.reset.return_value = None
        mock_wrapped_env_gen.get_observation.side_effect = [
            (0, "Player 0 observation"),
            (1, "Opponent observation"),
            (0, "Player 0 observation again") # Should not be reached if max_steps=2
        ]
        mock_wrapped_env_gen.step.side_effect = [
            (False, {}), # LLM step
            (True, {})   # Opponent step ends episode
        ]
        mock_wrapped_env_gen.close.return_value = {0: 1.0, 1: -1.0} # Reward for player 0

        # Configure mocks to return different instances for init vs generate
        mock_make.side_effect = [mock_base_env_init] * 10 + [mock_base_env_gen] # 10 for init (size=10), 1 for generate
        mock_wrapper_class.side_effect = [mock_wrapped_env_init] * 10 + [mock_wrapped_env_gen]

        # Create the environment (uses init mocks)
        env = self.TextArenaEnv(
            env_id="mocked-id",
            task_dataset_size=10,
            max_steps_per_episode=2 # Max steps for generate phase
        )
        
        # Run generate (uses generate mocks)
        prompt_data = env.task_dataset[0] # Get a sample from the dataset created during init
        initial_prompt_messages = prompt_data['prompt']
        # Add state info if needed, though generate primarily uses session_id
        initial_prompt_messages[0]['state'] = prompt_data['state']

        result = env.generate([initial_prompt_messages], self.mock_llm, self.mock_sp)
        
        # --- Assertions ---
        self.assertIn("ids", result)
        self.assertIn("messages", result)
        self.assertIn("mask", result)
        self.assertIn("session_ids", result)
        
        # Check generate phase mock calls
        mock_wrapped_env_gen.reset.assert_called_once()
        self.assertEqual(mock_wrapped_env_gen.get_observation.call_count, 2) # Initial obs, obs after LLM step
        self.assertEqual(mock_wrapped_env_gen.step.call_count, 2) # One LLM step, one opponent step
        mock_wrapped_env_gen.close.assert_called_once()
        self.mock_llm.generate.assert_called_once() # LLM called once for player 0

        # Check completed_episodes
        session_id = result["session_ids"][0]
        self.assertIn(session_id, env.completed_episodes)
        self.assertEqual(env.completed_episodes[session_id]["reward"], 1.0)
    
    @patch('textarena.wrappers.LLMObservationWrapper') # Patch the wrapper
    @patch('textarena.make') # Patch make
    def test_reward_function(self, mock_make, mock_wrapper_class):
        """Test reward function with mocked TextArena and Wrapper."""
        # --- Mock setup for _create_task_dataset phase ---
        mock_base_env_init = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_wrapped_env_init = MagicMock(spec=['reset', 'get_observation', 'close'])
        mock_wrapped_env_init.reset.return_value = None
        mock_wrapped_env_init.get_observation.return_value = (0, "Observation during init")
        mock_wrapped_env_init.close.return_value = None

        # --- Mock setup for generate phase (called by test to get episode data) ---
        mock_base_env_gen = MagicMock(spec=['reset', 'get_observation', 'step', 'close'])
        mock_wrapped_env_gen = MagicMock(spec=['reset', 'get_observation', 'step', 'close'])
        mock_wrapped_env_gen.reset.return_value = None
        # Simulate episode ending immediately after player 0's (LLM) first action
        mock_wrapped_env_gen.get_observation.return_value = (0, "Player 0 observation")
        mock_wrapped_env_gen.step.return_value = (True, {})  # Episode ends on first step
        mock_wrapped_env_gen.close.return_value = {0: 2.0, 1: -2.0}  # Mock reward

        # Configure mocks
        mock_make.side_effect = [mock_base_env_init] * 10 + [mock_base_env_gen]
        mock_wrapper_class.side_effect = [mock_wrapped_env_init] * 10 + [mock_wrapped_env_gen]

        # Create the environment (uses init mocks)
        env = self.TextArenaEnv(
            env_id="mocked-id",
            task_dataset_size=10,
            max_steps_per_episode=2
        )
        
        # Run generate to create an episode and store reward (uses generate mocks)
        prompt_data = env.task_dataset[0]
        initial_prompt_messages = prompt_data['prompt']
        initial_prompt_messages[0]['state'] = prompt_data['state']
        result = env.generate([initial_prompt_messages], self.mock_llm, self.mock_sp)
        session_id = result["session_ids"][0]
        
        # Ensure episode data was stored correctly by generate
        self.assertIn(session_id, env.completed_episodes)
        self.assertEqual(env.completed_episodes[session_id]["reward"], 2.0)
        # Check generate phase mock calls
        mock_wrapped_env_gen.reset.assert_called_once()
        mock_wrapped_env_gen.get_observation.assert_called_once() # Only initial obs needed
        mock_wrapped_env_gen.step.assert_called_once() # LLM step ends episode
        mock_wrapped_env_gen.close.assert_called_once()
        self.mock_llm.generate.assert_called_once()

        # Get reward function
        reward_funcs = env.get_rubric()
        self.assertEqual(len(reward_funcs), 1)
        reward_func = reward_funcs[0]
        
        # Simulate call to reward function
        # Pass the session_id within a structure mimicking the 'prompts' argument
        mock_prompts_for_reward = [[{"role": "user", "content": "...", "session_id": session_id}]]
        mock_completions_for_reward = [[{"role": "assistant", "content": "test completion"}]]
        
        rewards = reward_func(prompts=mock_prompts_for_reward, completions=mock_completions_for_reward)
        
        # Check reward retrieved from completed_episodes
        self.assertEqual(rewards[0], 2.0)
        # Check that the episode data was cleaned up by the reward function
        self.assertNotIn(session_id, env.completed_episodes)

if __name__ == "__main__":
    unittest.main() 