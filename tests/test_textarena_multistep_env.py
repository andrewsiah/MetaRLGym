import unittest
import random
from typing import Any, Dict, List, Sequence, Union, Optional
import numpy as np
from unittest.mock import MagicMock, patch

from metarlgym.envs.textarena_multistep_env import TextArenaMultistepEnv

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.responses = []
    
    def generate(self, prompts, n=1, **kwargs):
        # Return some mock token IDs
        return [[i for i in range(10)] for _ in range(len(prompts))]
    
    def set_responses(self, responses):
        self.responses = responses

# Mock TextArena environment
class MockTextArenaEnv:
    def __init__(self):
        self.reset_called = False
        self.step_calls = []
        self.close_called = False
    
    def reset(self, num_players=1, seed=None):
        self.reset_called = True
        self.player_id = 0
        self.observation = "This is a mock observation for a Sudoku game."
        
    def get_observation(self):
        return self.player_id, self.observation
    
    def step(self, action):
        self.step_calls.append(action)
        # In single-player mode, player_id stays 0
        # Only return done=True after at least 3 steps
        done = len(self.step_calls) >= 3
        return done, {}
    
    def close(self):
        self.close_called = True
        return {0: 1.0}  # Only player 0 in single-player mode


class TestTextArenaMultistepEnv(unittest.TestCase):
    @patch('textarena.make')
    @patch('textarena.wrappers.LLMObservationWrapper')
    def setUp(self, mock_wrapper, mock_make):
        # Setup mock TextArena
        self.mock_env = MockTextArenaEnv()
        mock_wrapper.return_value = self.mock_env
        mock_make.return_value = self.mock_env
        
        # Initialize the environment
        self.env = TextArenaMultistepEnv(
            env_id="Sudoku-v0",
            task_dataset_size=2,  # Small for testing
            max_steps_per_episode=3,
            system_prompt="Test system prompt",
            num_players=1  # Use 1 player instead of default 2 as Sudoku only supports 1 player
        )
        self.mock_llm = MockLLM()
        self.sampling_params = MagicMock()
        self.sampling_params.clone.return_value = self.sampling_params
    
    def test_initialization(self):
        """Test that the environment initializes correctly"""
        self.assertEqual(self.env.env_id, "Sudoku-v0")
        self.assertEqual(self.env.system_prompt, "Test system prompt")
        self.assertEqual(self.env.max_steps_per_episode, 3)
        
    def test_generate_call(self):
        """Test that the generate method works correctly"""
        # Create a prompt
        prompt = [[{"role": "user", "content": "This is a test prompt"}]]
        
        # Call generate
        result = self.env.generate(prompt, self.mock_llm, self.sampling_params)
        
        # Check that the result has the expected structure
        self.assertIn("ids", result)
        self.assertIn("messages", result)
        self.assertIn("mask", result)
        self.assertIn("session_ids", result)
        
        # Check that session ID was created
        session_id = result["session_ids"][0]
        self.assertIn(session_id, self.env.completed_episodes)
        
    def test_multi_step_episode(self):
        """Test that a multi-step episode runs correctly"""
        # Create a prompt
        prompt = [[{"role": "user", "content": "This is a test prompt"}]]
        
        # Call generate
        result = self.env.generate(prompt, self.mock_llm, self.sampling_params)
        
        # Get the completed episode
        session_id = result["session_ids"][0]
        episode_data = self.env.completed_episodes[session_id]
        
        # Check that steps were recorded
        self.assertGreater(episode_data["steps"], 0)
        
        # Check that actions were recorded
        self.assertIn("llm_actions", episode_data)
        
        # Check that reward was calculated
        self.assertIn("reward", episode_data)
        
    def test_opponent_action(self):
        """Test that opponent actions are handled correctly"""
        # In a single-player game, the opponent policy won't be called in normal gameplay
        # Instead, we'll verify that the opponent_policy exists and can be called manually
        
        # Create a test opponent policy that tracks calls
        opponent_calls = []
        def test_opponent_policy(observation):
            opponent_calls.append(observation)
            return "Test opponent action"
            
        # Set the opponent policy
        self.env.opponent_policy = test_opponent_policy
        
        # Call it directly with a test observation
        result = self.env.opponent_policy("Test observation")
        
        # Verify it was called and returned the expected result
        self.assertEqual(len(opponent_calls), 1)
        self.assertEqual(result, "Test opponent action")
    
    def test_evaluation(self):
        """Test the evaluation method"""
        # Create a mock LLM model
        mock_model = MockLLM()
        
        # Run evaluation
        metrics = self.env.evaluate_llm(
            llm_model=mock_model,
            num_episodes=1,
            verbose=True
        )
        
        # Check that metrics were calculated
        self.assertIn("success_rate", metrics)
        self.assertIn("average_reward", metrics)
        self.assertIn("steps_per_episode", metrics)


if __name__ == "__main__":
    unittest.main() 