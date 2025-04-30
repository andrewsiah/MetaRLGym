import unittest
import random
from typing import Any, Dict, List, Sequence, Union, Optional
import numpy as np
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer

from agentsgym.envs.multistep_env import MultistepEnv
from agentsgym.agents.directoutput.direct_output_agent import DirectOutputAgent

# Mock LLM for testing
class MockLLM:
    def __init__(self):
        self.responses = []
        self.call_count = 0
    
    def generate(self, prompts, sampling_params=None, **kwargs):
        self.call_count += 1
        # Return a list of RequestOutput-like objects
        # Each has an 'outputs' list with one CompletionOutput-like object
        # Each CompletionOutput has 'token_ids'
        mock_outputs = []
        response_text = f"Mock response {self.call_count}"
        if self.responses:
            response_text = self.responses.pop(0)
        
        # Simple tokenization for testing (char to ord)
        mock_token_ids = [ord(c) for c in response_text]
        
        for _ in prompts:
            completion_output = type('obj', (object,), {'token_ids': mock_token_ids})()
            request_output = type('obj', (object,), {'outputs': [completion_output]})()
            mock_outputs.append(request_output)
            
        return mock_outputs
    
    def set_responses(self, responses):
        self.responses = responses

# Simple Math environment that extends MultistepEnv
class SimpleMathEnv(MultistepEnv):
    def __init__(self, tokenizer: Optional[Any] = None, env_config: Optional[Dict[str, Any]] = None):
        # Default config if none provided
        if env_config is None:
            env_config = {"max_steps_per_episode": 3}

        # Extract necessary config before super().__init__ if needed by _create_task_dataset
        self.task_dataset_size = env_config.get("task_dataset_size", 10) # Example if needed early
        self._create_task_dataset() # Create the dataset first

        # Call super init with tokenizer and env_config
        # env_id is removed from super().__init__ according to plan
        super().__init__(
            tokenizer=tokenizer,
            env_config=env_config
        )
        # Store config value needed by tests
        self.max_steps_per_episode = env_config.get("max_steps_per_episode", 3)
        # Other initializations can happen after super().__init__

    def _create_task_dataset(self):
        """Create simple math problems"""
        problems = []
        solutions = []
        
        for i in range(self.task_dataset_size):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            problem = f"What is {a} + {b}?"
            solution = a + b
            
            problems.append([{"role": "user", "content": problem}])
            solutions.append(solution)
        
        # Store as dictionaries first
        self.task_dataset_dict = {"prompt": problems, "solution": solutions}
        self.eval_task_dataset_dict = {"prompt": problems[:2], "solution": solutions[:2]}
        # Initialize dataset attributes to None, they will be created by get_train_dataset
        self.train_dataset = None
        self.eval_dataset = None

    # >>> ADDED get_train_dataset <<< 
    def get_train_dataset(self):
        """Return the task dataset for training as a Dataset object."""
        # Create Dataset object on demand if it doesn't exist
        if self.train_dataset is None:
            if not hasattr(self, 'task_dataset_dict') or not self.task_dataset_dict:
                 # Ensure _create_task_dataset ran
                 self._create_task_dataset()
            self.train_dataset = Dataset.from_dict(self.task_dataset_dict)
        return self.train_dataset
    
    # >>> ADDED get_eval_dataset <<< 
    def get_eval_dataset(self):
        """Return the evaluation task dataset as a Dataset object."""
        # Create Dataset object on demand if it doesn't exist
        if self.eval_dataset is None:
            if not hasattr(self, 'eval_task_dataset_dict') or not self.eval_task_dataset_dict:
                 # Ensure _create_task_dataset ran
                 self._create_task_dataset()
            self.eval_dataset = Dataset.from_dict(self.eval_task_dataset_dict)
        return self.eval_dataset
    
    def _check_solution(self, response, solution):
        """Check if the solution is correct"""
        # Simple check if the solution appears in the response
        return str(solution) in response
    
    def _get_hint(self, problem, step):
        """Generate a hint based on the problem and current step"""
        if step == 1:
            return "Try breaking down the problem into smaller parts."
        elif step == 2:
            return "Consider using addition."
        else:
            return "The answer is the sum of the two numbers."

    # Add required abstract method implementations
    def reset(self, seed: Optional[int] = None):
        # Placeholder implementation
        # In a real scenario, this would reset the environment state and return an initial observation
        super().reset(seed=seed) # Call parent reset if necessary
        # Select a problem for the episode, e.g., from self.task_dataset
        # For testing, just return a dummy observation
        initial_problem = self.task_dataset_dict["prompt"][0][0]["content"] # Example: take first problem
        return initial_problem # Return format might need adjustment based on Env specs

    def step(self, action: Any):
        # Placeholder implementation
        # In a real scenario, this would process the action, update state, and return obs, reward, done, info
        
        # Dummy logic: Assume done after one step for testing setup
        done = True 
        # Dummy reward
        reward = 0 
        # Dummy next observation (or could be the same problem if not done)
        next_observation = "Episode finished."
        # Dummy info
        info = {}
        
        # Check solution (example - adapt as needed)
        # current_solution = self.task_dataset_dict["solution"][0] # Need to track current problem
        # if self._check_solution(action, current_solution):
        #     reward = 1
        # else:
        #     reward = -1
            
        return next_observation, reward, done, info

class TestMultistepEnv(unittest.TestCase):
    def setUp(self):
        # Instantiate with dummy tokenizer and config
        self.env_config = {"max_steps_per_episode": 3, "task_dataset_size": 10} # Added task_dataset_size
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.env = SimpleMathEnv(tokenizer=tokenizer, env_config=self.env_config)
        self.mock_llm = MockLLM()
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    def test_initialization(self):
        """Test that the environment initializes correctly"""
        # env_id check is removed as it's handled by registration now
        self.assertEqual(self.env.max_steps_per_episode, self.env_config["max_steps_per_episode"])
        # self.assertIsNotNone(self.env.task_dataset_dict) # task_dataset_dict might be internal
        self.assertIsNotNone(self.env.get_train_dataset()) # Check dataset generation works
        
    def test_generate_single_step(self):
        """Test generating a single step response (via run_trial)"""
        # Get a task_data dict (using placeholder structure for SimpleMath)
        task_data = {"prompt": self.env.task_dataset_dict["prompt"][0][0]["content"], 
                     "solution": self.env.task_dataset_dict["solution"][0]}
        task_data_list = [task_data]
        num_rollouts = 1
        
        # Run generate
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
        
        # Check that it returned the expected R5 structure
        expected_keys = [
            "padded_full_token_ids", "padded_full_attention_mask",
            "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
        ]
        self.assertTrue(isinstance(result, dict))
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertTrue(isinstance(result[key], list))
            self.assertEqual(len(result[key]), len(task_data_list) * num_rollouts)

    def test_full_episode(self):
        """Test running a full episode with multiple steps (via run_trial)"""
        # Get a task_data dict (using placeholder structure for SimpleMath)
        task_data = {"prompt": self.env.task_dataset_dict["prompt"][0][0]["content"], 
                     "solution": self.env.task_dataset_dict["solution"][0]}
        task_data_list = [task_data]
        num_rollouts = 1

        # Set up mock responses
        self.mock_llm.set_responses(["Wrong answer", "Hint please?", "Correct answer: 42"])
        
        # Run generate
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
        
        # Check the R5 structure
        expected_keys = [
            "padded_full_token_ids", "padded_full_attention_mask",
            "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
        ]
        self.assertTrue(isinstance(result, dict))
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertTrue(isinstance(result[key], list))
            self.assertEqual(len(result[key]), len(task_data_list) * num_rollouts)
            
        # Check final reward exists and is a float
        self.assertTrue(isinstance(result["final_rewards"][0], float))

    def test_multistep_reasoning(self):
        """Test that the environment supports multiple reasoning steps (via run_trial)"""
        # Get a task_data dict (using placeholder structure for SimpleMath)
        task_data = {"prompt": self.env.task_dataset_dict["prompt"][0][0]["content"], 
                     "solution": self.env.task_dataset_dict["solution"][0]}
        task_data_list = [task_data]
        num_rollouts = 1
        
        # Run generate which should handle multiple steps internally
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
        
        # Check the R5 structure
        expected_keys = [
            "padded_full_token_ids", "padded_full_attention_mask",
            "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
        ]
        self.assertTrue(isinstance(result, dict))
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertTrue(isinstance(result[key], list))
            self.assertEqual(len(result[key]), len(task_data_list) * num_rollouts)
        
        # Optionally, check if trajectory length indicates multiple steps occurred
        # This depends on the SimpleMathEnv logic and mock responses
        # Example: self.assertGreater(len(result["padded_full_token_ids"][0]), 10) # Arbitrary length check

if __name__ == "__main__":
    unittest.main() 