import unittest
import random
from typing import Any, Dict, List, Sequence, Union, Optional
import numpy as np
from vllm import LLM, SamplingParams

from metarlgym.envs.multistep_env import MultistepEnv
from metarlgym.agents.directoutput.direct_output_agent import DirectOutputAgent

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
    def __init__(self, max_steps=3):
        super().__init__(
            env_id="SimpleMath-v0",
            task_dataset_size=10,
            system_prompt="You are solving a math problem. You can ask for hints.",
            max_steps_per_episode=max_steps,
        )
        
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
        
        self.task_dataset = {"prompt": problems, "solution": solutions}
        self.eval_task_dataset = {"prompt": problems[:2], "solution": solutions[:2]}
    
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
        initial_problem = self.task_dataset["prompt"][0][0]["content"] # Example: take first problem
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
        # current_solution = self.task_dataset["solution"][0] # Need to track current problem
        # if self._check_solution(action, current_solution):
        #     reward = 1
        # else:
        #     reward = -1
            
        return next_observation, reward, done, info

class TestMultistepEnv(unittest.TestCase):
    def setUp(self):
        self.env = SimpleMathEnv()
        self.mock_llm = MockLLM()
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    def test_initialization(self):
        """Test that the environment initializes correctly"""
        self.assertEqual(self.env.env_id, "SimpleMath-v0")
        self.assertEqual(self.env.max_steps_per_episode, 3)
        self.assertIsNotNone(self.env.task_dataset)
        
        prompts = [self.env.task_dataset["prompt"][0]]
        
        # Run generate
        # Create a mock agent to pass
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(prompts, self.mock_llm, self.sampling_params, agent=mock_agent)
        
        # Check that it returned the expected structure
        self.assertIn("ids", result)
        self.assertIn("messages", result)
        self.assertIn("mask", result)
    
    def test_generate_single_step(self):
        """Test generating a single step response"""
        prompts = [self.env.task_dataset["prompt"][0]]
        
        # Run generate
        # Create a mock agent to pass
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(prompts, self.mock_llm, self.sampling_params, agent=mock_agent)
        
        # Check that it returned the expected structure
        self.assertIn("ids", result)
        self.assertIn("messages", result)
        self.assertIn("mask", result)
        
    def test_full_episode(self):
        """Test running a full episode with multiple steps"""
        prompts = [self.env.task_dataset["prompt"][0]]
        
        # Set up mock responses
        self.mock_llm.set_responses(["Wrong answer", "Hint please?", "Correct answer: 42"])
        
        # Run generate
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(prompts, self.mock_llm, self.sampling_params, agent=mock_agent)
        
        # Check that the final reward is calculated
        self.assertIn("session_ids", result)
        
        # We should be able to get a session ID from the result
        session_id = result["session_ids"][0]
        
        # The session should have a completed episode
        self.assertIn(session_id, self.env.completed_episodes)
        
        # Check reward
        episode_data = self.env.completed_episodes[session_id]
        self.assertIn("reward", episode_data)
        
    def test_multistep_reasoning(self):
        """Test that the environment supports multiple reasoning steps"""
        prompts = [self.env.task_dataset["prompt"][0]]
        
        # Run generate which should handle multiple steps internally
        mock_agent = DirectOutputAgent(llm=self.mock_llm, sampling_params=self.sampling_params)
        result = self.env.run_trial(prompts, self.mock_llm, self.sampling_params, agent=mock_agent)
        
        # Get session ID
        session_id = result["session_ids"][0]
        episode_data = self.env.completed_episodes[session_id]
        
        # Check that steps were recorded
        self.assertIn("steps", episode_data)
        self.assertGreaterEqual(episode_data["steps"], 0)
        
        # Verify actions were recorded
        self.assertIn("llm_actions", episode_data)

if __name__ == "__main__":
    unittest.main() 