import pytest
import random
from typing import Any, Dict, List, Sequence, Union, Optional, Tuple
import numpy as np
from vllm import LLM, SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agentsgym.envs.multistep_env import MultistepEnv
from agentsgym.agents.directoutput.direct_output_agent import DirectOutputAgent
from agentsgym.agents.base import Agent

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


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def tokenizer():
    # Using a simple tokenizer for testing
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def env_config():
    """Fixture for environment configuration."""
    return {"max_steps_per_episode": 3, "task_dataset_size": 10}

@pytest.fixture
def mock_llm():
    """Fixture for the MockLLM."""
    return MockLLM()

@pytest.fixture
def sampling_params():
    """Fixture for SamplingParams."""
    return SamplingParams(temperature=0.7, max_tokens=100)

@pytest.fixture
def simple_math_env(tokenizer, env_config):
    """Fixture for the SimpleMathEnv instance."""
    return SimpleMathEnv(tokenizer=tokenizer, env_config=env_config)

@pytest.fixture
def mock_agent(mock_llm, sampling_params):
    """Fixture for the DirectOutputAgent using MockLLM."""
    return DirectOutputAgent(llm=mock_llm, sampling_params=sampling_params)


# --- Pytest Test Functions (Refactored from unittest methods) ---

def test_simple_math_env_initialization(simple_math_env, env_config):
    """Test that the SimpleMathEnv initializes correctly using pytest style."""
    assert simple_math_env.max_steps_per_episode == env_config["max_steps_per_episode"]
    assert simple_math_env.get_train_dataset() is not None # Check dataset generation works


def test_simple_math_env_generate_single_step(simple_math_env, mock_agent):
    """Test generating a single step response (via run_trial) using pytest style."""
    # Get a task_data dict (using placeholder structure for SimpleMath)
    # Ensure dataset is created before accessing task_dataset_dict directly
    _ = simple_math_env.get_train_dataset() 
    task_data = {"prompt": simple_math_env.task_dataset_dict["prompt"][0][0]["content"], 
                 "solution": simple_math_env.task_dataset_dict["solution"][0]}
    task_data_list = [task_data]
    num_rollouts = 1
    
    # Run generate
    result = simple_math_env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
    
    # Check that it returned the expected R5 structure
    expected_keys = [
        "padded_full_token_ids", "padded_full_attention_mask",
        "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
    ]
    assert isinstance(result, dict)
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], list)
        assert len(result[key]) == len(task_data_list) * num_rollouts


def test_simple_math_env_full_episode(simple_math_env, mock_agent, mock_llm):
    """Test running a full episode with multiple steps (via run_trial) using pytest style."""
    # Get a task_data dict (using placeholder structure for SimpleMath)
    _ = simple_math_env.get_train_dataset() 
    task_data = {"prompt": simple_math_env.task_dataset_dict["prompt"][0][0]["content"], 
                 "solution": simple_math_env.task_dataset_dict["solution"][0]}
    task_data_list = [task_data]
    num_rollouts = 1

    # Set up mock responses
    mock_llm.set_responses(["Wrong answer", "Hint please?", "Correct answer: 42"])
    
    # Run generate
    result = simple_math_env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
    
    # Check the R5 structure
    expected_keys = [
        "padded_full_token_ids", "padded_full_attention_mask",
        "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
    ]
    assert isinstance(result, dict)
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], list)
        assert len(result[key]) == len(task_data_list) * num_rollouts
            
    # Check final reward exists and is a float
    assert isinstance(result["final_rewards"][0], float)


def test_simple_math_env_multistep_reasoning(simple_math_env, mock_agent):
    """Test that the environment supports multiple reasoning steps (via run_trial) using pytest style."""
    # Get a task_data dict (using placeholder structure for SimpleMath)
    _ = simple_math_env.get_train_dataset() 
    task_data = {"prompt": simple_math_env.task_dataset_dict["prompt"][0][0]["content"], 
                 "solution": simple_math_env.task_dataset_dict["solution"][0]}
    task_data_list = [task_data]
    num_rollouts = 1
    
    # Run generate which should handle multiple steps internally
    result = simple_math_env.run_trial(task_data_list=task_data_list, agent=mock_agent, num_rollouts=num_rollouts)
    
    # Check the R5 structure
    expected_keys = [
        "padded_full_token_ids", "padded_full_attention_mask",
        "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
    ]
    assert isinstance(result, dict)
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], list)
        assert len(result[key]) == len(task_data_list) * num_rollouts
        
    # Optionally, check if trajectory length indicates multiple steps occurred
    # This depends on the SimpleMathEnv logic and mock responses
    # Example: assert len(result["padded_full_token_ids"][0]) > 10 # Arbitrary length check

# A simple agent that returns a fixed action
class DummyAgent(Agent):
    def __init__(self, action_text="fixed_action"):
        self._action_text = action_text

    def get_action(
        self,
        messages: List[Dict[str, Any]],
        agent_state: Optional[Any] = None
    ) -> Tuple[str, Any]:
        return self._action_text, None # Return fixed action and no state update


# A minimal MultistepEnv subclass for testing reward logic
class DummyMultistepEnv(MultistepEnv):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, env_config: Dict[str, Any]):
        super().__init__(tokenizer, env_config)
        # Store reward and done status to be returned by step
        self.mock_step_reward = env_config.get("mock_step_reward", 0.0)
        self.mock_step_done = env_config.get("mock_step_done", True)
        self.step_called = 0

    def get_train_dataset(self, **kwargs: Any) -> Dataset:
        # Provide a minimal dataset structure compliant with R1 schema
        return Dataset.from_list([{
            "env_class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "env_config": self.env_config,
            "task_data": {"content": "dummy_task"}
        }])

    def _initialize_episode(self, session_id, task_info: Dict[str, Any]):
        self.step_called = 0 # Reset step count for new episode
        return {"observation": task_info.get("content", ""), "done": False, "steps": 0}

    def reset(self, seed: Optional[int] = None):
        """Minimal implementation to satisfy abstract method requirement."""
        # In a real scenario, this would reset state and return an initial observation.
        # For this dummy class, we don't need complex logic.
        # We can return a dummy observation or None.
        self.logger.info(f"Dummy reset called with seed: {seed}")
        initial_obs = self._initialize_episode("dummy_session", {"content": "dummy_task"})
        return initial_obs # Or return suitable initial state structure

    def step(self, session_id, state, llm_action):
        self.step_called += 1
        next_state = state.copy()
        next_state["steps"] += 1
        # Only return the configured reward/done on the first step for simplicity
        if self.step_called == 1:
            reward = self.mock_step_reward
            done = self.mock_step_done
        else:
            reward = 0.0 # No reward after first step
            done = True # End after >1 step
        
        info = {"agent_action": llm_action} # Include agent action as expected
        # Ensure done is boolean for loop control in _run_complete_episode
        is_done = done if isinstance(done, bool) else done[0] 
        next_state["done"] = is_done 
        return next_state, reward, is_done, info

    def _format_prompt(self, state, step):
        return f"Prompt for step {step+1}" # Simple prompt

    def _calculate_reward(self, state, llm_actions, final_action, step_rewards: List[float]):
        # Use the default cumulative reward calculation for simplicity
        return super()._calculate_reward(state, llm_actions, final_action, step_rewards)

def test_run_trial_reward_broadcasting(tokenizer):
    """
    Verify that _run_complete_episode (called by run_trial) broadcasts
    the step reward to all agent tokens for that step.
    """
    agent_action_text = "agent response"
    dummy_agent = DummyAgent(action_text=agent_action_text)
    task_data = {"content": "initial observation"}

    # Scenario 1: Positive Reward
    env_config_pos = {"mock_step_reward": 1.0, "mock_step_done": True}
    env_pos = DummyMultistepEnv(tokenizer, env_config_pos)
    results_pos = env_pos.run_trial(task_data_list=[task_data], agent=dummy_agent, num_rollouts=1)

    # Scenario 2: Zero Reward
    env_config_zero = {"mock_step_reward": 0.0, "mock_step_done": True}
    env_zero = DummyMultistepEnv(tokenizer, env_config_zero)
    results_zero = env_zero.run_trial(task_data_list=[task_data], agent=dummy_agent, num_rollouts=1)

    # --- Verification ---
    agent_tokens = tokenizer.encode(agent_action_text, add_special_tokens=False)
    agent_token_len = len(agent_tokens)

    # Check Positive Reward Scenario
    per_token_rewards_pos = results_pos["padded_per_token_rewards"][0]
    agent_mask_pos = results_pos["padded_agent_token_mask"][0]
    
    agent_reward_indices_pos = [i for i, mask_val in enumerate(agent_mask_pos) if mask_val == 1]
    
    assert len(agent_reward_indices_pos) == agent_token_len, "Incorrect number of agent tokens identified (Positive)"
    for idx in agent_reward_indices_pos:
        assert per_token_rewards_pos[idx] == pytest.approx(1.0), \
            f"Agent token at index {idx} should have reward 1.0, but got {per_token_rewards_pos[idx]}"
    # Check non-agent tokens have 0 reward (assuming only one step reward)
    non_agent_reward_sum_pos = sum(per_token_rewards_pos[i] for i, mask_val in enumerate(agent_mask_pos) if mask_val == 0)
    assert non_agent_reward_sum_pos == pytest.approx(0.0), "Non-agent tokens should have 0 reward (Positive)"


    # Check Zero Reward Scenario
    per_token_rewards_zero = results_zero["padded_per_token_rewards"][0]
    agent_mask_zero = results_zero["padded_agent_token_mask"][0]

    agent_reward_indices_zero = [i for i, mask_val in enumerate(agent_mask_zero) if mask_val == 1]

    assert len(agent_reward_indices_zero) == agent_token_len, "Incorrect number of agent tokens identified (Zero)"
    for idx in agent_reward_indices_zero:
         assert per_token_rewards_zero[idx] == pytest.approx(0.0), \
             f"Agent token at index {idx} should have reward 0.0, but got {per_token_rewards_zero[idx]}"
    # Check non-agent tokens have 0 reward
    non_agent_reward_sum_zero = sum(per_token_rewards_zero[i] for i, mask_val in enumerate(agent_mask_zero) if mask_val == 0)
    assert non_agent_reward_sum_zero == pytest.approx(0.0), "Non-agent tokens should have 0 reward (Zero)"

