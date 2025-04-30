import random
import pytest
from agentsgym.envs.TwentyQuestions.env import TwentyQuestionsEnv
from agentsgym.agents.base import Agent
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Any, Dict, List, Tuple
import textarena as ta

class MockGamemaster:
    """
    Simple mock gamemaster that returns predefined responses.
    """
    def __init__(self, responses):
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        if not self._responses:
            return "I don't know"
        return self._responses.pop(0)

@pytest.fixture(autouse=True)
def fix_random_seed():
    random.seed(0)
    yield

@pytest.fixture
def mock_master():
    # First two responses: Yes, No; then default
    return MockGamemaster(responses=["Yes", "No"])

@pytest.fixture
def env(mock_master):
    # small max_turns for testing
    # Pass specific args via env_config
    env_config = {
        "hardcore": False,
        "max_turns": 3,
        "gamemaster_agent": mock_master # Still needed for internal logic
    }
    return TwentyQuestionsEnv(tokenizer=None, env_config=env_config)

def test_reset_sets_game_state(env):
    """Test that reset initializes game_state with target_word and history."""
    seed = 42
    env.reset(num_players=1, seed=seed)
    # game_state must include 'target_word'
    assert hasattr(env, 'state')
    gs = env.state.game_state
    assert 'target_word' in gs
    # rendered_text includes the target word
    rendered = gs.get('rendered_text', '')
    assert gs['target_word'] in rendered

def test_get_dataset_and_eval_split():
    """Test that get_dataset and get_eval_dataset return disjoint sets with correct structure."""
    # Instantiate with dummy tokenizer and config
    env_config = {"hardcore": False, "max_turns": 3}
    env = TwentyQuestionsEnv(tokenizer=None, env_config=env_config)
    # training dataset
    train_ds = env.get_train_dataset()
    assert hasattr(train_ds, 'column_names')
    assert 'prompt' in train_ds.column_names
    assert 'solution' in train_ds.column_names
    # evaluation dataset
    eval_ds = env.get_eval_dataset()
    # disjoint solutions
    train_sols = set(train_ds['solution'])
    eval_sols = set(eval_ds['solution'])
    assert train_sols.isdisjoint(eval_sols)

def test_reset_seed_reproducible():
    """Reset with same seed yields same hidden word."""
    env = TwentyQuestionsEnv(hardcore=False, max_turns=3)
    env.reset(num_players=1, seed=123)
    w1 = env.game_word
    env.reset(num_players=1, seed=123)
    w2 = env.game_word
    assert w1 == w2

def test_default_trial_params_exist():
    """Default trial parameters (episodes_per_trial, free_shots) are set and valid."""
    # Instantiate with dummy tokenizer and config
    env_config = {"hardcore": False, "max_turns": 3}
    env = TwentyQuestionsEnv(tokenizer=None, env_config=env_config)
    # after implementing, env should have these attributes
    assert hasattr(env, 'episodes_per_trial')
    assert hasattr(env, 'free_shots')
    # defaults
    assert env.episodes_per_trial == 1
    assert env.free_shots == 0

def test_invalid_free_shots_params_raise():
    """Setting free_shots >= episodes_per_trial should error."""
    # Pass params via env_config
    env_config = {
        "hardcore": False,
        "max_turns": 3,
        "episodes_per_trial": 2,
        "free_shots": 2
    }
    with pytest.raises(ValueError):
        TwentyQuestionsEnv(tokenizer=None, env_config=env_config)

# Helper function to create a mock agent
class MockAgent(Agent):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    def get_action(self, messages: List[Dict[str, Any]], state: Any = None) -> Tuple[str, Any]:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response, None # Return text and None state

@pytest.fixture
def twenty_questions_env():
    # Setup a basic environment for testing
    # Uses default configuration
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Need a real tokenizer
    env_config = {
        'hardcore': False,
        'max_turns': 5, # Shorten for testing
        'gamemaster_model_name': "mock" # Avoid real API calls if gamemaster is used
    }
    # Temporarily mock the OpenRouterAgent if needed to avoid API calls during init
    # For this test, we might not need it if _run_complete_episode uses our MockAgent
    env = TwentyQuestionsEnv(tokenizer=tokenizer, env_config=env_config)
    return env

@pytest.fixture
def mock_agent():
    # Simple agent that asks questions and then guesses
    return MockAgent(["Is it alive?", "Is it bigger than a breadbox?", "[testword]"])

def test_twenty_questions_initialization(twenty_questions_env):
    env = twenty_questions_env
    assert isinstance(env, TwentyQuestionsEnv)
    assert env.hardcore is False
    assert env.max_turns == 5
    assert env.tokenizer is not None
    assert env.env_config['max_turns'] == 5
    assert isinstance(env.gamemaster, ta.agents.OpenRouterAgent) # Or mock if gamemaster is mocked

def test_get_train_dataset_schema(twenty_questions_env):
    env = twenty_questions_env
    dataset = env.get_train_dataset()
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0 # Check dataset is not empty
    # Check schema of the first row
    row = dataset[0]
    assert "env_class_path" in row
    assert "env_config" in row
    assert "task_data" in row
    assert isinstance(row["env_class_path"], str)
    assert isinstance(row["env_config"], dict)
    assert isinstance(row["task_data"], dict)
    assert "solution" in row["task_data"]
    # Check env_config matches
    assert row["env_config"] == env.env_config

# Add the new test function
def test_run_trial_single_task(twenty_questions_env, mock_agent):
    """Tests running a single trial (1 task, 1 rollout) with the refactored env."""
    env = twenty_questions_env
    agent = mock_agent
    
    # 1. Get dataset and select one task
    dataset = env.get_train_dataset()
    assert len(dataset) > 0, "Training dataset is empty, cannot run trial test."
    task_row = dataset[0]
    task_data_list = [task_row['task_data']] # run_trial expects a list of task_data dicts
    num_rollouts = 1
    
    # 2. Call run_trial
    results = env.run_trial(task_data_list=task_data_list, agent=agent, num_rollouts=num_rollouts)
    
    # 3. Assertions on the results structure (basic checks based on R5)
    assert isinstance(results, dict)
    expected_keys = [
        "padded_full_token_ids", "padded_full_attention_mask", 
        "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
    ]
    for key in expected_keys:
        assert key in results
        assert isinstance(results[key], list)
        assert len(results[key]) == len(task_data_list) * num_rollouts # Should be 1 trajectory
        
    # Check trajectory data types and consistency
    assert isinstance(results["padded_full_token_ids"][0], list) 
    assert isinstance(results["padded_full_attention_mask"][0], list)
    assert isinstance(results["padded_agent_token_mask"][0], list)
    assert isinstance(results["padded_per_token_rewards"][0], list)
    assert isinstance(results["final_rewards"][0], float)
    
    # Check that all padded sequences in this single task group have the same length
    seq_len = len(results["padded_full_token_ids"][0])
    assert seq_len > 0, "Padded sequence length is zero."
    assert len(results["padded_full_attention_mask"][0]) == seq_len
    assert len(results["padded_agent_token_mask"][0]) == seq_len
    assert len(results["padded_per_token_rewards"][0]) == seq_len
    
    print(f"run_trial completed successfully for one task. Output shape (tokens): {seq_len}")

# You might need other tests for _initialize_episode, _step_episode, _run_complete_episode 
# based on the T3, T4 requirements in env_update.md, but this covers the user request.

# Example test (similar to T3) for _run_complete_episode (needs adaptation)
# def test_run_complete_episode_output(twenty_questions_env, mock_agent):
#     env = twenty_questions_env
#     agent = mock_agent
#     tokenizer = env.tokenizer
#     session_id = "test_session_rce"
#     task_data = {"solution": "testword"}

#     # Need to manually initialize state first
#     initial_state = env._initialize_episode(session_id, task_data)
#     env.active_states[session_id] = initial_state
#     env.agent_states[session_id] = None
    
#     # Run the episode
#     episode_results = env._run_complete_episode(session_id, agent)

#     # Assertions based on R4 output format
#     assert isinstance(episode_results, dict)
#     expected_keys = ["full_token_ids", "full_attention_mask", "agent_token_mask", "per_token_rewards", "final_reward"]
#     for key in expected_keys:
#         assert key in episode_results
    
#     assert isinstance(episode_results["full_token_ids"], list)
#     assert isinstance(episode_results["full_attention_mask"], list)
#     # ... more detailed checks on content, masks, rewards distributions etc.
