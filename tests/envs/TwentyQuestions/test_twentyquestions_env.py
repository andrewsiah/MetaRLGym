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

    assert "task_data" in train_ds.column_names # Check for part of new schema
    # evaluation dataset
    eval_ds = env.get_eval_dataset()
    # disjoint solutions (access via task_data)
    train_sols = set(item['solution'] for item in train_ds['task_data'])
    eval_sols = set(item['solution'] for item in eval_ds['task_data'])
    assert train_sols.isdisjoint(eval_sols)

    

def test_reset_seed_reproducible():
    """Reset with same seed yields same hidden word."""
    # Initialize using env_config
    env_config = {"hardcore": False, "max_turns": 3}
    env = TwentyQuestionsEnv(tokenizer=None, env_config=env_config)
    env.reset(num_players=1, seed=123)
    # Access game_word via the env, not a direct attribute if not exposed
    # Assuming reset updates some internal state accessible via a property or method
    # If env.game_word was the way, keep it, otherwise adjust as needed.
    # Let's assume env.game_word is correct for now based on original code.
    w1 = env.game_word
    env.reset(num_players=1, seed=123)
    w2 = env.game_word
    assert w1 == w2

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
def twenty_questions_env(mock_master):
    # Setup a basic environment for testing
    # Uses default configuration
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Need a real tokenizer
    env_config = {
        'hardcore': False,
        'max_turns': 5, # Shorten for testing
        'gamemaster_model_name': "mock", # Use mock model name, will be patched
        # 'gamemaster_agent': mock_master <<< Removed config injection
    }
    # Initialize the env normally (might log error for "mock" model if not caught)
    env = TwentyQuestionsEnv(tokenizer=tokenizer, env_config=env_config)
    # --- Monkeypatch the gamemaster instance --- 
    env.gamemaster = mock_master 
    # -----------------------------------------
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
    assert isinstance(env.gamemaster, MockGamemaster)

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
    # Check env_config matches (after removing non-serializable parts)
    expected_config = env.env_config.copy()
    expected_config.pop('gamemaster_agent', None)
    assert row["env_config"] == expected_config

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

def test_run_trial_multi_step(twenty_questions_env):
    """Tests running a multi-step trial where the agent asks questions before guessing."""
    env = twenty_questions_env
    
    # 1. Get task
    dataset = env.get_train_dataset()
    assert len(dataset) > 0, "Training dataset is empty, cannot run multi-step test."
    task_row = dataset[5] # Use a different task row
    task_data = task_row['task_data']
    solution = task_data['solution']
    print(f"\n--- Running Multi-Step Test ---")
    print(f"Target word: {solution}")

    # 2. Define multi-step agent responses (making an incorrect guess)
    incorrect_guess = "house"
    if solution == incorrect_guess:
        incorrect_guess = "tree" # Ensure guess is incorrect
        
    agent_responses = [
        "Is it man-made?",
        "Can you find it indoors?",
        "Is it smaller than a car?",
        f"[{incorrect_guess}]" # Final incorrect guess
    ]
    agent = MockAgent(agent_responses)
    print(f"Agent will ask {len(agent_responses)-1} questions then guess: {incorrect_guess}")

    # 3. Run trial
    num_rollouts = 1
    results = env.run_trial(task_data_list=[task_data], agent=agent, num_rollouts=num_rollouts)

    # 4. Assertions
    assert isinstance(results, dict)
    # Check structure
    expected_keys = [
        "padded_full_token_ids", "padded_full_attention_mask", 
        "padded_agent_token_mask", "padded_per_token_rewards", "final_rewards"
    ]
    for key in expected_keys:
        assert key in results
        assert len(results[key]) == num_rollouts, f"Expected 1 result for key '{key}', got {len(results[key])}"

    # Check reward (should be <= 0 for incorrect guess)
    final_reward = results["final_rewards"][0]
    print(f"Final reward for incorrect guess: {final_reward}")
    assert final_reward <= 0, "Expected non-positive reward for incorrect guess."

    # Check sequence length (should reflect multiple turns)
    seq_len = len(results["padded_full_token_ids"][0])
    print(f"Sequence length for multi-step trial: {seq_len}")
    # Estimate min length based on 3 Q/A pairs + guess. Varies with tokenizer.
    assert seq_len > 40, "Sequence length seems too short for multiple turns."
    print(f"--- Multi-Step Test Completed ---")
