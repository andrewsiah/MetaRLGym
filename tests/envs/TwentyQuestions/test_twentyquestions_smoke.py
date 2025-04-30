import pytest
import random

from agentsgym.envs.TwentyQuestions.env import TwentyQuestionsEnv
from agentsgym.agents.base import Agent
from transformers import AutoTokenizer
from typing import Any, Dict, List, Tuple

# Helper class (copied from test_twentyquestions_env.py for simplicity)
class MockAgent(Agent):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    def get_action(self, messages: List[Dict[str, Any]], state: Any = None) -> Tuple[str, Any]:
        if self.call_count >= len(self.responses):
            # Default behavior if more actions are requested than provided
            return "[some_default_guess]", None 
        response = self.responses[self.call_count]
        self.call_count += 1
        return response, None

@pytest.fixture(autouse=True)
def fix_random_seed():
    # ensure reproducibility
    random.seed(0)
    yield

def test_smoke_twentyquestions_env_logs_and_flow():
    """ Test the basic flow: ask question, get response, guess correctly."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Use a mock gamemaster that gives predictable answers
    class LocalMockGamemaster:
        def __call__(self, prompt: str) -> str:
            if "living thing" in prompt.lower():
                return "Yes"
            return "I don't know"
    
    env_config = {
        "hardcore": False, 
        "max_turns": 5, 
        "gamemaster_model_name": "mock" # Will be patched
    }
    env = TwentyQuestionsEnv(tokenizer=tokenizer, env_config=env_config)
    # --- Monkeypatch the gamemaster instance --- 
    env.gamemaster = LocalMockGamemaster()
    # -----------------------------------------
    
    # Reset to get the target word for task_data
    env.reset(num_players=1, seed=123)
    target_word = env.game_word # Assumes reset sets this attribute
    task_data = {"solution": target_word, "seed": 123}
    
    # Agent asks a question, then guesses correctly
    mock_agent = MockAgent(["Is it a living thing?", f"[{target_word}]"])
    
    # Run the trial
    results = env.run_trial(task_data_list=[task_data], agent=mock_agent, num_rollouts=1)
    
    # Assertions on results
    assert len(results["final_rewards"]) == 1, "Expected one final reward"
    # Successful guess should yield positive reward (assuming default TA reward)
    assert results["final_rewards"][0] > 0, f"Expected positive reward for correct guess, got {results['final_rewards'][0]}"
    # Check sequence length is reasonable
    assert len(results["padded_full_token_ids"][0]) > 10, "Expected non-trivial sequence length"

def test_incorrect_guess_ends_episode_with_invalid_move():
    """ Test that an incorrect guess results in a penalty/negative reward."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Mock gamemaster (not strictly needed as we only guess)
    class LocalMockGamemaster:
        def __call__(self, prompt: str) -> str: return "I don't know"
        
    env_config = {
        "hardcore": False, 
        "max_turns": 5,
        "gamemaster_model_name": "mock" # Will be patched
    }
    env = TwentyQuestionsEnv(tokenizer=tokenizer, env_config=env_config)
    # --- Monkeypatch the gamemaster instance --- 
    env.gamemaster = LocalMockGamemaster()
    # -----------------------------------------
    
    # Reset to get target word
    env.reset(num_players=1, seed=456)
    target_word = env.game_word
    task_data = {"solution": target_word, "seed": 456}
    
    # Agent guesses incorrectly
    mock_agent = MockAgent(["[incorrect_guess]"])
    
    # Run the trial
    results = env.run_trial(task_data_list=[task_data], agent=mock_agent, num_rollouts=1)

    # Assertions on results
    assert len(results["final_rewards"]) == 1
    # Incorrect guess should yield non-positive reward (likely negative due to penalty)
    assert results["final_rewards"][0] <= 0, f"Expected non-positive reward for incorrect guess, got {results['final_rewards'][0]}"