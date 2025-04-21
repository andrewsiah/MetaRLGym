import random
import pytest

from metarlgym.envs.TwentyQuestions.env import TwentyQuestionsEnv

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
    return TwentyQuestionsEnv(hardcore=False, max_turns=3, gamemaster_agent=mock_master)

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

def test_question_step_records_history_and_not_done(env):
    """Test that asking a question returns done=False and records history."""
    env.reset(num_players=1, seed=1)
    question = "Is it an animal?"
    done, info = env.step(question)
    assert done is False, "Environment should not be done after a question step."
    hist = env.state.game_state.get('history', [])
    assert hist, "History should not be empty after asking a question"
    last_q, last_a = hist[-1]
    assert last_q == question
    assert last_a in ["Yes", "No", "I don't know"]

def test_correct_guess_ends_episode(env):
    """Test that guessing acts as final step and ends episode."""
    env.reset(num_players=1, seed=2)
    correct = env.game_word
    done, info = env.step(f"[{correct}]")
    assert done is True, "Environment should be done after a correct guess."
    # reward for correct guess should be +1
    rewards = getattr(env.state, 'rewards', None)
    assert rewards is not None, "Rewards should be set on correct guess."
    assert rewards.get(0) == 1, "Winner reward should be 1"
    # info reason indicates success
    reason = info.get('reason', '').lower()
    assert 'congratulations' in reason or 'guessed' in reason, f"Unexpected win reason: {info.get('reason')!r}"

def test_incorrect_guess_logs_invalid_and_not_done(env):
    """Test first invalid move doesn't end game but logs warning."""
    env.reset(num_players=1, seed=3)
    done, info = env.step('[not_the_word]')
    # first invalid move should not end the game
    assert done is False, "Environment should NOT be done after the first incorrect guess"
    # logs should contain an invalid move warning
    # print(f"Logs before assertion in test_incorrect_guess_logs_invalid_and_not_done: {env.state.logs}") # DEBUG
    logs = getattr(env.state, 'logs', [])
    assert any('attempted an invalid move' in msg.lower() for _, msg in logs), "Log message for invalid move not found"

# Remove unittest structure
# class TestTwentyQuestionsEnv(unittest.TestCase):
#    ...

# if __name__ == '__main__':
#     unittest.main()