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

@pytest.mark.xfail(reason="Trial logic not yet implemented")
def test_get_dataset_and_eval_split():
    """Test that get_dataset and get_eval_dataset return disjoint sets with correct structure."""
    env = TwentyQuestionsEnv(hardcore=False, max_turns=3)
    # training dataset
    train_ds = env.get_dataset()
    assert hasattr(train_ds, 'column_names')
    assert 'prompt' in train_ds.column_names
    assert 'solution' in train_ds.column_names
    # evaluation dataset
    eval_ds = env.get_eval_dataset()
    # disjoint solutions
    train_sols = set(train_ds['solution'])
    eval_sols = set(eval_ds['solution'])
    assert train_sols.isdisjoint(eval_sols)

@pytest.mark.xfail(reason="Trial logic not yet implemented")
def test_reset_seed_reproducible():
    """Reset with same seed yields same hidden word."""
    env = TwentyQuestionsEnv(hardcore=False, max_turns=3)
    env.reset(num_players=1, seed=123)
    w1 = env.game_word
    env.reset(num_players=1, seed=123)
    w2 = env.game_word
    assert w1 == w2

@pytest.mark.xfail(reason="Trial logic not yet implemented")
def test_generate_structure():
    """Test that generate() returns correct top-level keys and consistent lengths."""
    # dummy agent always returns a dummy guess
    class DummyAgent:
        def get_action(self, messages, state):
            return ("[dummy]", state)

    env = TwentyQuestionsEnv(hardcore=False, max_turns=3, gamemaster_agent=MockGamemaster(["Yes", "No"]))
    prompts = [[{"content": "start trial"}]]
    # run generation
    output = env.generate(prompts=prompts, llm=None, sampling_params=None, agent=DummyAgent())
    # expected keys
    assert set(output.keys()) == {"ids", "messages", "mask", "session_ids"}
    # lengths match number of prompts
    assert len(output['ids']) == len(prompts)
    assert len(output['messages']) == len(prompts)
    assert len(output['mask']) == len(prompts)
    assert len(output['session_ids']) == len(prompts)
    
@pytest.mark.xfail(reason="Trial parameters not yet implemented")
def test_default_trial_params_exist():
    """Default trial parameters (episodes_per_trial, free_shots) are set and valid."""
    env = TwentyQuestionsEnv(hardcore=False, max_turns=3)
    # after implementing, env should have these attributes
    assert hasattr(env, 'episodes_per_trial')
    assert hasattr(env, 'free_shots')
    # defaults
    assert env.episodes_per_trial == 1
    assert env.free_shots == 0

@pytest.mark.xfail(reason="Trial parameter validation not yet implemented")
def test_invalid_free_shots_params_raise():
    """Setting free_shots >= episodes_per_trial should error."""
    # if episodes_per_trial=2, free_shots must be <2
    with pytest.raises(ValueError):
        TwentyQuestionsEnv(hardcore=False, max_turns=3,
                           episodes_per_trial=2,
                           free_shots=2)
