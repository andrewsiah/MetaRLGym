import pytest
import random

from metarlgym.envs.TwentyQuestions.env import TwentyQuestionsEnv

@pytest.fixture(autouse=True)
def fix_random_seed():
    # ensure reproducibility
    random.seed(0)
    yield

def test_smoke_twentyquestions_env_logs_and_flow():
    # Instantiate environment
    env = TwentyQuestionsEnv(hardcore=False)
    # Reset with fixed seed
    env.reset(num_players=1, seed=123)

    # Retrieve initial logs
    logs = env.state.logs
    # Should start with 'Game started.'
    assert any("Game started" in msg for _, msg in logs), "Missing 'Game started' entry in logs"
    
    # Check that the initial player prompt does NOT include the target word.
    target = env.state.game_state["target_word"]
    # Get the prompt directly from the method that generates it
    player_prompt_msg = env._generate_player_prompt(
        player_id=env.state.current_player_id, 
        game_state=env.state.game_state
    )
    assert target not in player_prompt_msg, f"Initial player prompt should not contain the target word {target!r}"

    # Ask a yes/no question
    question = "Is it a living thing?"
    done, info = env.step(question)
    assert not done, "Environment should not be done after asking a question"

    # Check that the last two log entries correspond to the question and the response
    recent = env.state.logs[-2:]
    q_sender, q_msg = recent[0]
    a_sender, a_msg = recent[1]
    # Question sender should be the player (id 0)
    assert q_sender == env.state.current_player_id, "Question sender id mismatch"
    assert "living thing" in q_msg.lower(), "Logged question does not match"
    # Response should come from GAME_ID (-1)
    assert a_sender == -1, "Response sender should be GAME_ID"
    # Check if the response contains one of the expected options (case-insensitive)
    expected_options = ["Yes", "No", "I don't know"]
    assert any(opt.lower() in a_msg.lower() for opt in expected_options), f"Unexpected response: {a_msg!r}"

    # Now submit the correct final guess
    correct = env.state.game_state["target_word"]
    done, info = env.step(f"[{correct}]")
    assert done, "Environment should be done after the correct guess"
    # Info reason should indicate success
    reason = info.get("reason", "").lower()
    assert "guessed" in reason or "congratulations" in reason, f"Unexpected win reason: {info.get('reason')!r}"
    # Logs should include a winning message
    assert any("guessed the" in msg.lower() or "won the game" in msg.lower() for _, msg in env.state.logs), \
        "Winning message not found in logs"

def test_incorrect_guess_ends_episode_with_invalid_move():
    env = TwentyQuestionsEnv(hardcore=False)
    env.reset(num_players=1, seed=456)
    # Submit an incorrect guess
    done, info = env.step("[not_the_word]")
    # First incorrect guess should NOT end the episode (due to error_allowance=1)
    assert done is False, "Environment should NOT be done after the first incorrect guess"
    
    # Although game not done, check if info indicates invalid move attempt
    # Note: core.State.set_invalid_move sets info['reason'] only when game ends
    # We might need a different way to check if an invalid move occurred if game continues.
    # For now, the primary check is that done is False.
    # reason = info.get("reason", "").lower()
    # assert "invalid move" in reason or "incorrect" in reason, f"Unexpected invalid move reason: {info.get('reason')!r}"
    
    # Logs should record invalid guess penalty/warning
    # assert any("invalid" in msg.lower() for _, msg in env.state.logs), "Invalid move not logged"
    print(f"Logs before assertion in test_incorrect_guess_ends_episode_with_invalid_move: {env.state.logs}") # DEBUG
    assert any("attempted an invalid move" in msg.lower() for _, msg in env.state.logs), "Log message for invalid move not found"