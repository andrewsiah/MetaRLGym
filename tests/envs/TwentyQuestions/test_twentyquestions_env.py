import unittest
import random

import textarena as ta
from metarlgym.envs.TwentyQuestions.env import TwentyQuestionsEnv


class MockGamemaster:
    """
    Simple mock gamemaster that returns predefined responses.
    """
    def __init__(self, responses):
        # Clone the list to avoid mutating input
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        if not self._responses:
            # Default to 'I don't know' when out of responses
            return "I don't know"
        return self._responses.pop(0)


class TestTwentyQuestionsEnv(unittest.TestCase):
    def setUp(self):
        # Use a small number of turns for testing
        self.max_turns = 3
        # Prepare mock responses for gamemaster
        # First question: 'Yes', second: 'No', default thereafter
        responses = ['Yes', 'No']
        self.mock_master = MockGamemaster(responses)
        # Initialize environment with injected mock gamemaster
        self.env = TwentyQuestionsEnv(hardcore=False, max_turns=self.max_turns, gamemaster_agent=self.mock_master)
        # Seed randomness for reproducibility
        random.seed(0)

    def test_reset_sets_game_state(self):
        """Test that reset initializes game_state with target_word and history."""
        # Use fixed seed for deterministic word selection
        seed = 42
        self.env.reset(num_players=1, seed=seed)
        # After reset, state.game_state should contain 'target_word'
        self.assertTrue(hasattr(self.env, 'state'))
        self.assertIn('target_word', self.env.state.game_state)
        # Initial rendered_text should include the target_word
        rendered = self.env.state.game_state.get('rendered_text', '')
        self.assertIn(self.env.state.game_state['target_word'], rendered)

    def test_question_step_returns_not_done(self):
        """Test that asking a question returns done=False and records history."""
        self.env.reset(num_players=1, seed=1)
        # Ask a yes/no question
        question = "Is it an animal?"
        done, info = self.env.step(question)
        # Since it's just a question, game should not be done
        self.assertFalse(done, msg="Environment should not be done after a question step.")
        # History should record the question and mock response
        hist = self.env.state.game_state.get('history', [])
        self.assertTrue(len(hist) >= 1)
        last_q, last_a = hist[-1]
        self.assertEqual(last_q, question)
        self.assertIn(last_a, ['Yes', 'No', "I don't know"])

    def test_guess_step_correct_and_incorrect(self):
        """Test that guessing acts as final step and ends episode."""
        # Test correct guess
        self.env.reset(num_players=1, seed=2)
        correct_word = self.env.game_word
        guess = f"[{correct_word}]"
        done, info = self.env.step(guess)
        self.assertTrue(done, msg="Environment should be done after a correct guess.")
        # After correct guess, rewards should be set indicating the winner
        rewards = getattr(self.env.state, 'rewards', None)
        self.assertIsNotNone(rewards, msg="Rewards should be set on correct guess.")
        # Assuming player 0 is the only player
        self.assertEqual(rewards.get(0), 1, msg="Winner reward should be 1")

        # Test incorrect guess
        self.env.reset(num_players=1, seed=3)
        wrong_guess = "[not_the_word]"
        done2, info2 = self.env.step(wrong_guess)
        self.assertTrue(done2, msg="Environment should be done after an incorrect guess.")
        # After incorrect guess (and exceeding error allowance), rewards should be set
        # invalids = getattr(self.env.state, 'invalid_moves', None)
        # self.assertIsNotNone(invalids, msg="Invalid moves should be recorded on wrong guess.")
        rewards2 = getattr(self.env.state, 'rewards', None)
        self.assertIsNotNone(rewards2, msg="Rewards should be set after incorrect guess leads to game end.")
        # Assuming player 0 is the only player and error_allowance=1
        self.assertEqual(rewards2.get(0), -1, msg="Loser reward should be -1 after incorrect guess")


if __name__ == '__main__':
    unittest.main()