#!/usr/bin/env python
"""Quick test script for TextArenaEnv implementation."""

import os
import sys
import argparse
import logging
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test TextArenaEnv implementation")
    parser.add_argument(
        "--env_id",
        type=str,
        default="SpellingBee-v0",
        help="TextArena environment ID to test"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=5,
        help="Number of tasks to initialize"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to run"
    )
    return parser.parse_args()

class MockLLM:
    """Mock LLM for testing that returns a simple response."""
    
    def __init__(self, fixed_response=None):
        self.fixed_response = fixed_response or "I choose A"
        self.calls = 0
    
    def generate(self, prompts, **kwargs):
        """Mock generate method that returns token IDs."""
        self.calls += 1
        logger.info(f"LLM called with {len(prompts)} prompts (call #{self.calls})")
        
        # Create random token IDs as a simple response
        results = []
        for _ in prompts:
            # Create token IDs for the fixed response
            token_ids = np.array([i+1 for i in range(len(self.fixed_response))])
            results.append(token_ids)
        
        return results

class MockSamplingParams:
    """Mock sampling parameters for testing."""
    
    def __init__(self):
        self.repetition_penalty = 1.0
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 40
        self.min_p = 0.0
        self.max_tokens = 100
    
    def clone(self):
        return self

def test_task_dataset(env):
    """Test that the task dataset was created successfully."""
    logger.info("Testing task dataset creation...")
    
    # Check dataset sizes
    logger.info(f"Train task dataset size: {len(env.task_dataset)}")
    logger.info(f"Eval task dataset size: {len(env.eval_task_dataset)}")
    
    # Check first task
    if len(env.task_dataset) > 0:
        first_task = env.task_dataset[0]
        logger.info(f"First task prompt: {first_task['prompt']}")
        logger.info(f"First task state keys: {first_task['state'].keys()}")
    
    return len(env.task_dataset) > 0

def test_episode_execution(env, mock_llm, mock_sp, num_episodes=1):
    """Test running complete episodes."""
    logger.info(f"Testing episode execution with {num_episodes} episodes...")
    
    results = []
    for i in range(num_episodes):
        logger.info(f"Running episode {i+1}/{num_episodes}")
        
        # Create a simple prompt
        prompt = [{"role": "user", "content": f"Test prompt for episode {i+1}"}]
        
        # Run generate to execute a complete episode
        result = env.generate([prompt], mock_llm, mock_sp)
        logger.info(f"Generation result keys: {result.keys()}")
        
        # Get the session ID
        session_id = result["session_ids"][0]
        logger.info(f"Session ID: {session_id}")
        
        # Check if episode completed successfully
        if session_id in env.completed_episodes:
            episode_data = env.completed_episodes[session_id]
            logger.info(f"Episode completed with reward: {episode_data['reward']}")
            logger.info(f"Episode steps: {episode_data['steps']}")
            logger.info(f"LLM actions: {episode_data['llm_actions']}")
            results.append(episode_data)
        else:
            logger.warning(f"Episode not found in completed_episodes dictionary")
    
    return results

def test_reward_function(env, episode_results):
    """Test the reward function."""
    logger.info("Testing reward function...")
    
    if not episode_results:
        logger.warning("No episode results to test reward function with")
        return False
    
    # Get the reward function
    reward_funcs = env.get_rubric()
    logger.info(f"Got {len(reward_funcs)} reward functions")
    
    # Test each episode result
    for episode_data in episode_results:
        session_id = None
        for k, v in env.completed_episodes.items():
            if v == episode_data:
                session_id = k
                break
        
        if session_id:
            # Test the reward function
            rewards = reward_funcs[0](
                prompts=[{"session_id": session_id}],
                completions=[{"role": "assistant", "content": "test"}]
            )
            logger.info(f"Computed rewards: {rewards}")
            
            # Check if reward matches
            if rewards and rewards[0] == episode_data["reward"]:
                logger.info("✓ Reward function returned correct reward")
            else:
                logger.warning(f"✗ Reward mismatch: func={rewards[0] if rewards else None}, episode={episode_data['reward']}")
    
    return True

def main():
    """Main test function."""
    args = parse_args()
    
    try:
        # Import here to allow setting up environment variables first
        from metarlgym.envs.textarena_env import TextArenaEnv
        
        # Create the environment
        logger.info(f"Creating TextArenaEnv with env_id={args.env_id}")
        env = TextArenaEnv(
            env_id=args.env_id,
            task_dataset_size=args.num_tasks,
            max_steps_per_episode=args.max_steps
        )
        
        # Test components
        if test_task_dataset(env):
            logger.info("✓ Task dataset creation successful")
        else:
            logger.error("✗ Task dataset creation failed")
            return 1
        
        # Create mock LLM and sampling params
        mock_llm = MockLLM()
        mock_sp = MockSamplingParams()
        
        # Test episode execution
        episode_results = test_episode_execution(env, mock_llm, mock_sp, args.num_episodes)
        if episode_results:
            logger.info("✓ Episode execution successful")
        else:
            logger.error("✗ Episode execution failed")
            return 1
        
        # Test reward function
        if test_reward_function(env, episode_results):
            logger.info("✓ Reward function test successful")
        else:
            logger.error("✗ Reward function test failed")
            return 1
        
        logger.info("All tests completed successfully!")
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you've installed the required packages")
        return 1
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 