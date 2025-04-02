from typing import Any, Dict, List, Sequence, Union, Optional, Callable
import logging
import random
import uuid
from datasets import Dataset
import numpy as np
import textarena as ta
from vllm import LLM, SamplingParams
from trl.trainer.grpo_trainer import RewardFunc
import re

from metarlgym.envs.multistep_env import MultistepEnv


def extract_marked_action(llm_response, markers=None):
    """Extract action from consistent markers in LLM response.
    
    Args:
        llm_response (str): Raw LLM response text
        markers (list): Optional list of marker patterns to try
        
    Returns:
        str: Extracted action or original response if no markers found
    """
    if markers is None:
        markers = [
            (r'\\boxed\{(.*?)\}', lambda m: m.group(1)),  # \boxed{action}
            (r'<answer>(.*?)</answer>', lambda m: m.group(1)),  # <answer>action</answer>
            (r'\[FINAL ANSWER: (.*?)\]', lambda m: m.group(1)),  # [FINAL ANSWER: action]
            (r'Final Answer:\s*(.*?)(?:\n|$)', lambda m: m.group(1)),  # Final Answer: action
            (r'Therefore,\s*(?:the\s*)?(?:move|answer|action)\s*is:?\s*\[(.*?)\]', lambda m: f"[{m.group(1)}]")  # Therefore, the move is: [A0 B0]
        ]
    
    for pattern, extractor in markers:
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            extracted = extractor(match).strip()
            
            # Add square brackets if they're not already present
            if not re.match(r'^\[.*\]$', extracted):
                extracted = f"[{extracted}]"
                
            return extracted
    
    # If no markers found, return the original response
    return llm_response


class TextArenaMultistepEnv(MultistepEnv):
    """
    MultistepEnv adapter for TextArena environments.
    
    This class integrates TextArena environments with the MultistepEnv framework,
    allowing for standardized multi-step interactions between an LLM agent and
    TextArena game environments.
    """
    
    def __init__(
        self,
        env_id: str,
        task_dataset_size: int = 1000,
        system_prompt: str = "",
        max_steps_per_episode: int = 10,
        observation_key: str = "observation",
        num_players: int = 2,
        opponent_policy: Optional[Callable] = None,
        tokenizer=None,
        seed: int = 42,
        **kwargs
    ):
        """Initialize TextArenaMultistepEnv.
        
        Args:
            env_id: TextArena environment ID (e.g., "SpellingBee-v0")
            task_dataset_size: Number of task initializations to generate
            system_prompt: System prompt to prefix observations with
            max_steps_per_episode: Maximum steps per episode
            observation_key: Key in the observation dict to use as prompt
            num_players: Number of players in the environment
            opponent_policy: Function that takes observation and returns action for opponents
            tokenizer: Tokenizer to use
            seed: Random seed for task generation
        """
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.tokenizer = tokenizer
        
        # Initialize parent class
        super().__init__(
            env_id=env_id,
            task_dataset_size=task_dataset_size,
            system_prompt=system_prompt,
            max_steps_per_episode=max_steps_per_episode,
            observation_key=observation_key,
            seed=seed,
            **kwargs
        )
        
        # Default opponent policy (simple random agent)
        if self.opponent_policy is None:
            self.opponent_policy = self._default_opponent_policy
    
    def _default_opponent_policy(self, observation):
        """Default policy for opponent agents."""
        # Very simple policy that just extracts possible actions from the observation
        # and randomly selects one
        if isinstance(observation, str):
            # If the observation text contains options or choices, try to extract them
            if "options:" in observation.lower():
                options_part = observation.lower().split("options:")[1].split("\n")[0]
                options = [opt.strip() for opt in options_part.split(",")]
                return random.choice(options)
            return "I pass" # Default action if we can't extract options
        
        # If observation is a dict with options or actions key
        if isinstance(observation, dict):
            if "actions" in observation:
                actions = observation["actions"]
                if isinstance(actions, list) and actions:
                    return random.choice(actions)
            if "options" in observation:
                options = observation["options"]
                if isinstance(options, list) and options:
                    return random.choice(options)
        
        # Default action
        return "I pass"
    
    def _create_task_dataset(self):
        """Create a dataset of task initializations (not full episodes)."""
        self.logger.info(f"Generating {self.task_dataset_size} task initializations")
        
        # Lists to store initial states
        task_prompts = []
        task_states = []
        
        # Generate task initializations
        for task_id in range(self.task_dataset_size):
            if task_id % 100 == 0:
                self.logger.info(f"Generating task {task_id}/{self.task_dataset_size}")
            
            # Initialize environment
            try:
                env = ta.make(env_id=self.env_id)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                
                # Reset the environment to get initial state
                env.reset(num_players=self.num_players)
                
                # Get the first observation (for player 0)
                player_id, observation = env.get_observation()
                
                # Format observation as a prompt
                prompt = observation if isinstance(observation, str) else observation.get(self.observation_key, str(observation))
                if self.system_prompt:
                    formatted_prompt = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    formatted_prompt = [{"role": "user", "content": prompt}]
                
                # Store the initial state
                task_states.append({
                    "env_id": self.env_id,
                    "task_id": task_id,
                    "player_id": player_id,
                    "observation": observation,
                    "num_players": self.num_players,
                })
                task_prompts.append(formatted_prompt)
                
                # Close the environment
                env.close()
            except Exception as e:
                self.logger.error(f"Error initializing environment {self.env_id} for task {task_id}: {e}")
                # Skip this task
                continue
        
        # Check if we have any successful initializations
        if not task_prompts:
            self.logger.warning(f"Failed to initialize any environments. Using dummy data.")
            task_prompts = [[{"role": "user", "content": f"This is a dummy prompt for {self.env_id}"}]]
            task_states = [{
                "env_id": self.env_id,
                "task_id": 0,
                "player_id": 0,
                "observation": f"This is a dummy observation for {self.env_id}",
                "num_players": self.num_players,
            }]
        
        # Create the task dataset
        self.task_dataset = {"prompt": task_prompts, "state": task_states}
        
        # Split into train and eval if we have enough data
        if len(task_prompts) > 1:
            train_size = max(1, int(len(task_prompts) * 0.9))
            self.eval_task_dataset = {
                "prompt": task_prompts[train_size:],
                "state": task_states[train_size:]
            }
            self.task_dataset = {
                "prompt": task_prompts[:train_size],
                "state": task_states[:train_size]
            }
        else:
            self.eval_task_dataset = self.task_dataset
        
        self.logger.info(f"Created task dataset with {len(self.task_dataset['prompt'])} samples")
    
    def _initialize_episode(self, session_id, task_info):
        """Initialize a TextArena environment episode."""
        self.logger.info(f"[{session_id}] Initializing TextArena environment")
        
        # Initialize environment
        env = ta.make(env_id=self.env_id)
        env = ta.wrappers.LLMObservationWrapper(env=env)
        
        # Reset the environment
        env.reset(num_players=self.num_players)
        
        # Get the first observation (for player 0)
        player_id, observation = env.get_observation()
        
        # Store the environment
        self.active_envs[session_id] = env
        
        # Initialize state
        initial_state = {
            "task_id": task_info.get("task_id", 0),
            "player_id": player_id,
            "observation": observation,
            "env_id": self.env_id,
            "num_players": self.num_players,
            "steps": 0,
            "done": False,
            "actions": []
        }
        
        return initial_state
    
    def _format_prompt(self, state, step):
        """Format the current state as a prompt for the LLM."""
        # Get the observation
        observation = state["observation"]
        
        # Format observation as prompt
        if isinstance(observation, str):
            prompt_text = observation
        else:
            prompt_text = observation.get(self.observation_key, str(observation))
        
        # Add instructions to format the action
        prompt_text += "\n\nAfter providing your reasoning, format your final action as \\boxed{your_action} where your_action is your chosen move in the required format."
        
        return prompt_text
    
    def _process_llm_response(self, llm_response, tokenizer=None):
        """Process the raw LLM response into a structured action."""
        # Use the marker extraction to get the action from the response
        action = extract_marked_action(llm_response)
        return action
    
    def _step_episode(self, session_id, state, llm_action):
        """Take a step in the episode based on the LLM action."""
        self.logger.info(f"[{session_id}] Taking step with action: {llm_action}")
        
        # Get the active environment
        env = self.active_envs[session_id]
        
        # Create a copy of the state for updating
        next_state = state.copy()
        next_state["steps"] += 1
        next_state["actions"].append(llm_action)
        
        # Take step with LLM action
        done, info = env.step(action=llm_action)
        next_state["done"] = done
        
        # Check if the episode is done
        if done:
            self.logger.info(f"[{session_id}] Episode ended by LLM step at step {next_state['steps']}.")
            # Get rewards
            rewards = env.close()
            
            # Get reward for player 0 (the one we're training)
            if isinstance(rewards, dict):
                reward = rewards.get(0, 0)
            elif isinstance(rewards, (int, float)):
                reward = float(rewards)
            else:
                self.logger.warning(f"[{session_id}] Unexpected reward type: {type(rewards)}. Setting reward to 0.")
                reward = 0.0
                
            return next_state, reward, done, {"rewards": rewards, "info": info}
        
        # Handle the opponent's turn if needed
        while not next_state["done"] and next_state["steps"] < self.max_steps_per_episode:
            # Get next observation and player
            player_id, observation = env.get_observation()
            next_state["player_id"] = player_id
            next_state["observation"] = observation
            
            # If it's player 0's turn again, break the loop
            if player_id == 0:
                break
            
            # Use opponent policy to get action
            opponent_action = self.opponent_policy(observation)
            self.logger.info(f"[{session_id}] Opponent action: {opponent_action}")
            
            # Take step with opponent action
            done, info = env.step(action=opponent_action)
            next_state["done"] = done
            
            if done:
                self.logger.info(f"[{session_id}] Episode ended by opponent step at step {next_state['steps']}.")
                # Get rewards
                rewards = env.close()
                
                # Get reward for player 0 (the one we're training)
                if isinstance(rewards, dict):
                    reward = rewards.get(0, 0)
                elif isinstance(rewards, (int, float)):
                    reward = float(rewards)
                else:
                    self.logger.warning(f"[{session_id}] Unexpected reward type: {type(rewards)}. Setting reward to 0.")
                    reward = 0.0
                
                return next_state, reward, done, {"rewards": rewards, "info": info}
        
        # If we get here, the episode is still ongoing
        return next_state, 0.0, next_state["done"], {}
    
    def _calculate_reward(self, state, llm_actions, final_action):
        """Calculate the final reward for the episode."""
        # If the environment is not done, return 0
        if not state["done"]:
            self.logger.warning(f"Calculating reward for incomplete episode. Returning 0.")
            return 0.0
        
        # For completed episodes, the reward should have been calculated in _step_episode
        # But we can also calculate it here if needed
        session_id = next(iter([sid for sid, s in self.active_states.items() if s == state]), None)
        if session_id is not None and session_id in self.active_envs:
            env = self.active_envs[session_id]
            try:
                rewards = env.close()
                
                # Get reward for player 0 (the one we're training)
                if isinstance(rewards, dict):
                    return rewards.get(0, 0)
                elif isinstance(rewards, (int, float)):
                    return float(rewards)
            except Exception as e:
                self.logger.error(f"Error calculating reward: {e}")
        
        # Default reward is 0
        return 0.0
    
    def evaluate_llm(
        self,
        llm_model: Any,
        num_episodes: int = 10,
        opponent_policy: Optional[Callable] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate an LLM using the TextArena environment.
        
        Args:
            llm_model: LLM model to evaluate
            num_episodes: Number of episodes to run
            opponent_policy: Policy function for opponents (overrides default)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Override opponent policy if provided
        original_policy = self.opponent_policy
        if opponent_policy is not None:
            self.opponent_policy = opponent_policy
        
        # Run the parent class's evaluation method
        metrics = super().evaluate_llm(
            llm_model=llm_model,
            num_episodes=num_episodes,
            verbose=verbose
        )
        
        # Restore original opponent policy
        if opponent_policy is not None:
            self.opponent_policy = original_policy
        
        return metrics 