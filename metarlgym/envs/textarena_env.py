from typing import Any, Dict, List, Sequence, Union, Optional, Callable
import logging
import random
import uuid
from datasets import Dataset
import numpy as np
import textarena as ta
from vllm import LLM, SamplingParams  # type: ignore
from trl.trainer.grpo_trainer import RewardFunc
import re

from metarlgym.envs.environment import Environment


def extract_marked_action(llm_response, markers=None, logger=None, session_id=None):
    """Extract action from consistent markers in LLM response.
    
    Args:
        llm_response (str): Raw LLM response text
        markers (list): Optional list of marker patterns to try
        logger: Optional logger for detailed extraction logs
        session_id: Optional session ID for logging context
        
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
    
    # Log attempt to extract action if logger provided
    if logger and session_id:
        logger.debug(f"[{session_id}] Attempting to extract marked action from response of length {len(llm_response)}")
    
    for pattern, extractor in markers:
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            extracted = extractor(match).strip()
            
            # Add square brackets if they're not already present
            if not re.match(r'^\[.*\]$', extracted):
                extracted = f"[{extracted}]"
            
            # Log successful extraction if logger provided
            if logger and session_id:
                logger.debug(f"[{session_id}] Successfully extracted action using pattern '{pattern}': {extracted}")
                
            return extracted
    
    # If no markers found, return the original response and log if enabled
    if logger and session_id:
        logger.debug(f"[{session_id}] No markers found in response, using raw response")
    
    return llm_response


class TextArenaEnv(Environment):
    """Adapter for TextArena environments to work with GRPO training in an online RL manner.
    
    This class allows TextArena environments to be used with GRPO training by:
    1. Maintaining active environment instances during training
    2. Providing a live interface between GRPO and TextArena
    3. Computing rewards from real-time interactions
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
        """Initialize TextArenaEnv.
        
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
        super().__init__(**kwargs)
        self.env_id = env_id
        self.task_dataset_size = task_dataset_size
        self.system_prompt = system_prompt
        self.max_steps_per_episode = max_steps_per_episode
        self.observation_key = observation_key
        self.num_players = num_players
        self.opponent_policy = opponent_policy
        self.tokenizer = tokenizer
        self.seed = seed
        self.logger = logging.getLogger(f"metarlgym.envs.{self.__class__.__name__}")
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Environment state tracking
        self.active_envs = {}  # Maps session IDs to active environments
        self.active_states = {}  # Maps session IDs to current state info
        self.completed_episodes = {}  # Maps session IDs to completed episode data
        
        # Default opponent policy (simple random agent)
        if self.opponent_policy is None:
            self.opponent_policy = self._default_opponent_policy
        
        # Generate task dataset for initializations
        self._create_task_dataset()
    
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
        
        # Create the task dataset
        self.task_dataset = Dataset.from_dict({
            "prompt": task_prompts,
            "state": task_states,
        })
        
        # Split into train and eval
        dataset_dict = self.task_dataset.train_test_split(test_size=0.1)
        self.task_dataset = dataset_dict["train"]
        self.eval_task_dataset = dataset_dict["test"]
        
        self.logger.info(f"Created task dataset with {len(self.task_dataset)} samples")
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        """Return the task dataset for training."""
        return self.task_dataset
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset:
        """Return the evaluation task dataset."""
        return self.eval_task_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return reward functions for training."""
        # Define a reward function that computes rewards from completed episodes
        def reward_func(prompts=None, completions=None, **kwargs):
            """Online reward function for TextArena environments."""
            rewards = []
            
            for prompt_messages, completion in zip(prompts, completions):
                # Get the session ID from the prompt messages
                session_id = None
                for message in prompt_messages:
                    if isinstance(message, dict) and "session_id" in message:
                        session_id = message["session_id"]
                        break
                
                if session_id is None or session_id not in self.completed_episodes:
                    # Session not found or not completed
                    rewards.append(0.0)
                    continue
                
                # Get the completed episode data
                episode_data = self.completed_episodes[session_id]
                reward = episode_data.get("reward", 0.0)
                
                # Clean up completed episode data to avoid memory leaks
                del self.completed_episodes[session_id]
                
                # Append the reward
                rewards.append(float(reward))
            
            return rewards
        
        return [reward_func]
    
    def _run_complete_episode(self, session_id, llm, sampling_params):
        """Run a complete episode with the LLM making decisions for player 0.
        
        Args:
            session_id: ID of the session
            llm: LLM client for generating responses
            sampling_params: Sampling parameters
            
        Returns:
            Dict containing episode data including the LLM's responses, 
            observations, actions, and final reward
        """
        # Get the active environment and state
        env = self.active_envs[session_id]
        state = self.active_states[session_id]
        self.logger.info(f"[{session_id}] Starting episode (Task ID: {state.get('task_id', 'N/A')})")

        # Initialize episode data
        episode_data = {
            "observations": [],
            "llm_actions": [],
            "opponent_actions": [],
            "llm_responses": [],  # Raw text from LLM
            "response_ids": [],   # Token IDs from the responses
            "reward": 0.0,
            "done": False,
            "steps": 0
        }
        
        # Run the episode until completion or max steps
        self.logger.debug(f"[{session_id}] Entering episode loop. Initial state: player={state['player_id']}, done={state['done']}, steps={state['steps']}")
        while not state["done"] and state["steps"] < self.max_steps_per_episode:
            current_step = state["steps"]
            current_player = state["player_id"]
            self.logger.debug(f"[{session_id}] Step {current_step+1}/{self.max_steps_per_episode}: Player {current_player}'s turn.")

            # If not player 0's turn, use opponent policy
            if state["player_id"] != 0:
                self.logger.debug(f"[{session_id}] Using opponent policy for player {current_player}.")
                # Use opponent policy to get action
                opponent_action = self.opponent_policy(state["observation"])
                episode_data["opponent_actions"].append(opponent_action)
                self.logger.debug(f"[{session_id}] Opponent action: {opponent_action}")

                # Take step with opponent action
                done, info = env.step(action=opponent_action)
                self.logger.debug(f"[{session_id}] env.step result: done={done}, info={info}")
                state["done"] = done
                state["steps"] += 1

                if done:
                    self.logger.info(f"[{session_id}] Episode ended by opponent step at step {state['steps']}.")
                    break

                # Get next observation
                player_id, observation = env.get_observation()
                self.logger.debug(f"[{session_id}] Next observation for player {player_id}.") # Add observation logging if needed: : {observation}")
                state["player_id"] = player_id
                state["observation"] = observation

                # If still not player 0's turn, continue with opponent moves
                continue

            # It's player 0's turn, use the LLM to generate a response
            self.logger.debug(f"[{session_id}] Player 0's turn. Preparing LLM prompt.")
            observation = state["observation"]
            episode_data["observations"].append(observation)

            # Format observation as prompt for the LLM
            prompt_text = observation if isinstance(observation, str) else observation.get(self.observation_key, str(observation))
            if self.system_prompt:
                formatted_text = f"system: {self.system_prompt}\nuser: {prompt_text}"
            else:
                formatted_text = f"user: {prompt_text}"
            
            # Add instruction for formatting the final answer
            formatted_text += "\n\nAfter providing your reasoning, format your final action as \\boxed{your_action} where your_action is your chosen move in the required format."
            
            # Get LLM action
            self.logger.debug(f"[{session_id}] Generating LLM response...")
            custom_sp = sampling_params.clone()
            completion_ids_list = llm.generate( # Renamed to avoid conflict
                prompts=[formatted_text],
                n=1,
                repetition_penalty=custom_sp.repetition_penalty,
                temperature=custom_sp.temperature, 
                top_p=custom_sp.top_p,
                top_k=custom_sp.top_k if custom_sp.top_k is not None else -1,
                min_p=custom_sp.min_p if custom_sp.min_p is not None else 0.0,
                max_tokens=custom_sp.max_tokens,
                guided_decoding_regex=getattr(custom_sp, 'guided_decoding_regex', None)
            )
            completion_ids = completion_ids_list[0] # VLLMClient returns list[list[int]], take the first list as n=1

            # Decode completion
            llm_response = "<decoding_error>"
            try:
                if self.tokenizer is not None:
                    llm_response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                else:
                    llm_response = f"Token IDs: {completion_ids}" # Fallback if no tokenizer
                self.logger.debug(f"[{session_id}] LLM raw response: {llm_response}")
            except Exception as e:
                 self.logger.error(f"[{session_id}] Error decoding LLM response: {e}")


            # Store the response
            episode_data["llm_responses"].append(llm_response)
            episode_data["response_ids"].append(completion_ids)
            
            # Use the marker extraction to get the action from the response with improved logging
            llm_action = extract_marked_action(llm_response, logger=self.logger, session_id=session_id)
                
            episode_data["llm_actions"].append(llm_action)
            self.logger.debug(f"[{session_id}] Player 0 action (LLM response): {llm_action}")

            # Take a step in the environment
            done, info = env.step(action=llm_action)
            self.logger.debug(f"[{session_id}] env.step result: done={done}, info={info}")
            state["done"] = done
            state["steps"] += 1

            if done:
                self.logger.info(f"[{session_id}] Episode ended by LLM step at step {state['steps']}.")
                break

            # Get next observation and player
            player_id, observation = env.get_observation()
            self.logger.debug(f"[{session_id}] Next observation for player {player_id}.") # Add observation logging if needed: : {observation}")
            state["player_id"] = player_id
            state["observation"] = observation

        # Log reason for loop exit
        if state["done"]:
             self.logger.info(f"[{session_id}] Episode loop finished because done=True.")
        elif state["steps"] >= self.max_steps_per_episode:
             self.logger.warning(f"[{session_id}] Episode loop finished because max steps ({self.max_steps_per_episode}) reached.")
        else:
             self.logger.warning(f"[{session_id}] Episode loop finished unexpectedly (done={state['done']}, steps={state['steps']}).")


        # Episode is done, get final reward
        if state["done"]:
            try:
                self.logger.info(f"[{session_id}] Closing environment and getting rewards.")
                episode_rewards = env.close()
                # Get reward for player 0 (the one we're training)
                if isinstance(episode_rewards, dict):
                    reward = episode_rewards.get(0, 0)
                elif isinstance(episode_rewards, (int, float)): # Handle case where reward is just a number
                    reward = float(episode_rewards)
                else:
                    self.logger.warning(f"[{session_id}] Unexpected reward type: {type(episode_rewards)}. Setting reward to 0.")
                    reward = 0.0
                episode_data["reward"] = reward
                self.logger.info(f"[{session_id}] Final reward for player 0: {reward}")
            except Exception as e:
                self.logger.warning(f"[{session_id}] Error closing env or getting rewards: {e}")
                episode_data["reward"] = 0.0 # Ensure reward is set even on error
        else:
            # Episode reached max steps but didn't complete
            self.logger.warning(f"[{session_id}] Episode reached max steps ({self.max_steps_per_episode}) without completion flag. Attempting to close env.")
            try:
                episode_rewards = env.close()
                if isinstance(episode_rewards, dict):
                    reward = episode_rewards.get(0, 0)
                elif isinstance(episode_rewards, (int, float)):
                    reward = float(episode_rewards)
                else:
                    self.logger.warning(f"[{session_id}] Unexpected reward type at max steps: {type(episode_rewards)}. Setting reward to 0.")
                    reward = 0.0
                episode_data["reward"] = reward
                self.logger.info(f"[{session_id}] Reward at max steps for player 0: {reward}")
            except Exception as e:
                self.logger.warning(f"[{session_id}] Error closing env or getting rewards at max steps: {e}")
                episode_data["reward"] = 0.0 # Ensure reward is set even on error

        # Set done flag
        episode_data["done"] = state["done"]
        self.logger.info(f"[{session_id}] Episode finished. Final state: done={state['done']}, steps={state['steps']}, reward={episode_data['reward']:.2f}")

        # Clean up
        del self.active_envs[session_id]
        del self.active_states[session_id]
        
        return episode_data
    
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        """Generate complete episodes using LLM and process them for GRPO training.
        
        This method:
        1. Initializes new TextArena environments for each prompt
        2. Runs complete episodes with the LLM acting as player 0
        3. Stores episode data including rewards
        4. Returns final completions and tracking information
        """
        self.logger.info(f"Received {len(prompts)} prompts for generation.")
        # List to store states and session IDs
        states = []
        session_ids = []
        
        # Process each prompt
        for prompt_idx, prompt_message in enumerate(prompts):
            # Initialize a new environment regardless of whether it's a new prompt
            # This ensures we get fresh episodes each time
            session_id = str(uuid.uuid4())
            session_ids.append(session_id)
            self.logger.debug(f"[{session_id}] Initializing environment for prompt {prompt_idx+1}/{len(prompts)}.")
            
            # Get the state info from the dataset sample
            state_info = None
            if "state" in prompt_message[0]:
                state_info = prompt_message[0]["state"]
            
            if state_info is None:
                # No state info, use a random sample from the task dataset
                task_id = random.randint(0, len(self.task_dataset) - 1)
                state_info = self.task_dataset[task_id]["state"]
            
            # Initialize a new environment
            env = ta.make(env_id=self.env_id)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env.reset(num_players=self.num_players)
            
            # Get the first observation (for player 0)
            player_id, observation = env.get_observation()
            
            # Store active environment
            self.active_envs[session_id] = env
            self.active_states[session_id] = {
                "task_id": state_info.get("task_id", 0),
                "player_id": player_id,
                "observation": observation,
                "done": False,
                "steps": 0,
            }
            
            # Format the observation as a prompt
            prompt_text = observation if isinstance(observation, str) else observation.get(self.observation_key, str(observation))
            if self.system_prompt:
                formatted_prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_text, "session_id": session_id}
                ]
            else:
                formatted_prompt = [{"role": "user", "content": prompt_text, "session_id": session_id}]
            
            # Create state for this prompt
            states.append({
                "messages": formatted_prompt,
                "prompt_ids": [],
                "completion_ids": [],
                "completion_mask": [],
                "session_id": session_id
            })
        
        # Run complete episodes for each session
        for idx, session_id in enumerate(session_ids):
            # Run the complete episode
            self.logger.info(f"[{session_id}] Starting episode run for prompt {idx+1}/{len(prompts)}.")
            episode_data = self._run_complete_episode(session_id, llm, sampling_params)
            
            # Store the episode data for reward computation
            self.completed_episodes[session_id] = episode_data
            
            # Use the final LLM response as the completion
            if episode_data["llm_responses"]:
                # Get the last LLM response and token IDs
                final_response = episode_data["llm_responses"][-1]
                final_ids = episode_data["response_ids"][-1]
                
                # Extract the marked action from the response
                final_action = extract_marked_action(final_response)
                
                # Update state with completion info
                states[idx]["messages"].append({"role": "assistant", "content": final_action})
                states[idx]["completion_ids"] = final_ids
                states[idx]["completion_mask"] = [1] * len(final_ids)
            else:
                # No LLM responses in the episode (should not happen normally)
                self.logger.warning(f"No LLM responses in episode for session {session_id}")
                # Create an empty response
                states[idx]["messages"].append({"role": "assistant", "content": ""})
                states[idx]["completion_ids"] = []
                states[idx]["completion_mask"] = []
        
        # Return the output in the format expected by GRPO
        output = {
            "ids": [states[i]["completion_ids"] for i in range(len(states))],
            "messages": [states[i]["messages"][-1:] for i in range(len(states))],
            "mask": [states[i]["completion_mask"] for i in range(len(states))],
            "session_ids": [states[i]["session_id"] for i in range(len(states))]
        }
        self.logger.info(f"Generation complete. Returning {len(output['ids'])} completions.")
        return output
    
    def evaluate_llm(
        self,
        llm_model: str,
        num_episodes: int = 10,
        opponent_policy: Optional[Callable] = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate an LLM using the TextArena environment.
        
        Args:
            llm_model: Name of the model to evaluate
            num_episodes: Number of episodes to run
            opponent_policy: Policy function for opponents (overrides default)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with evaluation metrics
        """
        if opponent_policy is None:
            opponent_policy = self.opponent_policy
        
        # Create agent for evaluation
        try:
            if "gpt" in llm_model.lower():
                agent = ta.agents.OpenRouterAgent(model_name=llm_model)
            elif "claude" in llm_model.lower() and "/" not in llm_model:
                agent = ta.agents.OpenRouterAgent(model_name=f"anthropic/{llm_model}")
            elif ":" in llm_model:  # Looks like a Hugging Face model with version
                agent = ta.agents.OpenRouterAgent(model_name=llm_model)
            else:
                agent = ta.agents.OpenRouterAgent(model_name=llm_model)
        except Exception as e:
            self.logger.warning(f"Error creating OpenRouterAgent: {e}")
            # Fallback to a simple LLM agent using HF model directly
            from textarena.agents.basic_agents import HuggingFaceAgent
            agent = HuggingFaceAgent(model_name=llm_model)
        
        # Metrics to track
        metrics = {
            "win_rate": 0.0,
            "average_reward": 0.0,
            "steps_per_episode": 0.0
        }
        
        # Track wins and rewards
        wins = 0
        total_reward = 0.0
        total_steps = 0
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            if verbose:
                self.logger.info(f"Running evaluation episode {episode+1}/{num_episodes}")
            
            # Initialize environment
            env = ta.make(env_id=self.env_id)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            if verbose:
                env = ta.wrappers.SimpleRenderWrapper(
                    env=env,
                    player_names={0: "Model", 1: "Opponent"}
                )
            
            # Reset the environment
            env.reset(num_players=self.num_players)
            
            # Run the episode
            done = False
            steps = 0
            while not done and steps < self.max_steps_per_episode:
                player_id, observation = env.get_observation()
                
                # Use agent for player 0, opponent policy for others
                if player_id == 0:
                    action = agent(observation)
                else:
                    action = opponent_policy(observation)
                
                # Take a step
                done, info = env.step(action=action)
                steps += 1
            
            # Get rewards
            rewards = env.close()
            
            # Track metrics
            evaluated_agent_reward = rewards.get(0, 0) if isinstance(rewards, dict) else 0
            total_reward += evaluated_agent_reward
            total_steps += steps
            
            # Track wins (if rewards > 0 means a win in most TextArena environments)
            if evaluated_agent_reward > 0:
                wins += 1
            
            if verbose:
                self.logger.info(f"Episode {episode+1} reward: {evaluated_agent_reward}")
        
        # Calculate metrics
        metrics["win_rate"] = wins / num_episodes
        metrics["average_reward"] = total_reward / num_episodes
        metrics["steps_per_episode"] = total_steps / num_episodes
        
        return metrics
