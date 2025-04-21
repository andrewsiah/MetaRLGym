# An env where generate() involves doing multiple steps()

from typing import Any, Dict, List, Sequence, Union, Optional, Callable
import logging
import random
import uuid
import os
from datasets import Dataset
import numpy as np
import textarena as ta
from vllm import LLM, SamplingParams
from metarlgym.agents.directoutput.direct_output_agent import DirectOutputAgent
from trl.trainer.grpo_trainer import RewardFunc
from metarlgym.agents.base import Agent

from metarlgym.envs.environment import Environment


class MultistepEnv(Environment):
    """
    Base class for multi-step environments that allows for:
    1. Multiple reasoning steps within a single generate call
    2. Support for interactive environments (like TextArena games)
    3. Flexible action space for different types of multi-step tasks

    This class serves as a bridge between MetaRLGym's Environment API and 
    TextArena's Env API, allowing for standardized environment development.
    """
    
    def __init__(
        self,
        env_id: str,
        task_dataset_size: int = 1000,
        system_prompt: str = "",
        max_steps_per_episode: int = 5,
        observation_key: str = "observation",
        seed: int = 42,
        tokenizer = None,
        episodes_per_trial: int = 1,
        free_shots: int = 0,
        **kwargs
    ):
        """Initialize MultistepEnv.
        
        Args:
            env_id: Environment ID 
            task_dataset_size: Number of task initializations to generate
            system_prompt: System prompt to prefix observations with
            max_steps_per_episode: Maximum steps per episode
            observation_key: Key in the observation dict to use as prompt
            seed: Random seed for task generation
            tokenizer: Tokenizer for decoding LLM responses
        """
        super().__init__(**kwargs)
        self.env_id = env_id
        self.task_dataset_size = task_dataset_size
        self.system_prompt = system_prompt
        self.max_steps_per_episode = max_steps_per_episode
        self.observation_key = observation_key
        self.seed = seed
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(f"metarlgym.envs.{self.__class__.__name__}")
        
        if self.tokenizer:
            self.logger.info(f"MultistepEnv initialized with tokenizer: {type(self.tokenizer).__name__}")
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Trial parameters: multiple episodes and free unscored shots
        if episodes_per_trial < 1:
            raise ValueError(f"episodes_per_trial must be >= 1 (got {episodes_per_trial})")
        if free_shots < 0 or free_shots >= episodes_per_trial:
            raise ValueError(
                f"free_shots ({free_shots}) must be >= 0 and < episodes_per_trial ({episodes_per_trial})"
            )
        self.episodes_per_trial = episodes_per_trial
        self.free_shots = free_shots
        
        # Environment state tracking
        self.active_envs = {}  # Maps session IDs to active environments
        self.active_states = {}  # Maps session IDs to current state info
        self.completed_episodes = {}  # Maps session IDs to completed episode data
        
        # Generate task dataset for initializations
        self._create_task_dataset()
        # Per-session agent states
        self.agent_states: Dict[str, Any] = {}
    
    def _create_task_dataset(self):
        """
        Create a dataset of task initializations.
        This should be implemented by subclasses to create specific task datasets.
        """
        self.logger.info(f"Generating {self.task_dataset_size} task initializations")
        
        # Default implementation creates an empty dataset
        self.task_dataset = {"prompt": [], "solution": []}
        self.eval_task_dataset = {"prompt": [], "solution": []}
        
        self.logger.info(f"Created task dataset with 0 samples. Override _create_task_dataset in subclass.")
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        """Return the task dataset for training."""
        if isinstance(self.task_dataset, Dataset):
            return self.task_dataset
        return Dataset.from_dict(self.task_dataset)
    
    def get_eval_dataset(self, **kwargs: Any) -> Dataset:
        """Return the evaluation task dataset."""
        if isinstance(self.eval_task_dataset, Dataset):
            return self.eval_task_dataset
        return Dataset.from_dict(self.eval_task_dataset)
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return reward functions for training.
        
        Note on reward calculation pattern:
        - MultistepEnv environments calculate rewards internally during episode execution
          (either in _step_episode or _calculate_reward methods)
        - These rewards are stored in self.completed_episodes with session_id as key
        - The reward function returned by get_rubric() simply retrieves these pre-calculated 
          rewards based on session_id from completed_episodes
        - GRPOEnvTrainer expects and calls this reward function in _generate_and_score_completions
        
        This separation of concerns keeps environment-specific reward logic in the environment
        while providing a standardized interface for trainers.
        """
        # Define a reward function that retrieves pre-computed rewards from completed episodes
        def reward_func(prompts=None, completions=None, **kwargs):
            """Retrieves pre-calculated rewards from completed episodes dictionary.
            
            This function doesn't calculate rewards directly - it just looks up rewards
            that were already calculated during episode execution in the environment.
            """
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
                    self.logger.warning(f"No completed episode found for session {session_id}. Returning 0.0 reward.")
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
    
    def _initialize_episode(self, session_id, task_info):
        """
        Initialize an episode with the given session ID and task info.
        This method should be implemented by subclasses to set up the environment state.
        
        Args:
            session_id: ID of the session
            task_info: Information about the task to initialize
            
        Returns:
            Dict containing the initial state information
        """
        self.logger.info(f"[{session_id}] Initializing episode")
        
        # Default implementation just returns the task info
        initial_state = {
            "task_id": task_info.get("task_id", 0),
            "observation": task_info.get("content", ""),
            "solution": task_info.get("solution", None),
            "done": False,
            "steps": 0,
        }
        
        return initial_state
    
    def _step_episode(self, session_id, state, llm_action):
        """
        Take a step in the episode based on the current state and LLM action.
        This method should be implemented by subclasses to perform the environment transition.
        
        Args:
            session_id: ID of the session
            state: Current state of the episode
            llm_action: Action taken by the LLM
            
        Returns:
            Tuple containing (next_state, reward, done, info)
        """
        self.logger.info(f"[{session_id}] Taking step with action: {llm_action}")
        
        # Default implementation just increments the step count and checks for completion
        next_state = state.copy()
        next_state["steps"] += 1
        done = next_state["steps"] >= self.max_steps_per_episode
        
        # Simple reward: 1.0 if correct solution, 0.0 otherwise
        reward = 0.0
        info = {"action": llm_action}
        
        return next_state, reward, done, info
    
    def _format_prompt(self, state, step):
        """
        Format the current state as a prompt for the LLM.
        This method should be implemented by subclasses to create task-specific prompts.
        
        Args:
            state: Current state of the episode
            step: Current step count
            
        Returns:
            Formatted prompt for the LLM
        """
        # Default implementation just returns the observation
        observation = state["observation"]
        if isinstance(observation, str):
            prompt_text = observation
        else:
            prompt_text = observation.get(self.observation_key, str(observation))
        
        # Add step information to the prompt
        prompt_text += f"\n\nStep {step+1}/{self.max_steps_per_episode}"
        
        return prompt_text
    
    def _process_llm_response(self, llm_response, tokenizer=None):
        """
        Process the raw LLM response into a structured action.
        This method should be implemented by subclasses for task-specific processing.
        
        Args:
            llm_response: Raw response from the LLM
            tokenizer: Optional tokenizer for decoding
            
        Returns:
            Processed LLM action
        """
        # Default implementation just returns the raw response
        return llm_response
    
    def _calculate_reward(self, state, llm_actions, final_action):
        """
        Calculate the final reward for the episode.
        This method should be implemented by subclasses for task-specific reward calculation.
        
        Args:
            state: Final state of the episode
            llm_actions: List of all LLM actions taken during the episode
            final_action: Final action taken by the LLM
            
        Returns:
            Final reward value
        """
        # Default implementation: no reward
        return 0.0
    
    def _run_complete_episode(self, session_id, llm, sampling_params, agent: Agent):
        """Run a complete episode with the LLM using the provided agent.
        
        Args:
            session_id: ID of the session
            llm: LLM client for generating responses (can be used by env logic if needed)
            sampling_params: Sampling parameters
            agent: The agent instance responsible for generating actions.
            
        Returns:
            Dict containing episode data including the LLM's responses, 
            observations, actions, and final reward
        """
        # Get the active state
        state = self.active_states[session_id]
        task_id = state.get('task_id', 'N/A')
        self.logger.info(f"[{session_id}] Starting episode (Task ID: {task_id})")

        # Initialize episode data
        episode_data = {
            "observations": [],
            "llm_actions": [],
            "llm_responses": [],  # Raw text from LLM
            "response_ids": [],   # Token IDs from the responses
            "reward": 0.0,
            "done": False,
            "steps": 0,
            "task_id": task_id
        }
        
        # Run the episode until completion or max steps
        self.logger.debug(f"[{session_id}] Entering episode loop. Initial state: done={state['done']}, steps={state['steps']}")
        # Initialize conversation history and retrieve per-session agent state
        agent_state = self.agent_states.get(session_id)
        messages: List[Dict[str, Any]] = []

        
        while not state["done"] and state["steps"] < self.max_steps_per_episode:
            current_step = state["steps"]
            self.logger.info(f"[{session_id}] Step {current_step+1}/{self.max_steps_per_episode}")

            # Format observation as prompt for the LLM
            prompt_text = self._format_prompt(state, current_step)
            episode_data["observations"].append(prompt_text)
            # Append user prompt to conversation history
            messages.append({"role": "user", "content": prompt_text, "session_id": session_id})

            # Agent generates the next response
            action, agent_state = agent.get_action(messages, agent_state)
            self.agent_states[session_id] = agent_state
            token_ids = getattr(agent, "last_token_ids", None)

            # Record LLM responses and token IDs
            episode_data["llm_responses"].append(action)
            episode_data["response_ids"].append(token_ids)

            # Process the agent's action
            llm_action = self._process_llm_response(action, tokenizer=self.tokenizer)
            episode_data["llm_actions"].append(llm_action)
            self.logger.info(f"[{session_id}] Agent action: {llm_action}")

            # Append assistant message to history
            messages.append({"role": "assistant", "content": llm_action, "session_id": session_id})

            # Take a step in the environment
            next_state, reward, done, info = self._step_episode(session_id, state, llm_action)
            state = next_state
            state["done"] = done
            episode_data["steps"] += 1
            
            # Log step results
            self.logger.info(f"[{session_id}] Step {current_step+1} result: reward={reward}, done={done}")
            if info:
                self.logger.debug(f"[{session_id}] Step info: {info}")

            if done:
                self.logger.info(f"[{session_id}] Episode ended at step {state['steps']}.")
                break

        # Log reason for loop exit
        if state["done"]:
             self.logger.info(f"[{session_id}] Episode loop finished because done=True.")
        elif state["steps"] >= self.max_steps_per_episode:
             self.logger.warning(f"[{session_id}] Episode loop finished because max steps ({self.max_steps_per_episode}) reached.")
        else:
             self.logger.warning(f"[{session_id}] Episode loop finished unexpectedly (done={state['done']}, steps={state['steps']}).")

        # Calculate final reward
        final_action = episode_data["llm_actions"][-1] if episode_data["llm_actions"] else None
        reward = self._calculate_reward(state, episode_data["llm_actions"], final_action)
        episode_data["reward"] = reward
        self.logger.info(f"[{session_id}] Final reward: {reward}")

        # Set done flag
        episode_data["done"] = state["done"]
        
        # Log episode summary
        self.logger.info(f"[{session_id}] Episode summary:")
        self.logger.info(f"  - Task ID: {task_id}")
        self.logger.info(f"  - Steps completed: {state['steps']}/{self.max_steps_per_episode}")
        self.logger.info(f"  - Done: {state['done']}")
        self.logger.info(f"  - Final reward: {episode_data['reward']:.4f}")
        self.logger.info(f"  - Actions taken: {len(episode_data['llm_actions'])}")
        
        # Create a more detailed summary file for debugging
        try:
            # Get the log directory if available
            import inspect
            log_dir = None
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_dir = os.path.dirname(handler.baseFilename)
                    break
            
            if log_dir:
                # Create a detailed episode summary file
                summary_path = os.path.join(log_dir, f"episode_{session_id}.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"=== Episode Summary for {session_id} ===\n\n")
                    f.write(f"Task ID: {task_id}\n")
                    f.write(f"Environment: {self.env_id}\n")
                    f.write(f"Steps: {state['steps']}/{self.max_steps_per_episode}\n")
                    f.write(f"Done: {state['done']}\n")
                    f.write(f"Final reward: {episode_data['reward']:.4f}\n\n")
                    
                    # Write detailed step information
                    for i in range(len(episode_data['observations'])):
                        f.write(f"\n--- Step {i+1} ---\n")
                        f.write(f"Observation:\n{episode_data['observations'][i]}\n\n")
                        f.write(f"LLM Response:\n{episode_data['llm_responses'][i]}\n\n")
                        f.write(f"Processed Action:\n{episode_data['llm_actions'][i]}\n\n")
                
                self.logger.info(f"[{session_id}] Detailed episode log written to: {summary_path}")
        except Exception as e:
            self.logger.warning(f"[{session_id}] Failed to write detailed episode log: {e}")

        # Clean up
        del self.active_states[session_id]
        
        return episode_data
    
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        agent: Agent,
        **kwargs: Any
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        """Generate complete episodes using LLM and process them for GRPO training.
        
        This method:
        1. Initializes new environments for each prompt
        2. Runs complete episodes with the LLM
        3. Stores episode data including rewards
        4. Returns final completions and tracking information
        """
        self.logger.info(f"Received {len(prompts)} prompts for generation.")
        # List to store states and session IDs
        states = []
        session_ids = []
        
        # Process each prompt
        for prompt_idx, prompt_message in enumerate(prompts):
            # Initialize a new environment
            session_id = str(uuid.uuid4())
            session_ids.append(session_id)
            self.logger.debug(f"[{session_id}] Initializing environment for prompt {prompt_idx+1}/{len(prompts)}.")
            
            # Extract task information from the prompt
            task_info = {}
            if isinstance(prompt_message, list) and prompt_message:
                if isinstance(prompt_message[0], dict):
                    if "state" in prompt_message[0]:
                        task_info = prompt_message[0]["state"]
                    elif "content" in prompt_message[0]:
                        task_info = {"content": prompt_message[0]["content"]}
            
            if not task_info:
                # No task info, use a random sample from the task dataset
                task_id = random.randint(0, len(self.task_dataset["prompt"]) - 1)
                task_info = {
                    "task_id": task_id,
                    "content": self.task_dataset["prompt"][task_id][0]["content"] if self.task_dataset["prompt"] else "",
                    "solution": self.task_dataset["solution"][task_id] if "solution" in self.task_dataset else None
                }
            
            # Initialize the episode
            initial_state = self._initialize_episode(session_id, task_info)
            self.active_states[session_id] = initial_state
            
            # Format the initial prompt
            prompt_text = self._format_prompt(initial_state, 0)
            if self.system_prompt:
                formatted_prompt = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_text, "session_id": session_id}
                ]
            else:
                formatted_prompt = [{"role": "user", "content": prompt_text, "session_id": session_id}]
            
            # Create state for this prompt, including task_info for re-initialization
            states.append({
                "messages": formatted_prompt,
                "prompt_ids": [],
                "completion_ids": [],
                "completion_mask": [],
                "session_id": session_id,
                "task_info": task_info
            })
        
        # Initialize per-session agent states
        self.agent_states = {sid: None for sid in session_ids}
        # Run complete trials (multiple episodes) for each session
        for idx, session_id in enumerate(session_ids):
            task_info = states[idx].get("task_info", {})
            self.logger.info(
                f"[{session_id}] Starting trial {idx+1}/{len(prompts)} "
                f"with {self.episodes_per_trial} episodes (free_shots={self.free_shots})"
            )
            episodes = []
            for ep_idx in range(self.episodes_per_trial):
                if ep_idx > 0:
                    # re-initialize for next episode
                    initial_state = self._initialize_episode(session_id, task_info)
                    self.active_states[session_id] = initial_state
                # run the complete episode
                ep_data = self._run_complete_episode(session_id, llm, sampling_params, agent)
                # zero-out free shots rewards
                if ep_idx < self.free_shots:
                    ep_data["reward"] = 0.0
                episodes.append(ep_data)
            # final episode determines trainer reward
            final_ep = episodes[-1]
            self.completed_episodes[session_id] = final_ep
            # record all episodes
            states[idx]["all_episodes"] = episodes
            # use final response as completion
            if final_ep.get("llm_responses"):
                final_response = final_ep["llm_responses"][-1]
                final_ids = final_ep.get("response_ids", [])[-1]
                states[idx]["messages"].append({"role": "assistant", "content": final_response})
                states[idx]["completion_ids"] = final_ids
                states[idx]["completion_mask"] = [1] * len(final_ids)
            else:
                self.logger.warning(
                    f"No LLM responses in final episode for session {session_id}"
                )
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
        llm_model: Any,
        num_episodes: int = 10,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate an LLM using the multi-step environment.
        
        Args:
            llm_model: LLM model to evaluate (can be name or model object)
            num_episodes: Number of episodes to run
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create LLM interface based on input type
        if isinstance(llm_model, str):
            from vllm import LLM
            try:
                llm = LLM(model=llm_model)
            except Exception as e:
                self.logger.error(f"Failed to load model '{llm_model}': {e}")
                return {"error": 1.0}
        else:
            llm = llm_model
        
        # Sampling parameters for evaluation
        sampling_params = SamplingParams(temperature=0.1, max_tokens=512)
        
        # Metrics to track
        metrics = {
            "success_rate": 0.0,
            "average_reward": 0.0,
            "steps_per_episode": 0.0
        }
        
        # Track successes and rewards
        successes = 0
        total_reward = 0.0
        total_steps = 0
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            if verbose:
                self.logger.info(f"Running evaluation episode {episode+1}/{num_episodes}")
            
            # Select a random task
            task_id = random.randint(0, len(self.task_dataset["prompt"]) - 1)
            prompt = [self.task_dataset["prompt"][task_id]]
            
            # Run generate
            result = self.generate(
                prompts=prompt,
                llm=llm,
                sampling_params=sampling_params
            )
            
            # Get the session ID
            session_id = result["session_ids"][0]
            
            # Get the episode data
            episode_data = self.completed_episodes[session_id]
            
            # Track metrics
            reward = episode_data["reward"]
            steps = episode_data["steps"]
            
            total_reward += reward
            total_steps += steps
            
            if reward > 0:
                successes += 1
            
            if verbose:
                self.logger.info(f"Episode {episode+1} reward: {reward}")
            
            # Clean up
            del self.completed_episodes[session_id]
        
        # Calculate metrics
        metrics["success_rate"] = successes / num_episodes
        metrics["average_reward"] = total_reward / num_episodes
        metrics["steps_per_episode"] = total_steps / num_episodes
        
        return metrics

