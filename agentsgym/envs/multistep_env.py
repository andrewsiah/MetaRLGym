# An env where run_trial() involves doing multiple steps()

from typing import Any, Dict, List, Sequence, Union, Optional, Callable
import logging
import random
import os
from datasets import Dataset
import numpy as np
import textarena as ta
from vllm import LLM, SamplingParams
from agentsgym.agents.directoutput.direct_output_agent import DirectOutputAgent
from agentsgym.agents.base import Agent
from transformers import PreTrainedTokenizerBase
import uuid

from agentsgym.envs.environment import Environment


class MultistepEnv(Environment):
    """
    Base class for multi-step environments that allows for:
    1. Multiple reasoning steps within a single run_trial call
    2. Support for interactive environments (like TextArena games)
    3. Flexible action space for different types of multi-step tasks

    This class serves as a bridge between AgentsGym's Environment API and 
    TextArena's Env API, allowing for standardized environment development.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        env_config: Dict[str, Any]
    ):
        """Initialize MultistepEnv.
        
        Args:
            tokenizer: Tokenizer for text processing within the environment.
            env_config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.env_config = env_config
        self.logger = logging.getLogger(f"agentsgym.envs.{self.__class__.__name__}")
        
        if self.tokenizer:
            self.logger.info(f"MultistepEnv initialized with tokenizer: {type(self.tokenizer).__name__}")
        else:
            self.logger.warning("MultistepEnv initialized without a tokenizer.")
        
        # Environment state tracking
        self.active_envs = {}  # Maps session IDs to active environments
        self.active_states = {}  # Maps session IDs to current state info
        self.completed_episodes = {}  # Maps session IDs to completed episode data
        
        # Per-session agent states
        self.agent_states: Dict[str, Any] = {}

        self.train_dataset = None
        self.eval_dataset = None

    
    def get_train_dataset(self, **kwargs: Any) -> Dataset:
        """Return the task dataset for training."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_train_dataset.")
    
    def get_eval_dataset(self) -> Dataset:
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_eval_dataset.")
    
    def _initialize_episode(self, session_id, task_info: Dict[str, Any]):
        """
        Initialize an episode with the given session ID and task info.
        This method should be implemented by subclasses to set up the environment state.
        
        Args:
            session_id: ID of the session
            task_info: Information about the task to initialize (Type Hint: Dict[str, Any])
            
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
    
    def step(self, session_id, state, llm_action):
        """
        Take a step in the episode based on the current state and LLM action.
        This method must be implemented by subclasses to perform the environment transition
        and return results in the expected format.
        
        Args:
            session_id: ID of the session
            state: Current state of the episode
            llm_action: Action taken by the LLM
            
        Returns:
            Tuple containing (next_state, reward, done, info)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'step' method.")
    
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
        prompt_text = observation
        
        # Add step information to the prompt
        prompt_text += f"\n\nStep {step+1}" # <<< Simplified step info
        
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
    
    def _calculate_reward(self, state, llm_actions, final_action, step_rewards: List[float]):
        """
        Calculate the final reward for the episode.
        The default implementation returns the sum of rewards received from _step_episode. (Cumulative reward)
        Subclasses can override this for task-specific final reward calculation.

        e.g. 
        - Terminal reward = 1.0 if correct solution, 0.0 otherwise
        - Path efficiency reward = -0.1 for each step taken

        Args:
            state: Final state of the episode
            llm_actions: List of all LLM actions taken during the episode
            final_action: Final action taken by the LLM
            step_rewards: List of rewards returned by _step_episode at each step.

        Returns:
            Final reward value
        """
        # Default implementation: Cumulative reward
        return sum(step_rewards)
    
    def _run_complete_episode(self, session_id, agent: Agent):
        """Run a complete episode using the provided agent.
        
        Args:
            session_id: ID of the session
            agent: The agent instance responsible for generating actions.
            
        Returns:
            Dict containing unpadded lists for the episode: 
            {"full_token_ids": List[int], "full_attention_mask": List[int], 
             "agent_token_mask": List[int], "per_token_rewards": List[float], 
             "final_reward": float}
        """
        # Get the active state
        state = self.active_states[session_id]
        task_id = state.get('task_id', 'N/A')
        self.logger.info(f"[{session_id}] Starting episode (Task ID: {task_id})")

        # Initialize structures for R4 output
        full_token_ids = []
        full_attention_mask = []
        agent_token_mask = [] # 0 for env/prompt tokens, 1 for agent tokens
        per_token_rewards = [] # Will be populated after episode completion
        step_rewards = [] # Store scalar rewards per step
        all_llm_actions = [] # Store processed actions for final reward calculation
        
        # Run the episode until completion or max steps
        # self.logger.debug(f"[{session_id}] Entering episode loop. Initial state: done={state['done']}, steps={state['steps']}")
        # Initialize conversation history and retrieve per-session agent state
        agent_state = self.agent_states.get(session_id)
        messages: List[Dict[str, Any]] = []

        # Track token counts per turn
        prompt_token_counts = []
        response_token_counts = []
        
        is_done = False # <<< Loop control flag
        while not is_done:
            current_step = state["steps"]
            self.logger.info(f"[{session_id}] Step {current_step+1}")

            # Format observation as prompt for the LLM
            prompt_text = self._format_prompt(state, current_step)
            
            # Tokenize prompt and update R4 structures
            # Assume tokenizer adds BOS if needed, but not EOS for prompts
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            prompt_token_counts.append(prompt_len)
            
            full_token_ids.extend(prompt_tokens)
            full_attention_mask.extend([1] * prompt_len)
            agent_token_mask.extend([0] * prompt_len) # Prompt tokens are not from agent
            
            # Append user prompt to conversation history
            messages.append({"role": "user", "content": prompt_text, "session_id": session_id})

            # Agent generates the next response
            action_text, agent_state = agent.get_action(messages, agent_state)
            self.agent_states[session_id] = agent_state
            # Attempt to get token IDs from agent (might not always be available)
            response_token_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
            response_len = len(response_token_ids)
            response_token_counts.append(response_len)

            # Update R4 structures with agent response tokens
            full_token_ids.extend(response_token_ids)
            full_attention_mask.extend([1] * response_len)
            agent_token_mask.extend([1] * response_len) # Response tokens are from agent

            # Process the agent's action
            llm_action = self._process_llm_response(action_text, tokenizer=self.tokenizer)
            all_llm_actions.append(llm_action) # <<< Store for final reward calc
            self.logger.info(f"[{session_id}] Agent action: {llm_action}")

            # Append assistant message to history
            messages.append({"role": "assistant", "content": llm_action, "session_id": session_id})

            # Take a step in the environment
            # next_state, reward, done, info = self._step_episode(session_id, state, llm_action) # <<< Call renamed method
            next_state, reward, done, info = self.step(session_id, state, llm_action) # <<< Call renamed method
            state = next_state # Update state
            # state["done"] = done # Keep original done info in state if needed later
            is_done = done if isinstance(done, bool) else done[0] # Update loop control flag
            step_rewards.append(reward) # <<< Store scalar step reward
            
            # Log step results
            self.logger.info(f"[{session_id}] Step {current_step+1} result: reward={reward}, done={done}")
            if info:
                self.logger.debug(f"[{session_id}] Step info: {info}")

        # Log reason for loop exit
        if is_done:
             self.logger.info(f"[{session_id}] Episode loop finished because done=True.")
        else:
             self.logger.warning(f"[{session_id}] Episode loop finished unexpectedly (done={state['done']}, steps={state['steps']}). Consider adding max step check based on env_config.")

        # Calculate final scalar reward 
        final_action = all_llm_actions[-1] if all_llm_actions else None
        # Pass the list of step rewards to _calculate_reward
        final_reward = self._calculate_reward(state, all_llm_actions, final_action, step_rewards)
        self.logger.info(f"[{session_id}] Final reward: {final_reward}") # Log the correct final reward

        # --- Per-token reward calculation (Example: Distribute step reward to preceding agent tokens) ---
        total_tokens = len(full_token_ids)
        per_token_rewards = [0.0] * total_tokens # Initialize with zeros
        
        # Iterate through steps to distribute rewards
        current_token_idx = 0
        for i in range(len(step_rewards)):
            prompt_len = prompt_token_counts[i]
            response_len = response_token_counts[i]
            step_reward = step_rewards[i]
            
            # Calculate start and end index for the agent's response tokens in this step
            response_start_idx = current_token_idx + prompt_len
            response_end_idx = response_start_idx + response_len
            
            # Distribute step reward evenly across agent tokens of that step
            if response_len > 0:
                reward_per_token = step_reward / response_len
                for j in range(response_start_idx, response_end_idx):
                    # Ensure index is within bounds (should always be if logic is correct)
                    if j < total_tokens:
                         per_token_rewards[j] = reward_per_token
                    else:
                        self.logger.warning(f"[{session_id}] Token index out of bounds during reward distribution.")
            
            # Move index to the start of the next prompt
            current_token_idx = response_end_idx
            
        # --- End Per-token reward calculation ---

        # Log episode summary (using final scalar reward)
        self.logger.info(f"[{session_id}] Episode summary:")
        self.logger.info(f"  - Task ID: {task_id}")
        self.logger.info(f"  - Steps completed: {state['steps']}")
        self.logger.info(f"  - Done: {state['done']}")
        self.logger.info(f"  - Final reward: {final_reward:.4f}") # Use final scalar reward
        self.logger.info(f"  - Actions taken: {len(all_llm_actions)}")
        self.logger.info(f"  - Total tokens: {total_tokens}") # Add token count info
        
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
                    f.write(f"Environment Class: {self.__class__.__name__}\n")
                    f.write(f"Steps: {state['steps']}\n")
                    f.write(f"Done: {state['done']}\n")
                    f.write(f"Final reward: {final_reward:.4f}\n\n")
                    
                    # Write detailed step information
                    for i in range(len(step_rewards)):
                        f.write(f"\n--- Step {i+1} ---\n")
                        f.write(f"Step reward: {step_rewards[i]}\n")
                        f.write(f"Per-token rewards: {per_token_rewards[i*prompt_len:(i+1)*prompt_len]}\n")
                        f.write(f"Full token IDs: {full_token_ids[i*prompt_len:(i+1)*prompt_len]}\n")
                        f.write(f"Full attention mask: {full_attention_mask[i*prompt_len:(i+1)*prompt_len]}\n")
                        f.write(f"Agent token mask: {agent_token_mask[i*prompt_len:(i+1)*prompt_len]}\n")
                
                self.logger.info(f"[{session_id}] Detailed episode log written to: {summary_path}")
        except Exception as e:
            self.logger.warning(f"[{session_id}] Failed to write detailed episode log: {e}")

        # Clean up
        del self.active_states[session_id]
        
        # Return data according to R4 schema
        return {
            "full_token_ids": full_token_ids,
            "full_attention_mask": full_attention_mask,
            "agent_token_mask": agent_token_mask,
            "per_token_rewards": per_token_rewards,
            "final_reward": final_reward # Scalar final reward for the episode
        }
    
    def run_trial(
        self,
        task_data_list: List[Dict[str, Any]],
        agent: Agent,
        num_rollouts: int
    ) -> Dict[str, Union[List[List[int]], List[float]]]:
        """Run multiple rollouts for a batch of tasks and return padded trajectory data.

        Args:
            task_data_list: A list of task data dictionaries, where each dictionary
                            is used to initialize an episode via _initialize_episode.
            agent: The agent instance to use for generating actions.
            num_rollouts: The number of independent rollouts to perform for EACH task
                          in task_data_list.

        Returns:
            A dictionary containing aggregated, padded trajectory data for all rollouts
            across all input tasks. Keys are:
            - "padded_full_token_ids": List[List[int]]
            - "padded_full_attention_mask": List[List[int]]
            - "padded_agent_token_mask": List[List[int]]
            - "padded_per_token_rewards": List[List[float]]
            - "final_rewards": List[float] (Scalar reward for each rollout)
        """
        all_rollout_results = [] # Store results from _run_complete_episode for all rollouts
        task_rollout_indices = [] # Keep track of which rollouts belong to which task

        for task_idx, task_data in enumerate(task_data_list):
            task_rollouts_start_idx = len(all_rollout_results)
            self.logger.info(f"Starting {num_rollouts} rollouts for task {task_idx+1}/{len(task_data_list)}.")
            for rollout_num in range(num_rollouts):
                session_id = str(uuid.uuid4()) # Generate unique ID for each rollout
                self.logger.debug(f"  - Rollout {rollout_num+1}/{num_rollouts} (session: {session_id})")
                
                try:
                    # Initialize episode using task_data
                    initial_state = self._initialize_episode(session_id, task_data)
                    self.active_states[session_id] = initial_state
                    # Ensure agent state is reset for each new rollout if necessary (agent might handle this)
                    self.agent_states[session_id] = None 

                    # Run the complete episode
                    episode_results = self._run_complete_episode(session_id, agent)
                    all_rollout_results.append(episode_results)
                
                except Exception as e:
                    self.logger.error(f"Error during rollout {rollout_num+1} for task {task_idx+1} (session {session_id}): {e}")
                    # Handle error: maybe append dummy data or skip?
                    # For now, appending a minimal structure to avoid downstream errors.
                    all_rollout_results.append({
                        "full_token_ids": [], "full_attention_mask": [], "agent_token_mask": [],
                        "per_token_rewards": [], "final_reward": 0.0 
                    })
                finally:
                    # Clean up agent state if it exists
                    if session_id in self.agent_states:
                        del self.agent_states[session_id]
                    # Ensure active state is cleaned up even if _run_complete_episode failed before cleanup
                    if session_id in self.active_states:
                        del self.active_states[session_id]
            
            task_rollouts_end_idx = len(all_rollout_results)
            task_rollout_indices.append((task_rollouts_start_idx, task_rollouts_end_idx))

        # --- Padding Logic --- 
        # Initialize lists to hold the final padded data for the entire batch
        batch_padded_token_ids = []
        batch_padded_attn_mask = []
        batch_padded_agent_mask = []
        batch_padded_rewards = []
        batch_final_rewards = []
        
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for padding but is not initialized.")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
             self.logger.warning("Using pad_token_id=0 as tokenizer.pad_token_id is None.")
             pad_token_id = 0 # Default padding if tokenizer doesn't have one

        # Pad results per task group
        for task_idx, (start_idx, end_idx) in enumerate(task_rollout_indices):
            task_rollouts = all_rollout_results[start_idx:end_idx]
            if not task_rollouts: continue # Skip if a task had no successful rollouts
            
            self.logger.debug(f"Padding results for task {task_idx+1} (Rollouts {start_idx}-{end_idx-1})")
            
            # Find max sequence length within this task's rollouts
            max_len = 0
            for rollout in task_rollouts:
                max_len = max(max_len, len(rollout["full_token_ids"]))
            self.logger.debug(f"  - Max sequence length for task {task_idx+1}: {max_len}")
            
            # Pad each rollout sequence for this task to max_len
            for rollout in task_rollouts:
                seq_len = len(rollout["full_token_ids"])
                padding_len = max_len - seq_len
                
                # Pad token IDs (use pad_token_id)
                padded_ids = rollout["full_token_ids"] + [pad_token_id] * padding_len
                batch_padded_token_ids.append(padded_ids)
                
                # Pad attention mask (use 0 for padding)
                padded_attn = rollout["full_attention_mask"] + [0] * padding_len
                batch_padded_attn_mask.append(padded_attn)
                
                # Pad agent mask (use 0 for padding - padding tokens are not agent tokens)
                padded_agent = rollout["agent_token_mask"] + [0] * padding_len
                batch_padded_agent_mask.append(padded_agent)
                
                # Pad per-token rewards (use 0.0 for padding)
                padded_rew = rollout["per_token_rewards"] + [0.0] * padding_len
                batch_padded_rewards.append(padded_rew)
                
                # Add the scalar final reward for this rollout
                batch_final_rewards.append(rollout["final_reward"])

        # Aggregate results into the final dictionary
        final_padded_results = {
            "padded_full_token_ids": batch_padded_token_ids,
            "padded_full_attention_mask": batch_padded_attn_mask,
            "padded_agent_token_mask": batch_padded_agent_mask,
            "padded_per_token_rewards": batch_padded_rewards,
            "final_rewards": batch_final_rewards
        }
        
        self.logger.info(f"run_trial completed. Returning {len(batch_final_rewards)} padded trajectories.")
        return final_padded_results
