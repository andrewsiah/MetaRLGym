import re, random, json, os
from typing import Any, Dict, Optional, Tuple
import importlib.resources
from datasets import Dataset
import textarena as ta
import nltk
from nltk.corpus import words
from nltk import pos_tag
from transformers import PreTrainedTokenizerBase


from .renderer import create_board_str
from agentsgym.envs.multistep_env import MultistepEnv


class TwentyQuestionsEnv(MultistepEnv):
    """ Twenty Questions game environment """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 env_config: Dict[str, Any]):
        """
        Initialize the Twenty Questions environment.

        Args:
            tokenizer: Tokenizer passed to the base class.
            env_config: Configuration dictionary. Expected keys:
                - hardcore (bool, optional): Whether to use more challenging words. Defaults to False.
                - max_turns (int, optional): Maximum number of turns allowed. Defaults to 21.
                - gamemaster_model_name (str, optional): Model name for the gamemaster agent. Defaults to "google/gemini-2.5-flash-preview".
                - train_split_ratio (float, optional): Ratio of words to use for training (0.0 to 1.0). Defaults to 0.8 if enough words, otherwise adapts.
        """
        # >>> Initialize attributes needed by get_train_dataset FIRST
        self.hardcore = env_config.get('hardcore', False)
        # Load the word list before super().__init__ calls get_train_dataset
        self.word_list = self._load_words()
        # Default dataset size if not overridden by MultistepEnv

        # Initialize base multi-step environment
        # This will call get_train_dataset which needs self.word_list
        super().__init__(tokenizer=tokenizer, env_config=env_config)
        # <<< Other initializations can happen after super().__init__
        self.max_turns = env_config.get('max_turns', 21)
        # Initialize the gamemaster (injectable for testing)
        gamemaster_model_name = env_config.get('gamemaster_model_name', "google/gemini-2.5-flash-preview")
        self.gamemaster = ta.agents.OpenRouterAgent(
            model_name=gamemaster_model_name
        )
        self.gamemaster_options = ["Yes", "No", "I don't know"]
        self.gamemaster_context = None
        self.gamemaster_history = []



    def _load_words(self, words_path: Optional[str] = None):
        """
        Load words from a JSON file.
        
        The JSON file must have the format:
        {
            "basic": ["word1", "word2", ...],
            "hardcore": ["word1", "word2", ...]
        }
        
        Args:
            words_path (str, optional): Path to the JSON file containing words.
            
        Returns:
            list: A list of words filtered by the current difficulty level.
            
        Raises:
            FileNotFoundError: If the `words_path` does not exist.
            ValueError: If the JSON file has an invalid format or no matching words are found.
        """
        try:
            if words_path is not None:
                # Use provided path
                if not os.path.exists(words_path):
                    raise FileNotFoundError(f"Words data file not found at: {words_path}")
                with open(words_path, "r", encoding="utf-8") as file:
                    word_data = json.load(file)
            else:
                # Use relative path from the current file
                current_dir = os.path.dirname(__file__)
                default_words_path = os.path.join(current_dir, 'twenty_questions_words.json')
                if not os.path.exists(default_words_path):
                    # Fallback or raise error if default file is missing
                    raise FileNotFoundError(f"Default words data file not found at: {default_words_path}")
                with open(default_words_path, "r", encoding="utf-8") as file:
                    word_data = json.load(file)
            category = "hardcore" if self.hardcore else "basic"
            category_data = word_data.get(category) # Get the dictionary for 'basic' or 'hardcore'

            if not isinstance(category_data, dict):
                 # Use self.logger if available, otherwise print
                 print(f"Warning: Expected a dictionary for category '{category}', but got {type(category_data)}. Returning empty list.")
                 # Or raise ValueError("...") if preferred
                 return []

            combined_words = []
            for subcategory_key, subcategory_list in category_data.items():
                if isinstance(subcategory_list, list):
                    combined_words.extend(subcategory_list)
                else:
                    # Use self.logger if available, otherwise print
                    print(f"Warning: Expected a list of words for subcategory '{subcategory_key}' in '{category}', but got {type(subcategory_list)}. Skipping.")
            
            if not combined_words:
                raise ValueError(f"No words found within subcategories for difficulty level '{category}'.")
                
            return combined_words
            
        except Exception as e:
            raise FileNotFoundError(f"Failed to load words data: {str(e)}")
    
    def get_train_dataset(self):
        """
        Create train and eval datasets of trials for TwentyQuestions.
        Each trial is one target word; guarantee disjoint eval set.

        NOTE: This method needs to be updated per R3 to match the new schema:
        { "env_class_path": str, "env_config": Dict, "task_data": Dict }
        Currently returns the old format.

        Returns:
            datasets.Dataset: The training dataset conforming to the R1 schema.
        """
        # Flatten word list already loaded
        all_words = list(self.word_list)
        num_total_words = len(all_words)
        
        # Determine train/eval split based on config
        default_ratio = 0.8
        train_ratio = self.env_config.get('train_split_ratio', default_ratio)
        
        if not (0.0 <= train_ratio <= 1.0):
            self.logger.warning(f"Invalid train_split_ratio ({train_ratio}). Using default {default_ratio}.")
            train_ratio = default_ratio
            
        if num_total_words < 2:
            # Handle edge case: Not enough words for a meaningful split
            self.logger.warning(f"Only {num_total_words} words available. Using all for training, eval set will be empty.")
            train_words = all_words
            eval_words = []
            n_train = num_total_words
        else:
            # Calculate ideal number of training samples
            n_train = int(num_total_words * train_ratio)
            # Ensure at least one sample for eval if possible
            if n_train >= num_total_words:
                 n_train = num_total_words - 1 
            # Ensure at least one sample for train
            if n_train == 0:
                n_train = 1
                
            # Shuffle words before splitting for randomness
            random.shuffle(all_words)
            train_words = all_words[:n_train]
            eval_words = all_words[n_train:]
        
        self.logger.info(f"Dataset split: {len(train_words)} train words, {len(eval_words)} eval words.")

        # Build rows according to the new schema (R1)
        env_class_path = f"{self.__class__.__module__}.{self.__class__.__name__}"
        # Use the current instance's env_config for each row (can be customized if needed)
        env_config = self.env_config
        
        train_data = []
        for word in train_words:
            task_data = {"solution": word}
            train_data.append({
                "env_class_path": env_class_path,
                "env_config": env_config.copy(), # Use a copy to avoid modification issues
                "task_data": task_data
            })
            
        eval_data = []
        for word in eval_words:
            task_data = {"solution": word}
            eval_data.append({
                "env_class_path": env_class_path,
                "env_config": env_config.copy(),
                "task_data": task_data
            })

        # Build prompt rows: dicts with 'state':{'solution': word} # <<< Old format, remove
        # train_prompts = [[{"state": {"solution": w}}] for w in train_words] # <<< Old format, remove
        # eval_prompts = [[{"state": {"solution": w}}] for w in eval_words] # <<< Old format, remove
        
        # Assign to internal datasets as Dataset objects
        # self.train_dataset = Dataset.from_dict({"prompt": train_prompts, "solution": train_words}) # <<< Old format, replace
        # self.eval_dataset = Dataset.from_dict({"prompt": eval_prompts, "solution": eval_words}) # <<< Old format, replace
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(eval_data)

        return self.train_dataset

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
    
    def get_gamemaster_response(self, action: str) -> str:
        """
        Get the gamemaster's response based on the provided action.

        Args:
            action (str): The player's question or statement.

        Returns:
            str: The gamemaster's response.
        """

        # Validate gamemaster state
        if self.gamemaster_context is None:
            raise ValueError("Gamemaster context is not set.")
        if self.gamemaster_history is None:
            raise ValueError("History is not set.")
        if self.gamemaster_options is None:
            raise ValueError("Gamemaster options are not set.")

        # Format available response options
        options = ", ".join(f"'{opt}'" for opt in self.gamemaster_options)

        # Construct conversation history
        history = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.gamemaster_history)

        # Create prompt
        prompt = (
            f"{self.gamemaster_context}\n"
            f"{history}\n\n"
            f"Q: {action}\n"
            f"Options: {options}\n\n"
            "Please respond with the most appropriate option."
        )

        # Get response from the gamemaster agent
        response = self.gamemaster(prompt).strip()
        # Validate response
        if any(option.lower() in response.lower() for option in self.gamemaster_options):
            self.gamemaster_history.append((action, response))  # Store valid responses
        else:
            response = "I'm sorry, I don't understand. Please try asking again."
            self.gamemaster_history.append((action, response))  # Log fallback response
        return response


    def reset(self, num_players: int, seed: Optional[int] = None):
        """ Reset the environment """
        # Re-seed Python RNG for determinism in word selection
        if seed is not None:
            random.seed(seed)
        ## initialize the game state
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1, max_turns=self.max_turns)

        ## load the game word
        if not self.word_list:
             raise ValueError("Word list is empty. Cannot select a game word.")
        self.game_word = random.choice(self.word_list)

        ## update the gamemaster
        self.gamemaster_context = (
            f"You are the gamemaster for the game of '20 Questions'.\n"
            f"You will provide responses to the players' questions that guides them into guessing the target word: {self.game_word}\n"
        )
        
        ## reset the game state
        game_state = {
            "target_word": self.game_word,
            "rendered_text": f"Game word: {self.game_word}"
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player """
        prompt = (
            f"You are Player {player_id}. You are playing 20 Questions ({'Hardcore' if self.hardcore else 'Basic'}).\n"
            "The gamemaster has chosen an object that can be one or two words.\n"
            "The game will last for a maximum of 20 questions. After 20 questions, the gamemaster will prompt you to make a guess.\n"
            "You may ask your question in any manner, so long they are not wrapped in square brackets.\n"
            "Then, to make your final word guess, ensure that you wrap it with square brackets, e.g. [plane], [diving bell].\n"
            "As you play, the history of your questions and gamemaster's responses will be displayed."
        )
        return prompt
    
    # === Trial hooks for MultistepEnv ===
    def _initialize_episode(self, session_id: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize a new episode for the given trial.
        task_info should contain 'solution': the hidden word.
        """
        # Set the target word explicitly
        solution = task_info.get('solution')
        if solution is None:
            raise ValueError(f"No solution provided for trial {session_id}")
        # Reseed and reset environment state
        self.reset(num_players=1, seed=task_info.get('seed', None))
        # Override the random word choice with the provided solution
        # (reset already set self.game_word, but we ensure correct word)
        self.game_word = solution
        # Rebuild gamemaster context to reflect forced word
        self.gamemaster_context = (
            f"You are the gamemaster for the game of '20 Questions'.\n"
            f"You will provide responses to the players' questions that guides them into guessing the target word: {self.game_word}\n"
        )
        # Reset TA state to use correct word
        game_state = {"target_word": self.game_word,
                      "rendered_text": f"Game word: {self.game_word}"}
        self.state.reset(seed=task_info.get('seed', None),
                         game_state=game_state,
                         player_prompt_function=self._generate_player_prompt)
        # Return state dict for MultistepEnv
        return {"ta_state": self.state, "done": False, "steps": 0}

    def _format_prompt(self, state: Dict[str, Any], step: int) -> str:
        """
        Format the current state into a prompt string for the LLM/agent.
        """
        ta_state = state.get('ta_state')
        if ta_state is None:
            raise ValueError("TA state missing in _format_prompt")
        # Use the renderer to show current board or game info
        prompt_text = create_board_str(game_state=ta_state.game_state)
        # Append step info
        prompt_text += f"\n\nStep {step+1}/{self.max_steps_per_episode}"
        return prompt_text

    def step(self, session_id: str, state: Dict[str, Any], llm_action: Any) -> Tuple[Dict[str, Any], float, bool, Any]:
        """
        Advance one step in the episode using the LLM action.
        This method now incorporates the game logic previously in the separate 'step' method.
        It interacts directly with the TextArena state.
        """
        ta_state = state.get('ta_state')
        if ta_state is None:
            raise ValueError("TA state missing in _step_episode")
        
        # Set the current TextArena state for internal logic (like get_gamemaster_response)
        self.state = ta_state
        action = llm_action # Use the action passed from the runner
        player_id = self.state.current_player_id
        
        ## update the observation in TextArena state
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        ## validate the action and interact with TextArena state
        action_search_pattern = re.compile(r"\[([a-zA-Z\s_]+)\]")
        action_match = action_search_pattern.search(action)
        info = {} # Initialize info dict

        if not action_match or (action_match and '?' in action):
            # --- Handle Question --- 
            gamemaster_response = self.get_gamemaster_response(action)
            info['gamemaster_response'] = gamemaster_response # Add response to info

            if "history" not in self.state.game_state:
                self.state.game_state["history"] = []
            self.state.game_state["history"].append((action, gamemaster_response))
            
            # Append final guess prompt if near max turns
            # Use self.max_turns derived from env_config during __init__
            if self.state.turn >= self.max_turns - 2:
                gamemaster_response += "\nYou have run out of questions. What is your final guess?"
            self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=gamemaster_response)
            
            # Advance TextArena state turn - crucial for questions
            done = self.state.step() # ta.State.step() returns done status
            info['reason'] = self.state.game_over_reason # Get reason from ta state

        else:
            # --- Handle Guess --- 
            action_text = action_match.group(1).lower()
            info['guess'] = action_text # Add guess to info
            
            if self.game_word.lower() in action_text: # Compare with game_word set during _initialize_episode
                reason=f"Congratulations! Player {player_id} guessed the word."
                self.state.set_winners(player_ids=[player_id], reason=reason)
                # Game ends on correct guess
                self.state.game_state["rendered_text"] = f"Game word: {self.game_word}"
                done = True # Explicitly set done for correct guess
                info['reason'] = reason
            else:
                reason=f"Invalid guess. Player {player_id} guessed incorrectly."
                self.state.set_invalid_move(player_id=player_id, reason=reason)
                # Game might end on incorrect guess (depending on error allowance in ta.State)
                self.state.game_state["rendered_text"] = f"Game word: {self.game_word}"
                # Let ta.State determine if the game ends after invalid move
                done = self.state.step() # ta.State.step() returns done status after invalid move is processed
                info['reason'] = reason if done else self.state.game_over_reason # Update reason

        # Extract reward for player 0 from the potentially updated TextArena state
        rewards = getattr(self.state, 'rewards', None)
        if rewards is not None:
            reward = float(rewards.get(0, 0.0))
        else:
            reward = 0.0
            
        # Build the next state dictionary required by _run_complete_episode
        # Include the updated ta_state
        next_state = {"ta_state": self.state, # Pass back the modified ta_state
                      "done": done,
                      "steps": state.get('steps', 0) + 1}
                      
        return next_state, reward, done, info
    
    def _load_word_list(self, word_list: list) -> list:
        """
        Load the word list for the game.

        Args:
            word_list: The word list to load.

        Returns:
            list: The loaded word list.
        """
        # NN: Noun
        return [word for word in word_list if pos_tag([word])[0][1] in ["NN"]]

