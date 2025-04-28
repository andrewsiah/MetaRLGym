from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union, Optional, Tuple
import logging

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams  # type: ignore
import textarena as ta

Message = Tuple[int, str]  # maps role to content
Observations = dict[int, List[Message]]  # consists of the message seen by each player after the action
Rewards = Dict[int, int]  # maps player ID to reward
Info = Dict[str, Any]  # additional information about the environment

class Environment(ta.Env):

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"metarlgym.envs.{self.__class__.__name__}")
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    @abstractmethod
    def reset(self, num_players: int, seed: Optional[int]=None):
        """
        Resets the environment to an initial state.

        Args:
            num_players (int): Number of players in the game.
            seed (Optional[int]): Seed for the random number generator to ensure reproducibility.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: str) -> Tuple[bool, Info]:
        """
        Performs a single step in the environment.

        Args:
            player_id (int): The ID of the player taking the action.
            action (str): The action to be taken by the player.

        Returns:
            Tuple containing:
                - done (bool): Whether the episode has concluded
                - info (Dict[str, Any]): Additional information about the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_dataset(self, **kwargs: Any) -> Dataset | None:
        # TODO: Fetch the dataset from the environment
        pass

    @abstractmethod
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass
    
    @abstractmethod
    def run_trial(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        pass
