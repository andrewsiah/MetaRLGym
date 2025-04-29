# Load environment variables from .env file if it exists
# This ensures API keys etc. are available when the package is imported
from dotenv import load_dotenv
load_dotenv()

# Import key components from submodules
from .envs import Environment, TwentyQuestionsEnv
from .agents import Agent, DirectOutputAgent
from .trainers import GRPOEnvTrainer
from .utils import (
    get_default_grpo_config,
    get_model_and_tokenizer,
    get_model, 
    get_tokenizer,
    extract_boxed_answer,
    extract_hash_answer,
    preprocess_dataset,
    setup_logging,
    print_prompt_completions_sample,
)


__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    # Envs
    "Environment",
    "TwentyQuestionsEnv",
    # Agents
    "Agent",
    "DirectOutputAgent",
    # Trainers
    "GRPOEnvTrainer",
    # Utils
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]