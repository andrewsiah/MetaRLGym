import logging
import os
import metarlgym as rlgym
import textarena as ta
from textarena.agents.basic_agents import OpenRouterAgent
from metarlgym.envs.textarena_env import TextArenaEnv
from metarlgym.utils.logging_utils import setup_logging, ensure_logs_directory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up run name
model_name = "Qwen/Qwen2.5-Math-1.5B"
run_name = "textarena_" + model_name.split("/")[-1].lower()

# Setup logging as early as possible with proper directory structure
logging_result = setup_logging(
    level="DEBUG",
    log_to_file=True,
    run_name=run_name
)

logger = logging_result["logger"]
log_file_path = logging_result["log_file_path"]

logger.info(f"Starting TextArena training run with model: {model_name}")
logger.info(f"Logging to: {log_file_path}")

# Verify API key is loaded
if "OPENROUTER_API_KEY" not in os.environ:
    logger.error("OPENROUTER_API_KEY environment variable not found!")
    raise ValueError("Please ensure OPENROUTER_API_KEY is set in your .env file")

# Initialize the base model we'll use for training
logger.info(f"Loading model and tokenizer: {model_name}")
model, tokenizer = rlgym.get_model_and_tokenizer(model_name)

# Create the TextArena environment
logger.info("Setting up TextArena environment")
base_env = ta.make("Stratego-v0")
base_env = ta.wrappers.LLMObservationWrapper(env=base_env)
base_env = ta.wrappers.SimpleRenderWrapper(
    env=base_env,
    player_names={0: "Learner", 1: "Opponent"}
)

# Create the opponent agent that we'll train against
logger.info("Setting up opponent agent")
opponent = OpenRouterAgent(
    model_name="meta-llama/llama-3.2-1b-instruct",
    verbose=True,
)

# Create the RL environment wrapper
logger.info("Creating RL environment wrapper")
rlgym_env = TextArenaEnv(
    env_id="Stratego-v0",
    task_dataset_size=1000,
    system_prompt="You are playing a game of Stratego. Make strategic moves to capture the opponent's flag while protecting your own.",
    max_steps_per_episode=10,
    opponent_policy=lambda obs: opponent(obs) if isinstance(obs, str) else opponent(obs.get("observation", str(obs))),
    tokenizer=tokenizer
)

# Get the dataset and rubric for training
logger.info("Preparing training dataset and rubric")
dataset = rlgym_env.get_dataset()
rubric = rlgym_env.get_rubric()

# Setup training configuration
logger.info(f"Creating training configuration with run name: {run_name}")
training_args = rlgym.get_default_grpo_config(run_name=run_name, num_gpus=1)

# Initialize the trainer
logger.info("Initializing GRPOEnvTrainer")
trainer = rlgym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=rlgym_env,
    args=training_args,
    train_dataset=dataset,
)

# Start training
logger.info("Starting training process")
trainer.train()
logger.info("Training completed")