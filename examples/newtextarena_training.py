import logging
import os
import metarlgym as rlgym
import textarena as ta
from textarena.agents.basic_agents import OpenRouterAgent
from metarlgym.envs.textarena_multistep_env import TextArenaMultistepEnv
from metarlgym.utils.logging_utils import setup_logging, ensure_logs_directory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up model name and run name
model_name = "Qwen/Qwen2.5-7B-Instruct"
run_name = "textarena_multistep_" + model_name.split("/")[-1].lower()

# When using accelerate, get the process rank for separate log files
try:
    from accelerate.state import AccelerateState
    accelerate_state = AccelerateState()
    process_rank = accelerate_state.process_index
except (ImportError, AttributeError):
    process_rank = 0

# Setup logging with proper directory structure and process-specific files
log_dir = ensure_logs_directory()
logging_result = setup_logging(
    level="DEBUG",
    log_to_file=True,
    log_dir=log_dir,
    log_file=f"{run_name}_rank{process_rank}_{rlgym.utils.logging_utils.datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    run_name=run_name
)

logger = logging_result["logger"]
log_file_path = logging_result["log_file_path"]

logger.info(f"Starting MultistepEnv TextArena training run with model: {model_name}")
logger.info(f"Logging to: {log_file_path} (Process rank: {process_rank})")

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
logger.info("Creating MultistepEnv environment wrapper")
rlgym_env = TextArenaMultistepEnv(
    env_id="Stratego-v0",
    task_dataset_size=1000,
    system_prompt="You are playing a game of Stratego. Make strategic moves to capture the opponent's flag while protecting your own.",
    max_steps_per_episode=10,
    opponent_policy=lambda obs: opponent(obs) if isinstance(obs, str) else opponent(obs.get("observation", str(obs))),
    tokenizer=tokenizer  # Explicitly pass tokenizer
)

# Verify tokenizer is set
if hasattr(rlgym_env, 'tokenizer') and rlgym_env.tokenizer:
    logger.info(f"Environment initialized with tokenizer: {type(rlgym_env.tokenizer).__name__}")
else:
    logger.warning("Environment doesn't have a tokenizer set!")

# Get the dataset and rubric for training
logger.info("Preparing training dataset and rubric")
dataset = rlgym_env.get_dataset()
logger.info(f"Dataset created with {len(dataset) if dataset else 0} samples")
rubric = rlgym_env.get_rubric()

# Setup training configuration
logger.info(f"Creating training configuration with run name: {run_name}")
training_args = rlgym.get_default_grpo_config(run_name=run_name, num_gpus=1)

# Log key training parameters
logger.info("Training configuration:")
logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  - Learning rate: {training_args.learning_rate}")
logger.info(f"  - Num epochs: {training_args.num_train_epochs}")

# Initialize the trainer with the tokenizer
logger.info("Initializing GRPOEnvTrainer")
trainer = rlgym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=rlgym_env,
    args=training_args,
    train_dataset=dataset,
)

# Set the tokenizer attribute on env_kwargs too
if not hasattr(trainer, 'env_kwargs'):
    trainer.env_kwargs = {}
trainer.env_kwargs['tokenizer'] = tokenizer
logger.info("Added tokenizer to trainer's env_kwargs")

# Start training
logger.info("Starting training process")
try:
    trainer.train()
    logger.info("Training completed successfully")
except Exception as e:
    logger.error(f"Training failed with error: {e}", exc_info=True)
    raise

logger.info(f"Training run {run_name} finished")