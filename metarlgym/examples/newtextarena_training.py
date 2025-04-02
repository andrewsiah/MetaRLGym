import logging
import metarlgym as rlgym
import textarena as ta
from textarena.agents.basic_agents import OpenRouterAgent
from metarlgym.envs.textarena_multistep_env import TextArenaMultistepEnv
import os
import sys
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# When using accelerate, get the process rank for separate log files
try:
    from accelerate.state import AccelerateState
    accelerate_state = AccelerateState()
    process_rank = accelerate_state.process_index
except (ImportError, AttributeError):
    process_rank = 0

# Use a process-specific log filename
log_file_path = os.path.join(log_dir, f"training_rank{process_rank}.log")

# Setup logging as early as possible, directing to file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=log_file_path,
    filemode='a', # Append to the log file
    force=True # Allow reconfiguring the root logger
)

# Also add a handler for console output
console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
console_handler.setLevel(logging.INFO) # Log INFO and above to console
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Add a debug message to verify logging is working
logging.info(f"Logging initialized for process rank {process_rank}")
logging.debug(f"Debug logging initialized, writing to {log_file_path}")

# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if "OPENROUTER_API_KEY" not in os.environ:
    raise ValueError("Please ensure OPENROUTER_API_KEY is set in your .env file")

# Initialize the base model we'll use for training
model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = rlgym.get_model_and_tokenizer(model_name)

# Create the TextArena environment
base_env = ta.make("Stratego-v0")
base_env = ta.wrappers.LLMObservationWrapper(env=base_env)
base_env = ta.wrappers.SimpleRenderWrapper(
    env=base_env,
    player_names={0: "Learner", 1: "Opponent"}
)

# Create the opponent agent that we'll train against
opponent = OpenRouterAgent(
    model_name="meta-llama/llama-3.2-1b-instruct",
    verbose=True,
)

# Create the RL environment wrapper
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
    logging.info(f"Environment initialized with tokenizer: {type(rlgym_env.tokenizer).__name__}")
else:
    logging.warning("Environment doesn't have a tokenizer set!")

# Get the dataset and rubric for training
logging.info("Getting dataset and rubric for training")
dataset = rlgym_env.get_dataset()
logging.info(f"Dataset created with {len(dataset) if dataset else 0} samples")
rubric = rlgym_env.get_rubric()
logging.info("Rubric obtained successfully")

# Setup training configuration
run_name = "textarena_" + model_name.split("/")[-1].lower()
logging.info(f"Setting up training with run name: {run_name}")
training_args = rlgym.get_default_grpo_config(run_name=run_name, num_gpus=1)
logging.info(f"Training arguments configured: {training_args}")

# Initialize the trainer with the tokenizer
logging.info("Initializing trainer")
trainer = rlgym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=rlgym_env,
    args=training_args,
    train_dataset=dataset,
)
logging.info("Trainer initialized successfully")

# Set the tokenizer attribute on env_kwargs too
if not hasattr(trainer, 'env_kwargs'):
    trainer.env_kwargs = {}
trainer.env_kwargs['tokenizer'] = tokenizer
logging.info("Added tokenizer to trainer's env_kwargs")

# Start training
logging.info("Starting training...")
trainer.train()
logging.info("Training completed") 