import logging
import os
import sys
import datetime
from typing import Optional, Dict, Union

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

def get_model_name(model_path: str) -> str:
    """
    Extract a simplified model name from a full model path.
    
    Args:
        model_path: The full model path (e.g., 'Qwen/Qwen3-1.7B')
        
    Returns:
        A simplified model name (e.g., 'Qwen2.5-1.5B')
    """
    # Extract the model name from the path
    model_name = os.path.basename(model_path)
    
    # Remove any "Instruct" suffix for cleaner names
    model_name = model_name.replace("-Instruct", "").replace("Instruct", "")
    
    return model_name

def ensure_logs_directory(subdirectory: Optional[str] = None) -> str:
    """
    Ensure the logs directory exists and create a date-based subdirectory if requested.
    
    Args:
        subdirectory: Optional subdirectory name. If None, uses current date in YYYY-MM-DD format.
        
    Returns:
        Path to the logs directory
    """
    # Create base logs directory
    base_logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(base_logs_dir, exist_ok=True)
    
    # Create subdirectory if requested
    if subdirectory is None:
        # Use current date as subdirectory name
        current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        subdirectory = current_date
    
    log_dir = os.path.join(base_logs_dir, subdirectory)
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir

def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, Union[str, logging.Logger]]:
    """
    Setup comprehensive logging configuration for the agentsgym package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
        log_to_file: Whether to log to file in addition to console. Defaults to True.
        log_dir: Directory to store log files. If None, creates a date-based directory.
        log_file: Name of the log file. If None, uses a timestamp-based name.
        run_name: Optional name for the run, used in log file naming.
        
    Returns:
        Dictionary containing logger and log file path
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Get the root logger for the agentsgym package
    logger = logging.getLogger("agentsgym")
    logger.setLevel(level.upper())
    # Clear any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False 

    # Create a StreamHandler that writes to stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(console_handler)
    
    log_file_path = None
    
    # Add file handler if requested
    if log_to_file:
        # Setup logs directory
        if log_dir is None:
            log_dir = ensure_logs_directory()
        else:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create log file name if not provided
        if log_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "training"
            if run_name:
                prefix = f"{run_name}"
            log_file = f"{prefix}_{timestamp}.log"
        
        log_file_path = os.path.join(log_dir, log_file)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))
        file_handler.setLevel(level.upper())
        logger.addHandler(file_handler)
        
        # Log the setup information
        logger.info(f"Logging to file: {log_file_path}")
    
    return {
        "logger": logger,
        "log_file_path": log_file_path
    }

def log_conversation(
    logger: logging.Logger,
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, any]] = None
) -> None:
    """
    Log a conversation message with structured format for better readability in logs.
    
    Args:
        logger: Logger instance to use
        session_id: ID of the conversation session
        role: Role of the speaker (system, user, assistant, etc.)
        content: Message content
        metadata: Optional metadata to include
    """
    # Create a formatted log message
    formatted_content = content.replace("\n", " \\n ")[:500]  # Truncate and format newlines
    if len(content) > 500:
        formatted_content += "... [truncated]"
    
    # Basic log message
    log_message = f"[{session_id}] {role.upper()}: {formatted_content}"
    
    # Add metadata if provided
    if metadata:
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        log_message += f" | {meta_str}"
    
    # Log at appropriate level
    logger.info(log_message)

def log_environment_step(
    logger: logging.Logger,
    session_id: str,
    player_id: int,
    action: str,
    observation: str,
    reward: Union[float, Dict[int, float], None] = None,
    done: bool = False,
    info: Optional[Dict] = None
) -> None:
    """
    Log an environment step with detailed information.
    
    Args:
        logger: Logger instance to use
        session_id: ID of the session
        player_id: ID of the player taking the action
        action: Action taken
        observation: Resulting observation
        reward: Reward received (if any)
        done: Whether the episode is done
        info: Additional info dict
    """
    step_summary = (
        f"[{session_id}] STEP: Player {player_id} | "
        f"Action: {str(action)[:100]}{'...' if len(str(action)) > 100 else ''} | "
        f"Done: {done}"
    )
    
    if reward is not None:
        step_summary += f" | Reward: {reward}"
    
    logger.info(step_summary)
    
    # Log detailed observation separately (potentially at debug level)
    if isinstance(observation, str):
        obs_snippet = observation.replace("\n", " \\n ")[:200]
        if len(observation) > 200:
            obs_snippet += "... [truncated]"
        logger.debug(f"[{session_id}] OBSERVATION: {obs_snippet}")
    else:
        logger.debug(f"[{session_id}] OBSERVATION: {type(observation)}")
    
    # Log additional info if available
    if info:
        info_str = str(info)[:200]
        if len(str(info)) > 200:
            info_str += "... [truncated]"
        logger.debug(f"[{session_id}] INFO: {info_str}")

def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[dict],
    rewards: list[float],
    step: int,
) -> None:
    """
    Print a formatted sample of prompts, completions, and rewards.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion dictionaries or lists
        rewards: List of reward floats
        step: Current step number
    """
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    table.add_column("Reward", style="bold cyan", justify="right")

    for prompt, completion, reward in zip(prompts, completions, rewards, strict=True):
        # Create a formatted Text object for completion with alternating colors based on role
        formatted_completion = Text()
        
        if isinstance(completion, dict):
            # Handle single message dict
            role = completion.get("role", "")
            content = completion.get("content", "")
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
        elif isinstance(completion, list):
            # Handle list of message dicts
            for i, message in enumerate(completion):
                if i > 0:
                    formatted_completion.append("\n\n")
                
                role = message.get("role", "")
                content = message.get("content", "")
                
                # Set style based on role
                style = "bright_cyan" if role == "assistant" else "bright_magenta"
                
                formatted_completion.append(f"{role}: ", style="bold")
                formatted_completion.append(content, style=style)
        else:
            # Fallback for string completions
            formatted_completion = Text(str(completion))

        table.add_row(Text(prompt), formatted_completion, Text(f"{reward:.2f}"))
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)