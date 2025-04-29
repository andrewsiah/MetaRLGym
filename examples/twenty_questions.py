import argparse
import logging
import os
import time
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from vllm import SamplingParams

import agentsgym as gym
from agentsgym.agents import DirectOutputAgent


model_name = "Qwen/Qwen2.5-Math-1.5B"
model, tokenizer = gym.get_model_and_tokenizer(model_name)

# Initialize the TwentyQuestions environment using gym alias
gym_env = gym.TwentyQuestionsEnv()

dataset = gym_env.get_train_dataset()
if Accelerator().is_main_process: # Use Accelerator to check rank
    print(f">>> Loaded dataset. Type: {type(dataset)}, Number of Rows: {len(dataset) if hasattr(dataset, '__len__') else 'N/A (IterableDataset)'}")
rubric = gym_env.get_rubric()

run_name = "twenty_questions_" + model_name.split("/")[-1].lower()
training_args = gym.get_default_grpo_config(run_name=run_name, num_gpus=2)

# Define SamplingParams again, as we need them for explicit agent creation
sampling_params = SamplingParams(
    temperature=training_args.temperature,
    top_p=training_args.top_p,
    top_k=training_args.top_k if training_args.top_k is not None else -1,
    min_p=training_args.min_p if training_args.min_p is not None else 0.0,
    max_tokens=training_args.max_completion_length,
    repetition_penalty=training_args.repetition_penalty,
)

# Initialize trainer first (it sets up vLLM client)
print(">>> Initializing GRPOEnvTrainer...")
trainer = gym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=gym_env,
    args=training_args,
    train_dataset=dataset,
    agent=None # Explicitly pass None initially, or let default happen
)
print(">>> GRPOEnvTrainer initialized.")

# Now create the agent explicitly using the trainer's vLLM client
# This should only happen on the main process where vLLM client is guaranteed to exist
agent = None
if trainer.accelerator.is_main_process:
    print(">>> Initializing DirectOutputAgent on main process...")
    agent = DirectOutputAgent(
        llm=trainer.vllm_client, # Use the client initialized by the trainer
        sampling_params=sampling_params,
        tokenizer=tokenizer
    )
    print(">>> DirectOutputAgent initialized.")

# Assign the agent to the trainer. 
# On non-main processes, agent will be None, which is fine as it's only used on the main process.
trainer.agent = agent

print(">>> Starting trainer.train()...")
trainer.train()
