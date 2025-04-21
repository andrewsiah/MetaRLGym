import argparse
import logging
import os
import time
from pathlib import Path

# Removed dotenv loading here - it's now handled in metarlgym/__init__
# from dotenv import load_dotenv
# load_dotenv()

import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

import metarlgym as rlgym


model_name = "Qwen/Qwen2.5-Math-1.5B"
model, tokenizer = rlgym.get_model_and_tokenizer(model_name)

# Initialize the TwentyQuestions environment using rlgym alias
rlgym_env = rlgym.TwentyQuestionsEnv()

dataset = rlgym_env.get_dataset()
rubric = rlgym_env.get_rubric()

run_name = "twenty_questions_" + model_name.split("/")[-1].lower()
training_args = rlgym.get_default_grpo_config(run_name=run_name, num_gpus=1)

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
trainer = rlgym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=rlgym_env,
    args=training_args,
    train_dataset=dataset,
    agent=None # Explicitly pass None initially, or let default happen
)

# Now create the agent explicitly using the trainer's vLLM client
agent = DirectOutputAgent(
    llm=trainer.vllm_client, # Use the client initialized by the trainer
    sampling_params=sampling_params,
    tokenizer=tokenizer
)

# Overwrite the trainer's agent with the explicitly created one
trainer.agent = agent

trainer.train()
