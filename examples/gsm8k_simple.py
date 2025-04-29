import agentsgym as gym

model_name = "Qwen/Qwen3-1.7B"
model, tokenizer = gym.get_model_and_tokenizer(model_name)

gym_env = gym.MathEnv(dataset="gsm8k", tokenizer=tokenizer)
dataset = gym_env.get_dataset()
rubric = gym_env.get_rubric()
run_name = "gsm8k_" + model_name.split("/")[-1].lower()
training_args = gym.get_default_grpo_config(run_name=run_name, num_gpus=2)
trainer = gym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=gym_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
