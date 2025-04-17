import metarlgym as rlgym

model_name = "Qwen/Qwen2.5-Math-1.5B"
model, tokenizer = rlgym.get_model_and_tokenizer(model_name)

rlgym_env = rlgym.TwentyQuestionsEnv(tokenizer=tokenizer)
dataset = rlgym_env.get_dataset()
rubric = rlgym_env.get_rubric()

run_name = "twenty_questions_" + model_name.split("/")[-1].lower()
training_args = rlgym.get_default_grpo_config(run_name=run_name, num_gpus=2)
trainer = rlgym.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=rlgym_env,
    args=training_args,
    train_dataset=dataset,
    agent_config=
)
trainer.train()
