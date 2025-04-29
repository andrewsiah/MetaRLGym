there's a lot of stuff that we don't need for every env, and can be removed
from multistepenv, e.g. 
- env_id is managed by register in registration, not here?
    - what to pass to player_id = 0, which is our \pi
- system prompt can be managed by each independent env


- we can remove get_rubric? and pass a vector of reward to the trainer, so it doesn't have to calculate advantage in trainer. our env calculates advantage per token, and also what token to mask out (e.g. environment tokens.)
- but we might need a calculate_reward function that aggregates all the reward functions (tool-use, winning game, accuracy, etc) and does proper credit assignment to each token.
i.e. our env always takes in a policy, a config for an env, and returns a trajectory in language, token_id, mask, and reward for each token.