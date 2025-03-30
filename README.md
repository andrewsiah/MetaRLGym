# MetaRLGym


This is a training repo for MetaRL with LLMs. 
MetaRL involves teaching a LLM to learn to learn, by training an LLM over many tasks from many environments.


Paper:
[A Survey of Meta Reinforcement Learning](https://arxiv.org/abs/2301.08028)

Our design constraint involves:
- Use training code from TRL, currently prioritizing GRPO.
- We use designs from OpenAI's Gym Library, inheriting it from TextArena.

## Glossary

- **Outer Loop (Slow RL)**: LLM learning across multiple environments and tasks
- **Inner Loop (Fast RL)**: Policy deployed in a single task, potentially adapting across episodes
- **Environment/Task Group**: A distribution of tasks (e.g., Wordle, Twenty Questions)
- **Task**: A specific MDP instance (e.g., the word "Apple" in Wordle)
- **Episode/Trajectory**: Complete sequence $(s_0, a_0, r_0, s_1, ..., s_T)$ in a task
- **Experience**: Single interaction $(s_t, a_t, r_t, s_{t+1})$ with environment
- **Step**: Taking action $a$ to transition from $s_t$ to $s_{t+1}$
- **Self-Play/Imagination Rollouts**: LLM simulating steps without environment interaction
- **Lifetime/Trial**: Multiple episodes of the same task with environment interaction
- **Meta-Trajectory**: Collection of episode data $D = \{\tau_1, \tau_2, ..., \tau_H\}$ from a trial
- **Free Shots**: Initial exploration episodes with zero rewards to prevent under-exploration