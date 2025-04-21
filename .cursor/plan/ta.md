two ways:
- go back to an old commit like in verifiers, then develop from scratch again.

pro: 
- codex won't be hallucinating from old codebase.
- it'll be a cleaner codebase

con: 
- it might have error that we have forgotten about?

idea:
this is hard, cause we had one commit that just committed everything.


idea 2:
- just add a new envs.py, wrap env, develop it in parallel with the old code, 
make it run and replicable. 
we can create a new branch, delete all unnecessary code that we don't want, then start developing there.
1. pass in the textarena code in our dir
2. plan out with a simple game in our env,
3. write out tests for this env
4. develop it, see if it runs and that the experiment is replicable
3. test out more envs, write out 
3. add in the agent/policy. 

pro

issue:
if we do trial, each episode have different rewards. but our current implementation just returns one reward for the whole meta-trajectory.

can we skip this for now? just focus on treating one trial as having only one episode, or giving them only one scalar reward.

later, we can ask the environment to directly reutnr the reward in the generate() function as well. 


Ok, I can't simply just reconfig the `prompt` column as random integer seeds because the train loss takes into account the prompt IDs 
          "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
    and these prompts are necessary so we know what not to train over.
    sounds like i need to return the mask and env_feedback to the trainer as well, so it knows what to mask out.

also because reset() in envs uses random(seed), Different seeds might result in the same task which we don't want so we could make it such that the seed can also be used as an indexer by taking the modulo of the number of tasks. 
or we can add an optional index in the reset, so that if there's a dataset, it will get the right one. 


don't ever change the temp_textarena dir. that's for reference.