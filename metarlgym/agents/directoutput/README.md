# DirectOutputAgent

A simple agent that concatenates a prompt history into a single string, calls an LLM once,
and returns the generated text as its action. Useful as a baseline for new custom agents.
## Usage

```python
from metarlgym.agents.directoutput.direct_output_agent import DirectOutputAgent

# llm: a vLLM LLM instance
# sampling_params: vLLM SamplingParams()
# tokenizer: (optional) tokenizer to decode token IDs
agent = DirectOutputAgent(llm, sampling_params, tokenizer)

# messages: List of {{'role': ..., 'content': ...}} dicts
action, _ = agent.get_action(messages)
```