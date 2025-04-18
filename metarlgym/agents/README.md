# Agent Interface

This directory defines the `Agent` interface and provides guidelines for implementing custom agents that can interact with MetarlGym environments.

## Overview

- Agents encapsulate decision-making logic.
- They take in prompts (observations) and return actions.
- Agents can maintain internal state or memory across steps and episodes.

MetarlGym multi-step language environments call the agent using the `get_action()` method to select actions based on a conversation history of messages.

This README covers:

- Interface definition
- Required methods
- How to implement a custom agent
- Example implementations
- Integration with environments

## Interface Definition

All agents must inherit from the `Agent` abstract base class defined in `base.py`. The key interface is:

```python
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict

class Agent(ABC):
    @abstractmethod
    def get_action(
        self,
        messages: List[Dict[str, Any]],
        agent_state: Any = None,
        **kwargs: Any
    ) -> Tuple[Any, Any]:
        """
        Select an action given the current conversation messages and optional internal state.

        Follows the standard language-generation call:
            action, next_agent_state = agent.get_action(messages, agent_state)

        Args:
            messages: List of message dictionaries (e.g., {"role": ..., "content": ..., ...})
                representing the conversation or prompt history.
            agent_state: Optional agent-specific internal state carried across steps.
            **kwargs: Additional agent-specific parameters.

        Returns:
            action: The generated message content or structured action.
            agent_state: Updated agent-specific internal state for the next invocation.
        """
        pass
```

## Implementing a Custom Agent

1. Create a subclass in `metarlgym/agents/your_strategy.py`.
2. Implement `get_action()`:
   - Accept the current conversation messages and optional internal `agent_state`.
   - Decide the next action (e.g., call an LLM or policy).
   - Return `(action, agent_state)`.
3. Implement `reset()` if your agent maintains state across episodes.
4. Implement `update_memory()` if your agent records interaction history.

### Example

```python
from metarlgym.agents.base import Agent
from typing import Any, Tuple, List, Dict

class EchoAgent(Agent):
    """A simple agent that echoes the last user message."""
    def get_action(
        self,
        messages: List[Dict[str, Any]],
        agent_state: Any = None
    ) -> Tuple[Any, Any]:
        # Extract and echo the last user content
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        action = user_msgs[-1] if user_msgs else ""
        # No internal state for this stateless agent
        return action, None

    def reset(self) -> None:
        # No internal state to reset
        pass
```

## Integration with Environments

Use the agent in a multi-step language environment:

```python
# Initialize agent state and conversation history
agent_state = None
session_id = "episode_1"
messages = [
    {"role": "system", "content": "System instructions here."},
    {"role": "user", "content": "Initial user prompt.", "session_id": session_id}
]

done = False
while not done:
    # Agent generates next assistant message
    action, agent_state = agent.get_action(messages, agent_state)
    # Append assistant message to conversation history
    messages.append({"role": "assistant", "content": action, "session_id": session_id})
    # Step the environment with the action
    done, info = env.step(action)
    # (Optional) record transitions, update memory, etc.
```