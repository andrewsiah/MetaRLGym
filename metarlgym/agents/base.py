from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict

class Agent(ABC):
    """
    Abstract base class for MetarlGym agents.

    Agents operate in an RL-style loop: they receive an observation and previous reward,
    then return an action and an updated internal state.
    """

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
            messages: History of messages (e.g., {"role": ..., "content": ...})
                representing the conversation or prompt history.
            agent_state: Optional agent-specific internal state carried across steps.
            **kwargs: Additional agent-specific parameters.

        Returns:
            action: The generated message content or structured action.
            agent_state: Updated agent-specific internal state for the next invocation.
        """
        pass

    def reset(self) -> None:
        """
        Reset the agent's internal state or memory before a new episode.
        Override if the agent maintains memory.
        """
        pass

    def update_memory(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        (Deprecated) Agents should manage memory via the state returned by get_action().
        Kept for backward compatibility.
        """
        pass