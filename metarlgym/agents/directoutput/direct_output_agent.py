from typing import Any, Tuple, List, Dict, Optional
from metarlgym.agents.base import Agent

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

class DirectOutputAgent(Agent):
    """
    A simple agent that directly generates an action from the LLM given a prompt history.

    This agent concatenates message contents into a single prompt string, calls the LLM once,
    decodes the output tokens, and returns the resulting text as its action.
    """
    def __init__(
        self,
        llm: Any,
        sampling_params: Any,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            llm: A vLLM LLM instance or any object with a `.generate()` method.
            sampling_params: Sampling parameters with attributes
                repetition_penalty, temperature, top_p, top_k, min_p, max_tokens.
            tokenizer: Optional tokenizer with `.decode()` to convert token ids to strings.
        """
        self.llm = llm
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer

    def get_action(
        self,
        messages: List[Dict[str, Any]],
        agent_state: Any = None,
        **kwargs: Any
    ) -> Tuple[str, Any]:
        """
        Generate a single LLM completion as the action.

        Args:
            messages: List of message dicts with "content" fields.
            agent_state: Ignored for this stateless agent.
            **kwargs: Unused.

        Returns:
            action: Generated text from the LLM.
            agent_state: None (stateless).
        """
        # Build prompt text from all message contents
        prompt = "\n".join([m.get("content", "") for m in messages])

        # Generate token ids
        completion_ids_list = self.llm.generate(
            prompts=[prompt],
            n=1,
            repetition_penalty=getattr(self.sampling_params, "repetition_penalty", None),
            temperature=getattr(self.sampling_params, "temperature", None),
            top_p=getattr(self.sampling_params, "top_p", None),
            top_k=getattr(self.sampling_params, "top_k", -1),
            min_p=getattr(self.sampling_params, "min_p", 0.0),
            max_tokens=getattr(self.sampling_params, "max_tokens", None),
        )
        # completion_ids_list is like List[RequestOutput]
        # RequestOutput has outputs: List[CompletionOutput]
        # CompletionOutput has token_ids: List[int]
        token_ids = completion_ids_list[0].outputs[0].token_ids
        # Cache token ids for external inspection
        self.last_token_ids = token_ids

        # Decode tokens to text
        if self.tokenizer is not None:
            try:
                action = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception:
                action = "".join(map(str, token_ids))
        else:
            action = "".join(map(str, token_ids))

        return action, None

    def reset(self) -> None:
        """No-op reset for stateless agent."""
        pass