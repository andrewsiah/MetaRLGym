from abc import abstractmethod
import json
import random
from typing import List, Dict, Sequence, Any, Union

from datasets import Dataset

from vllm import LLM, SamplingParams  # type: ignore
from metarlgym.envs.environment import Environment


class OneStepEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 tokenizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        self.tokenizer = tokenizer

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        
        custom_sp = sampling_params.clone() 
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        states = [{
            "messages": m,
            "prompt_ids": [],
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # Convert message dicts to text prompts
        text_prompts = []
        for prompt_messages in prompts:
            prompt_text = ""
            for msg in prompt_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_text += f"system: {content}\n"
                elif role == "user":
                    prompt_text += f"user: {content}\n"
                elif role == "assistant":
                    prompt_text += f"assistant: {content}\n"
            text_prompts.append(prompt_text)

        # Get completions using generate method
        completion_ids = llm.generate(
            prompts=text_prompts,
            n=custom_sp.n,
            repetition_penalty=custom_sp.repetition_penalty,
            temperature=custom_sp.temperature, 
            top_p=custom_sp.top_p,
            top_k=custom_sp.top_k if custom_sp.top_k is not None else -1,
            min_p=custom_sp.min_p if custom_sp.min_p is not None else 0.0,
            max_tokens=custom_sp.max_tokens,
            guided_decoding_regex=getattr(custom_sp, 'guided_decoding_regex', None)
        )

        # Process completions without relying on the chat API response format
        for i, comp_ids in enumerate(completion_ids):
            # Decode the completion IDs into text if tokenizer is available
            completion_text = "<generated_content>"
            if self.tokenizer is not None:
                completion_text = self.tokenizer.decode(comp_ids, 
                                                      skip_special_tokens=self.sampling_args.get("skip_special_tokens", False))
            
            # Add assistant message with the decoded text
            states[i]["messages"].append({"role": "assistant", "content": completion_text})
            
            # Update token tracking information 
            states[i]["completion_ids"] = comp_ids
            states[i]["completion_mask"] = [1] * len(comp_ids)

        output = {
            "ids": [states[i]["completion_ids"] for i in range(len(states))],
            "messages": [states[i]["messages"][-1:] for i in range(len(states))],
            "mask": [states[i]["completion_mask"] for i in range(len(states))]
        }
        return output