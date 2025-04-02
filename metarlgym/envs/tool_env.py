import inspect
import json
import logging
from typing import List, Dict, Any, Callable, Optional, Sequence

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams

from metarlgym.envs.multistep_env import MultistepEnv
from metarlgym.parsers import XMLParser
from metarlgym.rubrics import ToolRubric
from metarlgym.utils.data_utils import preprocess_dataset

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any"),
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolEnv(MultistepEnv):
    def __init__(
        self,
        env_id: str = "tool-environment",
        dataset_name: str = "gsm8k",
        tools: List[Callable] = [],
        system_prompt_template: str = "",
        few_shot: List[Dict[str, str]] = [],
        sampling_args: Dict[str, Any] = {},
        observation_key: str = "observation",
        max_steps_per_episode: int = 10,
        dataset_split: str = "train",
        eval_split: str = "test",
        tokenizer=None,
        **kwargs
    ):
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        if "{tool_descriptions}" in system_prompt_template:
            self.system_prompt = system_prompt_template.format(tool_descriptions=tool_descriptions)
        else:
            self.system_prompt = system_prompt_template + "\n\nTools available:\n" + tool_descriptions
        
        # Initialize with base MultistepEnv
        super().__init__(
            env_id=env_id,
            system_prompt=self.system_prompt,
            max_steps_per_episode=max_steps_per_episode,
            observation_key=observation_key,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Environment-specific settings
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.eval_split = eval_split
        self.few_shot = few_shot
        
        # Update sampling args
        self.default_sampling_args = {
            "stop": ["</tool>", "</answer>"],
            "include_stop_str_in_output": True
        }
        self.default_sampling_args.update(sampling_args)
        
        # Setup parsers and rubric
        self.llm_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        self.rubric = ToolRubric(parser=self.llm_parser, env_parser=self.env_parser)
        
        # Load datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load and preprocess the dataset."""
        self.logger.info(f"Loading {self.dataset_name} dataset for {self.env_id}")
        try:
            self.task_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split=self.dataset_split,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
            self.eval_task_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split=self.eval_split,
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
            self.logger.info(f"Loaded training dataset with {len(self.task_dataset)} examples and evaluation dataset with {len(self.eval_task_dataset)} examples")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            # Create empty datasets as fallback
            self.task_dataset = {"prompt": [], "solution": []}
            self.eval_task_dataset = {"prompt": [], "solution": []}

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return the rubric for evaluation."""
        return self.rubric.get_reward_funcs()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                try:
                    parsed = self.llm_parser.parse(message["content"])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        step_count += 1
                except Exception:
                    pass
        return step_count

    def _format_prompt(self, state, step):
        """Format the observation as a prompt."""
        # Extract the observation from the state
        observation = state.get("observation", "")
        
        # If this is the first step, use the initial prompt
        if step == 0:
            return observation
        
        # Otherwise, the prompt is already included in the messages history
        return observation

    def _initialize_episode(self, session_id, task_info):
        """Initialize a new episode."""
        self.logger.info(f"[{session_id}] Initializing tool episode with task: {task_info.get('task_id', 'unknown')}")
        
        # Extract content and answer from task info
        content = task_info.get("content", "")
        answer = task_info.get("answer", None)
        
        # Create initial state
        initial_state = {
            "task_id": task_info.get("task_id", 0),
            "observation": content,
            "answer": answer,
            "done": False,
            "steps": 0,
            "messages": [],
            "tool_history": []
        }
        
        if isinstance(content, list) and len(content) > 0:
            # Handle the case where content is a list of message dicts
            initial_state["messages"] = content
        else:
            # Create a simple message dict
            if self.system_prompt:
                initial_state["messages"] = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(content)}
                ]
            else:
                initial_state["messages"] = [
                    {"role": "user", "content": str(content)}
                ]
        
        return initial_state

    def is_completed(self, messages: List[Dict[str, str]]) -> bool:
        """Check if the episode is completed."""
        try:
            # Check if we've hit max steps by counting tool uses
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps_per_episode:
                return True
            
            # Get the last message
            if messages and messages[-1]["role"] == "assistant":
                parsed = self.llm_parser.parse(messages[-1]["content"])
                # Check if we got a valid answer field (not just None from failed parsing)
                return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception as e:
            self.logger.error(f"Error checking completion status: {e}")
        return False

    def call_tool(self, tool_json: str) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"
            
            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            
            # Call the tool function with arguments
            result = tool_func(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"

    def _process_llm_response(self, llm_response, tokenizer=None):
        """Process the LLM response to extract tool calls or answers."""
        try:
            parsed = self.llm_parser.parse(llm_response)
            
            # If we have a tool call, extract it
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                return {"type": "tool", "content": parsed.tool}
            
            # If we have an answer, extract it
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return {"type": "answer", "content": parsed.answer}
            
            # If we couldn't parse anything meaningful
            return {"type": "unknown", "content": llm_response}
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return {"type": "error", "content": str(e)}

    def _step_episode(self, session_id, state, llm_action):
        """Take a step in the episode based on the LLM action."""
        self.logger.info(f"[{session_id}] Taking step with action type: {llm_action.get('type', 'unknown')}")
        
        next_state = state.copy()
        next_state["steps"] += 1
        
        # Process based on action type
        if llm_action.get("type") == "tool":
            # Call the tool
            tool_result = self.call_tool(llm_action["content"])
            
            # Add to tool history
            next_state["tool_history"].append({
                "tool_call": llm_action["content"],
                "result": tool_result
            })
            
            # Format the tool result as an environment response
            env_response = self.env_parser.format(result=tool_result)
            
            # Add to message history
            next_state["messages"].append({"role": "user", "content": env_response})
            next_state["observation"] = env_response
            
            # Not done yet
            done = False
            reward = 0.1  # Small reward for successful tool execution
            
        elif llm_action.get("type") == "answer":
            # Got an answer, we're done
            done = True
            
            # Calculate reward (1.0 if correct, 0.0 if not)
            expected_answer = state.get("answer")
            actual_answer = llm_action["content"]
            
            if expected_answer is not None and str(actual_answer).strip() == str(expected_answer).strip():
                reward = 1.0
            else:
                reward = 0.0
            
            self.logger.info(f"[{session_id}] Final answer: {actual_answer}, Expected: {expected_answer}, Reward: {reward}")
            
        else:
            # Unknown action type
            done = False
            reward = 0.0
        
        # Check if we've reached max steps
        if next_state["steps"] >= self.max_steps_per_episode:
            done = True
        
        next_state["done"] = done
        
        # Info for logging
        info = {
            "action_type": llm_action.get("type", "unknown"),
            "reward": reward
        }
        
        return next_state, reward, done, info

    def _calculate_reward(self, state, llm_actions, final_action):
        """Calculate the final reward for the episode."""
        # If we already have the final answer and it's correct, return 1.0
        expected_answer = state.get("answer")
        
        if final_action and final_action.get("type") == "answer":
            actual_answer = final_action["content"]
            if expected_answer is not None and str(actual_answer).strip() == str(expected_answer).strip():
                return 1.0
        
        # Otherwise calculate reward based on tool usage
        successful_tools = 0
        total_tools = 0
        
        for action in llm_actions:
            if action.get("type") == "tool":
                total_tools += 1
                # Check the corresponding tool result
                tool_idx = total_tools - 1
                if tool_idx < len(state.get("tool_history", [])):
                    tool_result = state["tool_history"][tool_idx]["result"]
                    if not tool_result.startswith("Error:"):
                        successful_tools += 1
        
        # Tool success rate (20% of total reward)
        tool_reward = 0.0
        if total_tools > 0:
            tool_reward = 0.2 * (successful_tools / total_tools)
        
        return tool_reward
        
    def step_api(self, 
            client: Any,
            model: str,
            messages: List[Dict[str, str]],
            **kwargs: Any) -> tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = messages.copy()
        
        try:            
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
            )
            
            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant", 
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)
            
            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, process the LLM response to get an action
                llm_action = self._process_llm_response(assistant_msg["content"])
                
                # If it's a tool action, call the tool and add environment response
                if llm_action.get("type") == "tool":
                    tool_result = self.call_tool(llm_action["content"])
                    env_response = self.env_parser.format(result=tool_result)
                    env_msg = {"role": "user", "content": env_response}
                    messages_copy.append(env_msg)
            
            return messages_copy, rollout_is_completed
            
        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True
    
    def eval_api(self, 
                client: Any,
                model: str,
                n: int = 100,
                max_concurrent: int = 32,
                timeout: int = 60,
                sampling_args: Dict[str, Any] = {}):
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            n: Number of examples to evaluate (default 100)
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling
            
        Returns:
            Dictionary with evaluation metrics
        """
        from asyncio import Semaphore
        import asyncio
        from tqdm.asyncio import tqdm_asyncio
        
        self.logger.info(f"Evaluating {model} on {self.dataset_name}")
        
        # Get evaluation dataset
        eval_dataset = self.get_eval_dataset(n=n)
        if eval_dataset is None:
            self.logger.error("Failed to load evaluation dataset")
            return {"error": 1.0}
            
        # Storage for results
        results = []
        rewards = []
        metrics = {
            "success_rate": 0.0,
            "accuracy": 0.0,
            "tool_usage": 0.0,
            "steps_per_episode": 0.0
        }
        
        async def process_example(example, semaphore):
            async with semaphore:
                # Initialize conversation with system prompt and few-shot examples
                prompt = example["prompt"]
                messages = example["prompt"].copy()
                answer = example["answer"]
                
                # Save the length of initial messages to extract just the interaction part later
                initial_length = len(messages)
                
                # Run the conversation loop until completion or max steps
                for _ in range(self.max_steps_per_episode):
                    try:
                        # Run step_api to get model and environment response
                        step_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.step_api(
                                client=client,
                                model=model,
                                messages=messages,
                                **sampling_args
                            )
                        )
                        
                        # Unpack the step_api result
                        messages, is_completed = step_result
                        
                        # If the rollout is completed, break the loop
                        if is_completed:
                            break
                        
                    except Exception as e:
                        self.logger.error(f"Error processing example: {str(e)}")
                        break
                
                # Extract interaction part (not system/few-shot)
                interactions = messages[initial_length:]
                
                # Calculate rewards
                episode_data = {
                    "messages": messages,
                    "llm_actions": [],
                    "tool_history": []
                }
                
                # Process each assistant message to get actions
                tool_count = 0
                successful_tools = 0
                
                for i, msg in enumerate(interactions):
                    if msg["role"] == "assistant":
                        action = self._process_llm_response(msg["content"])
                        episode_data["llm_actions"].append(action)
                        
                        # Check if this is a tool action
                        if action.get("type") == "tool":
                            tool_count += 1
                            # Look for the next user message as tool result
                            if i + 1 < len(interactions) and interactions[i + 1]["role"] == "user":
                                tool_result = interactions[i + 1]["content"]
                                # Add to tool history
                                episode_data["tool_history"].append({
                                    "tool_call": action["content"],
                                    "result": tool_result
                                })
                                # Check if tool was successful
                                if not tool_result.startswith("<result>Error"):
                                    successful_tools += 1
                
                # Get the final action (last assistant action)
                final_action = None
                for msg in reversed(interactions):
                    if msg["role"] == "assistant":
                        final_action = self._process_llm_response(msg["content"])
                        break
                
                # Check if the final answer is correct
                is_correct = False
                if final_action and final_action.get("type") == "answer":
                    actual_answer = final_action["content"]
                    if answer is not None and str(actual_answer).strip() == str(answer).strip():
                        is_correct = True
                
                # Calculate reward
                reward = 1.0 if is_correct else 0.0
                # If not correct but used tools successfully, give partial credit
                if not is_correct and tool_count > 0:
                    reward += 0.2 * (successful_tools / tool_count)
                
                return {
                    "prompt": prompt,
                    "interactions": interactions,
                    "answer": answer,
                    "predicted": final_action.get("content") if final_action else None,
                    "correct": is_correct,
                    "tool_count": tool_count,
                    "successful_tools": successful_tools,
                    "steps": len(interactions) // 2,  # Every interaction is assistant + user
                    "reward": reward
                }
        
        async def run_all_examples():
            semaphore = Semaphore(max_concurrent)
            tasks = [process_example(example, semaphore) for example in eval_dataset]
            return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {model}")
        
        # Run the evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_all_examples())
        finally:
            loop.close()
        
        # Calculate metrics
        total_examples = len(results)
        correct_count = sum(1 for r in results if r["correct"])
        total_steps = sum(r["steps"] for r in results)
        total_tools = sum(r["tool_count"] for r in results)
        successful_tools = sum(r["successful_tools"] for r in results)
        total_reward = sum(r["reward"] for r in results)
        
        metrics["success_rate"] = correct_count / total_examples
        metrics["accuracy"] = correct_count / total_examples
        metrics["steps_per_episode"] = total_steps / total_examples
        metrics["average_reward"] = total_reward / total_examples
        
        if total_tools > 0:
            metrics["tool_success_rate"] = successful_tools / total_tools
            metrics["tools_per_episode"] = total_tools / total_examples
        
        self.logger.info(f"Evaluation results for {model}:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics