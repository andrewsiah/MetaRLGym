from typing import List, Dict, Any, Tuple
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from metarlgym.envs.simple_env import SimpleEnv
from metarlgym.parsers import XMLParser
from metarlgym.rubrics import MathRubric
from metarlgym.prompts import SIMPLE_PROMPT, MATH_FEW_SHOT
from metarlgym.utils import preprocess_dataset

from metarlgym.envs.multistep_env import MultistepEnv
import re
import random
from datasets import load_dataset
import numpy as np
from vllm import LLM, SamplingParams

def extract_answer(text):
    """Extract numerical answer from text."""
    # Look for common patterns indicating final answers
    patterns = [
        r"The answer is[: ]* *(-?\d+(?:\.\d+)?)",
        r"Therefore, [^.]*?= *(-?\d+(?:\.\d+)?)",
        r"(?:So|Thus|Hence)[, ].*? *=? *(-?\d+(?:\.\d+)?)",
        r"Final answer:? *(-?\d+(?:\.\d+)?)",
        r"(?:The )?(?:final )?(?:value|result|answer) (?:is|equals|=) *(-?\d+(?:\.\d+)?)",
        r"\\boxed{(-?\d+(?:\.\d+)?)}",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # If no pattern matches, look for the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

class MathEnv(MultistepEnv):
    """
    Multi-step environment for solving math problems with optional hints.
    
    Features:
    - Uses GSM8K or other math datasets
    - Supports requesting hints during problem-solving
    - Validates solutions against ground truth
    - Tracks reasoning steps and rewards correct solutions
    """
    
    def __init__(
        self,
        dataset_name: str = "gsm8k",
        dataset_split: str = "train",
        max_steps_per_episode: int = 3,
        reward_correct: float = 1.0,
        reward_step_penalty: float = -0.1,
        system_prompt: str = "You are a helpful math assistant. Solve the problem step-by-step.",
        hint_quality: str = "high",  # 'high', 'medium', 'low'
        **kwargs
    ):
        """Initialize MathEnv.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'gsm8k')
            dataset_split: Dataset split to use (e.g., 'train', 'test')
            max_steps_per_episode: Maximum steps per episode
            reward_correct: Reward for correct solution
            reward_step_penalty: Penalty for each additional step
            system_prompt: System prompt for the LLM
            hint_quality: Quality of hints ('high', 'medium', 'low')
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.reward_correct = reward_correct
        self.reward_step_penalty = reward_step_penalty
        self.hint_quality = hint_quality
        self.hint_llm = None  # Will be set if needed for generating hints
        
        # Determine task dataset size based on dataset
        if dataset_name == "gsm8k":
            task_dataset_size = 7000  # Approximate size of GSM8K train set
        else:
            task_dataset_size = 1000  # Default for other datasets
            
        # Initialize parent class
        super().__init__(
            env_id=f"Math-{dataset_name}-v0",
            task_dataset_size=task_dataset_size,
            system_prompt=system_prompt,
            max_steps_per_episode=max_steps_per_episode,
            **kwargs
        )
    
    def _create_task_dataset(self):
        """Create a dataset of math problems from the specified source."""
        self.logger.info(f"Loading {self.dataset_name} dataset...")
        
        try:
            # Load specified dataset
            if self.dataset_name == "gsm8k":
                # Use Hugging Face datasets to load GSM8K
                raw_dataset = load_dataset("gsm8k", "main")
                
                # Convert to our format
                prompts = []
                solutions = []
                
                # Process the dataset split
                for item in raw_dataset[self.dataset_split]:
                    question = item["question"]
                    answer = item["answer"]
                    
                    # Extract the numerical answer from the solution
                    numerical_answer = extract_answer(answer)
                    if numerical_answer is None:
                        # If extraction fails, try to find the last number in the answer
                        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", answer)
                        if numbers:
                            try:
                                numerical_answer = float(numbers[-1])
                            except ValueError:
                                numerical_answer = None
                    
                    # Only include problems where we can extract a numerical answer
                    if numerical_answer is not None:
                        prompts.append([{"role": "user", "content": question}])
                        solutions.append(numerical_answer)
                
                # Create our dataset
                self.task_dataset = {"prompt": prompts, "solution": solutions}
                
                # For evaluation, use a subset of the dataset
                eval_indices = random.sample(range(len(prompts)), min(100, len(prompts)))
                self.eval_task_dataset = {
                    "prompt": [prompts[i] for i in eval_indices],
                    "solution": [solutions[i] for i in eval_indices]
                }
                
                self.logger.info(f"Created task dataset with {len(prompts)} problems")
                
            else:
                # For custom datasets, create a simple dataset with example problems
                self.logger.warning(f"Dataset '{self.dataset_name}' not recognized, using example problems")
                
                # Create simple example problems
                prompts = []
                solutions = []
                
                for i in range(self.task_dataset_size):
                    a = random.randint(1, 100)
                    b = random.randint(1, 100)
                    op = random.choice(["+", "-", "*", "/"])
                    
                    if op == "+":
                        problem = f"What is {a} plus {b}?"
                        solution = a + b
                    elif op == "-":
                        problem = f"What is {a} minus {b}?"
                        solution = a - b
                    elif op == "*":
                        problem = f"What is {a} multiplied by {b}?"
                        solution = a * b
                    else:  # Division with clean results
                        product = a * b
                        problem = f"What is {product} divided by {a}?"
                        solution = b
                    
                    prompts.append([{"role": "user", "content": problem}])
                    solutions.append(solution)
                
                # Set up the datasets
                self.task_dataset = {"prompt": prompts, "solution": solutions}
                
                # For evaluation, use a subset of the dataset
                eval_size = min(int(self.task_dataset_size * 0.1), 100)
                self.eval_task_dataset = {
                    "prompt": prompts[:eval_size],
                    "solution": solutions[:eval_size]
                }
                
                self.logger.info(f"Created example task dataset with {len(prompts)} problems")
        
        except Exception as e:
            self.logger.error(f"Error creating task dataset: {e}")
            # Create a minimal dataset as fallback
            prompts = [
                [{"role": "user", "content": "What is 7 + 3?"}],
                [{"role": "user", "content": "If a train travels at 60 mph for 3 hours, how far does it go?"}]
            ]
            solutions = [10, 180]
            
            self.task_dataset = {"prompt": prompts, "solution": solutions}
            self.eval_task_dataset = {"prompt": prompts, "solution": solutions}
            
            self.logger.info(f"Created fallback task dataset with {len(prompts)} problems")
    
    def _initialize_episode(self, session_id, task_info):
        """Initialize a math problem episode."""
        self.logger.info(f"[{session_id}] Initializing math problem episode")
        
        # Extract problem content and solution
        content = task_info.get("content", "")
        solution = task_info.get("solution", None)
        
        # Initialize state
        initial_state = {
            "task_id": task_info.get("task_id", 0),
            "problem": content,
            "solution": solution,
            "observation": content,
            "hints_used": 0,
            "history": [],
            "done": False,
            "steps": 0,
        }
        
        return initial_state
    
    def _format_prompt(self, state, step):
        """Format the current state as a prompt for the LLM."""
        # Get the problem statement
        problem = state["problem"]
        
        # Add history of interactions if available
        history = ""
        if state["history"]:
            history = "\n\nYour previous work:\n" + "\n".join(state["history"])
        
        # Add information about hint availability
        hint_info = "\nYou can ask for a hint by writing 'HINT' at any point in your solution."
        
        # Format the final prompt
        prompt_text = (
            f"Solve this math problem step by step:\n\n{problem}{history}{hint_info}\n\n"
            f"You are on step {step+1}/{self.max_steps_per_episode} of your solution. "
            f"When you reach the final answer, format it as 'The answer is X'."
        )
        
        return prompt_text
    
    def _process_llm_response(self, llm_response, tokenizer=None):
        """Process the LLM response to identify if it's asking for a hint or providing a solution."""
        # Check if the response is asking for a hint
        if "HINT" in llm_response.upper():
            return {"type": "hint_request", "content": llm_response}
        
        # Otherwise, treat it as a solution attempt
        # Try to extract numerical answer
        answer = extract_answer(llm_response)
        
        return {
            "type": "solution_attempt",
            "content": llm_response,
            "extracted_answer": answer
        }
    
    def _generate_hint(self, state):
        """Generate a hint for the current problem based on hint quality setting."""
        problem = state["problem"]
        hints_used = state["hints_used"]
        
        # Simple hints based on quality setting
        if self.hint_quality == "low":
            # Low quality: generic hints
            hints = [
                "Try breaking down the problem into smaller steps.",
                "Consider what operations are needed to solve this problem.",
                "Make sure you understand what the problem is asking for."
            ]
        elif self.hint_quality == "medium":
            # Medium quality: more specific but still general
            if "divided by" in problem or "/" in problem:
                hints = [
                    "This problem involves division. Remember to set up the fractions correctly.",
                    "For division problems, think about what the numerator and denominator should be.",
                    "Consider whether there are any simplifications you can make before dividing."
                ]
            elif "multiply" in problem or "*" in problem or "product" in problem:
                hints = [
                    "This problem involves multiplication. Break down the factors if needed.",
                    "For multiplication, you can use distributive property if it helps.",
                    "Try to identify if there are any patterns that simplify the calculation."
                ]
            else:
                # Default hints for addition/subtraction/etc.
                hints = [
                    "Identify the key quantities in the problem.",
                    "Make sure your units are consistent throughout your solution.",
                    "Consider drawing a diagram to visualize the problem."
                ]
        else:  # high quality
            # For high quality, we could use another LLM call to generate a specific hint
            # This is a simplified version without the extra LLM call
            hints = [
                f"Look at the specific numbers in the problem and think about relationships between them.",
                f"Consider the core mathematical principle needed: is it ratios, rates, proportions, or something else?",
                f"Try to work backwards from what the problem is asking to determine your approach."
            ]
        
        # Return a hint based on how many have been used so far
        hint_index = min(hints_used, len(hints) - 1)
        return hints[hint_index]
    
    def _step_episode(self, session_id, state, llm_action):
        """Take a step in the episode based on the LLM action."""
        self.logger.info(f"[{session_id}] Processing action of type: {llm_action['type']}")
        
        # Create a copy of the state for updating
        next_state = state.copy()
        next_state["steps"] += 1
        
        # Initialize rewards and info
        reward = 0.0
        done = False
        info = {}
        
        # Process the action based on its type
        if llm_action["type"] == "hint_request":
            # Generate a hint and add it to the history
            hint = self._generate_hint(state)
            next_state["hints_used"] += 1
            next_state["history"].append(f"You asked for hint #{next_state['hints_used']}.")
            next_state["history"].append(f"Hint: {hint}")
            
            # Set the observation to include the hint
            next_state["observation"] = state["problem"]
            
            # Apply a small penalty for using a hint
            reward = self.reward_step_penalty
            info = {"hint_used": True, "hint": hint}
            
        elif llm_action["type"] == "solution_attempt":
            # Add the solution attempt to the history
            next_state["history"].append(llm_action["content"])
            
            # Check if the solution is correct
            correct = False
            extracted_answer = llm_action["extracted_answer"]
            
            if extracted_answer is not None and state["solution"] is not None:
                # Allow for small floating-point differences
                if isinstance(state["solution"], (int, float)) and isinstance(extracted_answer, (int, float)):
                    tolerance = 1e-6 if abs(state["solution"]) < 1 else 1e-6 * abs(state["solution"])
                    correct = abs(extracted_answer - state["solution"]) <= tolerance
                else:
                    correct = extracted_answer == state["solution"]
            
            # Calculate reward based on correctness
            if correct:
                reward = self.reward_correct - (next_state["steps"] - 1) * self.reward_step_penalty
                done = True
                info = {"correct": True, "extracted_answer": extracted_answer}
                self.logger.info(f"[{session_id}] Correct solution! Answer: {extracted_answer}")
            else:
                # If this is the last step and the answer is wrong, give negative reward
                if next_state["steps"] >= self.max_steps_per_episode:
                    reward = -self.reward_correct
                    done = True
                    info = {"correct": False, "extracted_answer": extracted_answer}
                    self.logger.info(f"[{session_id}] Incorrect final answer: {extracted_answer}, Expected: {state['solution']}")
                else:
                    # Not the last step, apply step penalty
                    reward = self.reward_step_penalty
                    info = {"correct": False, "extracted_answer": extracted_answer}
                    self.logger.info(f"[{session_id}] Incorrect answer so far: {extracted_answer}")
        
        # Update done flag
        next_state["done"] = done or next_state["steps"] >= self.max_steps_per_episode
        
        return next_state, reward, next_state["done"], info
    
    def _calculate_reward(self, state, llm_actions, final_action):
        """Calculate the final reward for the episode."""
        # If the last action wasn't a solution attempt, return 0
        if final_action is None or final_action["type"] != "solution_attempt":
            return 0.0
        
        # Check if the final answer is correct
        extracted_answer = final_action.get("extracted_answer")
        expected_solution = state.get("solution")
        
        if extracted_answer is not None and expected_solution is not None:
            # Allow for small floating-point differences
            if isinstance(expected_solution, (int, float)) and isinstance(extracted_answer, (int, float)):
                tolerance = 1e-6 if abs(expected_solution) < 1 else 1e-6 * abs(expected_solution)
                correct = abs(extracted_answer - expected_solution) <= tolerance
            else:
                correct = extracted_answer == expected_solution
                
            if correct:
                # Reward for correct answer, with penalty for steps used
                return self.reward_correct - (state["steps"] - 1) * self.reward_step_penalty
            else:
                # Penalty for incorrect answer
                return -self.reward_correct
        
        # Default case: no clear answer
        return 0.0
