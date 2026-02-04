"""
Generate LeetCode trajectories using Qwen-Agent with local Qwen-1.5B model
This script generates multi-turn interaction trajectories for LeetCode problems
Uses local model inference - no API keys or Docker required
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import time
import re

# Install required packages if needed
def install_requirements():
    """Install required packages"""
    import subprocess
    packages = [
        "transformers",
        "torch",
        "qwen-agent",
        "accelerate",
        "datasets"
    ]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package, "--break-system-packages"])

# Uncomment if you need to install packages
# install_requirements()

from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model


class LeetCodeTrajectoryGenerator:
    """Generate trajectories for LeetCode problems using Qwen-Agent with local model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_local: bool = True):
        """
        Initialize the trajectory generator
        
        Args:
            model_name: HuggingFace model name
            use_local: Use local model instead of API
        """
        self.model_name = model_name
        
        if use_local:
            # Configure for local model using local server
            llm_cfg = {
                'model': model_name,
                'model_server': 'http://localhost:8000/v1',  # base_url, also known as api_base
                'api_key': 'EMPTY',
                'generate_cfg': {
                    'top_p': 0.8,
                    'temperature': 0.7,
                    # 'max_tokens': 4096,
                    'max_tokens': 2048,
                }
            }
        else:
            # Configure for DashScope API (requires API key)
            llm_cfg = {
                'model': model_name,
                'model_server': 'dashscope',
                'generate_cfg': {
                    'top_p': 0.8,
                    'temperature': 0.7,
                    'max_tokens': 2048,
                }
            }
        
        # Create assistant WITHOUT code interpreter to avoid Docker requirement
        # For code execution, we'll handle it separately if needed
        print(f"Initializing agent with local model: {model_name}")
        self.agent = Assistant(
            llm=llm_cfg,
            name='LeetCode Solver',
            description='An AI assistant that solves LeetCode problems',
            function_list=[]  # Empty list - no tools to avoid Docker
        )
        print("Agent initialized successfully!")
        
    def load_leetcode_dataset(self, split: str = "test", max_samples: int = 100):
        """
        Load LeetCode dataset
        
        Args:
            split: Dataset split (train/test/validation)
            max_samples: Maximum number of samples to process
        
        Returns:
            List of LeetCode problems
        """
        # Try to load from HuggingFace datasets
        try:
            from datasets import load_dataset
            print(f"Loading dataset from HuggingFace...")
            
            # Try different dataset sources
            try:
                dataset = load_dataset("newfacade/LeetCodeDataset", split=split)
                print(f"Loaded from newfacade/LeetCodeDataset")
            except:
                try:
                    dataset = load_dataset("greengerong/leetcode", split=split)
                    print(f"Loaded from greengerong/leetcode")
                except:
                    print("Could not load from HuggingFace, using sample problems")
                    return self._get_sample_problems()[:max_samples]
            
            problems = []
            for item in list(dataset)[:max_samples]:
                # Normalize the problem format
                # Now correctly maps newfacade/LeetCodeDataset fields:
                problem = {
                    "id": item.get("question_id") or item.get("task_id"),
                    "title": f"LeetCode {item.get('question_id', 'Unknown')}",
                    "difficulty": item.get("difficulty", "Unknown"),
                    "description": item.get("problem_description", ""),  # ← was "description"
                    "starter_code": item.get("starter_code", ""),        # ← NEW
                    "tags": item.get("tags", []),                         # ← NEW,
                    "testcases": item.get("input_output", []),
                    "original_response": item.get('response', 'Unknown')
                }

                if problem['original_response'] == "Unknown":
                    continue

                problems.append(problem)
            
            return problems
            
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Using sample problems instead")
            return self._get_sample_problems()[:max_samples]
    
    def _get_sample_problems(self):
        """Get sample LeetCode problems as fallback"""
        return [
            {
                "id": 1,
                "title": "Two Sum",
                "difficulty": "Easy",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "examples": [
                    {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"}
                ],
                "constraints": ["2 <= nums.length <= 10^4"],
            },
            {
                "id": 217,
                "title": "Contains Duplicate",
                "difficulty": "Easy",
                "description": "Given an integer array nums, return true if any value appears at least twice in the array.",
                "examples": [
                    {"input": "nums = [1,2,3,1]", "output": "true"}
                ],
                "constraints": ["1 <= nums.length <= 10^5"],
            },
            {
                "id": 242,
                "title": "Valid Anagram",
                "difficulty": "Easy",
                "description": "Given two strings s and t, return true if t is an anagram of s.",
                "examples": [
                    {"input": 's = "anagram", t = "nagaram"', "output": "true"}
                ],
                "constraints": ["1 <= s.length, t.length <= 5 * 10^4"],
            }
        ]
    
    # def format_problem_prompt(self, problem: Dict) -> str:
    #     """
    #     Format a problem into the required code-generation prompt template.
    #     """

    #     prompt = (
    #         "You will be given a question (problem specification) and will generate "
    #         "a correct Python program that matches the specification and passes all tests.\n\n"
    #     )

    #     # Problem specification
    #     prompt += "Problem:\n"

    #     filtered_lines = [line.strip() for line in problem.get('description', '').splitlines() if line.strip()]
    #     result_string = '\n'.join(filtered_lines)
    #     prompt += f"{result_string}\n\n"
    #     # prompt += f"{problem.get('description', '').strip()}\n\n"

    #     # Public examples
    #     if problem.get("examples"):
    #         prompt += "Public Examples:\n"
    #         prompt += (
    #             "Here are some input and output examples of the expected code:\n"
    #         )

    #         for example in problem["examples"]:
    #             if isinstance(example, dict):
    #                 inp = example.get("input", "")
    #                 out = example.get("output", "")
    #                 prompt += f"Input: {inp}\n"
    #                 prompt += f"Output: {out}\n"
    #             else:
    #                 prompt += f"{example}\n"

    #         prompt += "\n"

    #     # Note section
    #     prompt += (
    #         "Note:\n"
    #         "You should first analyze the problem and form a high-level solution strategy, "
    #         "then utilize the tools to help you solve the problem.\n\n"
    #     )

    #     # Instruction section
    #     prompt += (
    #         "Instruction:\n"
    #         "Read the inputs from stdin, solve the problem, and write the answer to stdout "
    #         "(do not directly test on the sample inputs).\n"
    #         "Enclose your code within the delimiters shown below.\n"
    #         "Ensure that when the Python program runs, it correctly reads inputs, "
    #         "executes the algorithm, and writes output to stdout. The execution output format is:\n <interpreter>\n[execution output]\n</interpreter> \n\n"
    #     )

    #     # Submit section
    #     prompt += (
    #         "Submit:\n"
    #         "Before submitting your code, you can utilize tools to check its correctness.\n"
    #         "Once you make sure the current code is correct, do not call the tools again "
    #         "and submit your code within the following Python code block:\n\n"
    #         "```python\n"
    #         "# YOUR CODE HERE\n"
    #         "```\n"
    #     )

    #     return prompt

    # def format_problem_prompt(self, problem: Dict) -> str:

    #     # question = problem.get("description", "").strip()
    #     original_response = problem.get("original_response", "").strip()
    #     filtered_lines = [line.strip() for line in problem.get('description', '').splitlines() if line.strip()]
    #     question = '\n'.join(filtered_lines)

    #     # Part 1
    #     prompt = (
    #     "You are a helpful AI assistant. Initially, when solving a question, you would need to think step by step, without the ability to use\n"
    #     "code for calculation. Now, you have the capability to write code to use the code interpreter for calculation. The code will be\n"
    #     "executed by a sandbox, and the result can be returned to enhance your reasoning process. You can now leverage code to enhance\n"
    #     "your calculation while still maintaining the reasoning process.\n")

    #     # Part 2
    #     prompt += (
    #         "The thinking process can have multiple code snippets. Each code snippet is wrapped with:\n"
    #         "<code>\n"
    #         "```python\n"
    #         "code snippet\n"
    #         "```\n"
    #         "</code>, and should be executable. The returned result is wrapped with <interpreter> execution results \\texttt{{</interpreter>}}.\n\n"
    #     )

    #     # Part 3
    #     # prompt += (
    #     # "Goal:\n"
    #     # "Modify the original thinking process to make it more accurate by replacing manual calculation steps that can benefit from code\n"
    #     # "execution with the corresponding code snippets and their interpreter's execution results. The core reasoning logic from the original\n"
    #     # "thinking process, including any unsuccessful attempts, should remain unchanged. You should only replace the necessary manual\n"
    #     # "calculation steps with code and interpreter's execution results, without altering the rest tokens of the thinking process. Wrap the\n"
    #     # "revised thinking process within <revised_thinking_process> and </revised_thinking_process>}.\n\n")

    #     # Part 3: Goal
    #     prompt += (
    #         "Goal:\n"
    #         "Modify the original thinking process to make it more accurate by replacing manual calculation steps that can benefit from code\n"
    #         "execution with the corresponding code snippets and their interpreter's execution results. The core reasoning logic from the original\n"
    #         "thinking process, including any unsuccessful attempts, should remain unchanged. You should only replace the necessary manual\n"
    #         "calculation steps with code and interpreter's execution results, without altering the rest tokens of the thinking process. Wrap the\n"
    #         "revised thinking process within <revised_thinking_process> and </revised_thinking_process>}}.\n\n"
    #     )


    #     # Part 4: User Question + Original Thinking Process placeholders
    #     prompt += (
    #         "User Question:\n"
    #         "{question}\n"
    #         "Original Thinking Process (without code interpreter’s support):\n"
    #         "<original_thinking_process> {original_response} </original_thinking_process>\n\n"
    #     )

    #     # Part 5: Details section (extended with points 5–7)
    #     prompt += (
    #     "Details:\n"
    #     "1. Identify sections where code execution could speed up the reasoning process or make the calculation more accurate.\n"
    #     "2. Replace the manual calculation steps with code snippets and the corresponding interpreter's execution results.\n"
    #     "3. Keep the logical flow of the reasoning process intact, including any failed exploration attempts that were part of the ini tial\n"
    #     "process.\n"
    #     "4. The code snippets should be complete scripts, including necessary imports, and should not contain markdown symbols like\n"
    #     "<code>\n"
    #     "```python\n"
    #     "code snippet\n"
    #     "```\n"
    #     "</code>.\n"
    #     "5. Outputs in the code snippets must explicitly call the print function.\n"
    #     "6. Execution results should match the model's output exactly, with no extra or missing tokens.\n"
    #     "7. If the Original Thinking Process does not include an <answer> section at the end, please add it in the Revised Thinking Process:\n"
    #     "<answer>\n"
    #     "```python\n"
    #     "# YOUR ANSWER HERE\n"
    #     "print('\\boxed{\\'The final answer goes here.\\'}')\n"
    #     "```\n"
    #     "</answer>\n\n"
    #     )

    #     # Part 6: Revised Thinking Process placeholder
    #     prompt += "Revised Thinking Process (With code interpreter's support):\n"

    #     # Replace the placeholders with actual values
    #     # prompt = prompt.format(question=question, original_response=original_response)
    #     prompt = f"{prompt}"
    #     prompt = prompt.replace("{question}", question).replace("{original_response}", original_response)

    #     return prompt

    def format_problem_prompt(self, problem: Dict) -> str:

        # question = problem.get("description", "").strip()
        original_response = problem.get("original_response", "").strip()
        filtered_lines = [line.strip() for line in problem.get('description', '').splitlines() if line.strip()]
        question = '\n'.join(filtered_lines)

        # system_prompt = """You are an expert problem solver who thinks step-by-step and uses executable Python code to verify your reasoning. 
        # Your task is to take a problem and its basic solution, then enhance the thinking process by:
        # 1. Breaking down the problem step-by-step with detailed reasoning
        # 2. Writing executable Python code snippets to test ideas, explore the problem, and verify solutions
        # 3. Wrapping each code snippet in <code> tags: <code>\n```python\n[your code]\n```\n</code>
        # 4. Including multiple code blocks throughout your thinking as you explore and verify
        # 5. Providing a final, clean solution in the output

        # CRITICAL FORMATTING RULES:
        # - Each code block MUST be wrapped as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
        # - Code blocks can appear MULTIPLE times in your thinking (typically 2-4 times)
        # - Each code block should be a complete, executable Python script
        # - Include necessary imports in each code block
        # - The thinking should flow naturally with code blocks embedded where you test ideas

        # Given this thinking text, solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.
        # Each code snippet is wrapped with <code>\n```python\ncode snippet\n```\n</code> and should be executable. The returned result is wrapped with <interpreter> execution results \\texttt{{</interpreter>}}"""

        system_prompt = """You are an expert problem solver who thinks step-by-step and uses executable Python code to verify your reasoning. 
        Your task is to take a problem and its basic solution, then enhance the thinking process by:
        1. Breaking down the problem step-by-step with detailed reasoning
        2. Writing executable Python code snippets to test ideas, explore the problem, and reflect on solutions
        3. Wrapping each code snippet in <code> tags: <code>\n```python\n[your code]\n```\n</code>
        4. Including multiple code blocks throughout your thinking as you explore and verify
        5. Python code should be executable. The returned result is wrapped with <interpreter> execution results \\texttt{{</interpreter>}}
        5. Providing a final, clean solution in the output

        CRITICAL FORMATTING RULES:
        - Each code block MUST be wrapped as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
        - Code blocks can appear MULTIPLE times in your thinking (typically 2-4 times)
        - Each code block should be a complete, executable Python script
        - Include necessary imports in each code block
        - The thinking should flow naturally with code blocks embedded where you test ideas

        Given this thinking text, solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox."""

    
        # Part 1: Goal
        prompt = system_prompt


        # Part 2: User Question + Original Thinking Process placeholders
        prompt += (
            "User Question:\n"
            "{question}\n"
            "Original Thinking Process (without code interpreter's support):\n"
            "<original_thinking_process> {original_response} </original_thinking_process>\n\n"
        )

        # Part 5: Details section (extended with points 5–7)
        prompt += (
        "Details:\n"
        "1. Outputs in the code snippets must explicitly call the print function.\n"
        "2. If the Original Thinking Process does not include an <answer> section at the end, please add it in the Revised Thinking Process:\n"
        "<answer>\n"
        "```python\n"
        "# YOUR ANSWER HERE\n"
        "print('\\boxed{\\'The final answer goes here.\\'}')\n"
        "```\n"
        "</answer>\n\n"
        )

        # Part 6: Revised Thinking Process placeholder
        prompt += "Revised Thinking Process (With code interpreter's support):\n"

        # Replace the placeholders with actual values
        # prompt = prompt.format(question=question, original_response=original_response)
        prompt = f"{prompt}"
        prompt = prompt.replace("{question}", question).replace("{original_response}", original_response)

        return prompt


    def generate_trajectory(self, problem: Dict) -> Dict:
        """
        Generate a multi-turn interaction trajectory for a problem
        
        Args:
            problem: LeetCode problem dictionary
            
        Returns:
            Trajectory with conversation history
        """
        print(f"\n{'='*60}")
        print(f"Processing: {problem['title']} ({problem.get('difficulty', 'Unknown')})")
        print(f"{'='*60}")
        
        match = re.search(r"```python\s*(.*?)\s*```", problem["original_response"], re.DOTALL)
        current_python_code = match.group(1) if match else None

        trajectory = {
            "problem_id": problem.get("id", "unknown"),
            "problem_title": problem["title"],
            "difficulty": problem.get("difficulty", "Unknown"),
            "conversation": [],
            "final_solution": current_python_code,
            "success": False
        }
        
        # Initial prompt
        initial_prompt = self.format_problem_prompt(problem)

        print(initial_prompt)

        try:
            # Turn 1: Get initial solution
            print("[Turn 1] Generating initial solution...")
            messages = [{'role': 'user', 'content': initial_prompt}]
            
            responses = []
            for response in self.agent.run(messages=messages):
                responses.append(response)          
            
            if not responses:
                raise Exception("No response from agent")
            
            
            # Extract assistant's response
            assistant_response = responses[-1][-1]['content']

              # print
            print("Response: ", assistant_response)
            print("Exitingg....")
            exit(0)
            
            trajectory["conversation"].append({
                "turn": 1,
                "role": "user",
                "content": initial_prompt
            })
            trajectory["conversation"].append({
                "turn": 1,
                "role": "assistant",
                "content": assistant_response
            })
            
            print(f"Response length: {len(assistant_response)} chars")
            
            # Turn 2: Ask for optimization
            print("[Turn 2] Requesting optimization...")
            follow_up = """Great! Now please:
1. Optimize this solution if possible
2. Provide detailed time and space complexity analysis
3. Suggest any alternative approaches"""
            
            messages.append({'role': 'assistant', 'content': assistant_response})
            messages.append({'role': 'user', 'content': follow_up})
            
            responses = []
            for response in self.agent.run(messages=messages):
                responses.append(response)
            
            if responses:
                refined_response = responses[-1][-1]['content']
                trajectory["conversation"].append({
                    "turn": 2,
                    "role": "user",
                    "content": follow_up
                })
                trajectory["conversation"].append({
                    "turn": 2,
                    "role": "assistant",
                    "content": refined_response
                })
                trajectory["final_solution"] = refined_response
                trajectory["success"] = True
                
                print(f"Optimization length: {len(refined_response)} chars")
            else:
                trajectory["final_solution"] = assistant_response
                trajectory["success"] = True
            
            print("✓ Trajectory generated successfully")
            
        except Exception as e:
            print(f"✗ Error generating trajectory: {e}")
            trajectory["error"] = str(e)
        
        return trajectory
    
    def generate_trajectories_batch(
        self, 
        problems: List[Dict], 
        output_path: str = "trajectories.jsonl"
    ):
        """
        Generate trajectories for a batch of problems
        
        Args:
            problems: List of LeetCode problems
            output_path: Path to save trajectories
        """
        trajectories = []
        
        # Clear output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{'#'*60}")
            print(f"Problem {i}/{len(problems)}")
            print(f"{'#'*60}")
            
            trajectory = self.generate_trajectory(problem)
            trajectories.append(trajectory)
            
            # Save incrementally
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory, ensure_ascii=False) + '\n')
            
            # Small delay between problems
            time.sleep(0.5)
        
        # Print summary
        successful = sum(1 for t in trajectories if t.get("success", False))
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {len(trajectories)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(trajectories) - successful}")
        print(f"Success rate: {successful/len(trajectories)*100:.1f}%")
        print(f"Output file: {output_path}")
        print(f"{'='*60}")
        
        return trajectories


def main():
    """Main execution function"""
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for testing
    # MODEL_NAME = "Qwen/Qwen3-1.7B"  # Small model for testing
    MAX_SAMPLES = 10  # Number of problems to process
    OUTPUT_DIR = "./leetcode_trajectories/"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("LeetCode Trajectory Generator with Qwen-Agent")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max samples: {MAX_SAMPLES}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("="*60)
    
    # Initialize generator
    print(f"\nInitializing generator...")
    generator = LeetCodeTrajectoryGenerator(
        model_name=MODEL_NAME,
        use_local=True  # Use local model, not API
    )
    
    # Load LeetCode test split
    print("\nLoading LeetCode dataset (test split)...")
    problems = generator.load_leetcode_dataset(split="test", max_samples=MAX_SAMPLES)
    print(f"Loaded {len(problems)} problems")
    
    # Show first problem as example
    if problems:
        print(f"\nFirst problem example:")
        print(f"  ID: {problems[0].get('id')}")
        print(f"  Title: {problems[0].get('title')}")
        print(f"  Difficulty: {problems[0].get('difficulty')}")
    
    # Generate trajectories
    output_path = os.path.join(OUTPUT_DIR, "test_trajectories.jsonl")
    print(f"\nStarting trajectory generation...")
    print(f"This may take a while depending on model size and hardware...")
    
    trajectories = generator.generate_trajectories_batch(
        problems=problems,
        output_path=output_path
    )
    
    print(f"\n✓ All done! Check {output_path} for results.")


if __name__ == "__main__":
    main()