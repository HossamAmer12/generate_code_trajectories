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
import textwrap
import requests
import datetime
import traceback


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


# ------------------------------
# Code execution utility
# ------------------------------

CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE
)


def log_execution(pre_text, code, output, log_file = "code_exec.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"[{timestamp}] Executed Code Block\n")
        f.write("-" * 60 + "\n")
        if pre_text:
            f.write("PREVIOUS TEXT (before code block):\n")
            f.write(pre_text + "\n")
            f.write("-" * 60 + "\n")
        f.write("CODE:\n")
        f.write("```python\n")
        f.write(code + "\n")
        f.write("```\n\n")
        f.write("OUTPUT:\n")
        f.write(output + "\n")
        f.write("=" * 60 + "\n\n")

def execute_code_blocks(text, timeout=5, context_chars=1024):
    """
    Finds <code> blocks, executes them, and injects <interpreter> output.
    """
    def run_block(match):
        generated_code = textwrap.dedent(match.group(1)).strip()
        # print("Executing raw code:\n", generated_code)

        # Extract context before this code block
        start_index = match.start()
        pre_text = text[max(0, start_index - context_chars):start_index].strip()

        try:
            # 🔥 send `code` to sandbox here
            sandbox_payload = {"code": generated_code,
            "language": "python"}

            sandbox_response = requests.post(
                "http://localhost:8080/run_code",  # Your sandbox endpoint
                headers={"Content-Type": "application/json"},
                json=sandbox_payload)
            sandbox_result = sandbox_response.json()
            output = sandbox_result.get("run_result", {}).get("stdout", "").strip()
            output = output.strip("\n")
        except Exception as e:
            output = "ERROR:\n" + traceback.format_exc()
        
        # debug
        # 🔒 Log code + output together
        log_execution(pre_text, generated_code, output)

        return match.group(0) + f"\n<interpreter>\n{output}\n</interpreter>\n"
        
    # print("****************************Blocks: ")
    return CODE_BLOCK_RE.sub(run_block, text)

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
                    # "title": f"LeetCode {item.get('question_id', 'Unknown')}",
                    "title": f"{item.get('task_id', 'Unknown')}",
                    "difficulty": item.get("difficulty", "Unknown"),
                    "description": item.get("problem_description", ""),  # ← was "description"
                    "starter_code": item.get("starter_code", ""),        # ← NEW
                    "tags": item.get("tags", []),                         # ← NEW,
                    "testcases": item.get("input_output", []),
                    "original_response": item.get('response', 'Unknown')
                }


                if problem['original_response'] == "Unknown" or problem["original_response"] == "":
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
    
    # This version can be considered
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

    # def format_problem_prompt(self, problem: Dict) -> str:

    #     # question = problem.get("description", "").strip()
    #     original_response = problem.get("original_response", "").strip()
    #     filtered_lines = [line.strip() for line in problem.get('description', '').splitlines() if line.strip()]
    #     question = '\n'.join(filtered_lines)

    #     system_prompt = """You are an expert problem solver who thinks step-by-step and uses executable Python code to verify your reasoning. 
    #     Your task is to take a problem and its basic solution, then enhance the thinking process by:
    #     1. Breaking down the problem step-by-step with detailed reasoning
    #     2. Writing executable Python code snippets to test ideas, explore the problem, and verify solutions
    #     3. Wrapping each code snippet in <code> tags: <code>\n```python\n[your code]\n```\n</code>
    #     4. Including multiple code blocks throughout your thinking as you explore and verify
    #     5. Providing a final, clean solution in the output
    #     6. If you write test cases in your solution, mention that you will do test cases.
    #     7. Your own test and edge cases are in Python script and wrap them up in <code> tags: <code>\n```python\n[your code]\n```\n</code>.

    #     CRITICAL FORMATTING RULES:
    #     - Each code block MUST be wrapped as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
    #     - Code blocks can appear MULTIPLE times in your thinking (typically 2-4 times)
    #     - Each code block should be a complete, executable Python script
    #     - Include necessary imports in each code block
    #     - The thinking should flow naturally with code blocks embedded where you test ideas

    #     Given this thinking text, solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.
    #     Each code snippet is wrapped with <code>\n```python\ncode snippet\n```\n</code> and should be executable. The returned result is wrapped with <interpreter> execution results \\texttt{{</interpreter>}}"""

    #     # system_prompt = """You are an expert problem solver who thinks step-by-step and uses executable Python code to verify your reasoning. 
    #     # Your task is to take a problem and its basic solution, then enhance the thinking process by:
    #     # 1. Breaking down the problem step-by-step with detailed reasoning
    #     # 2. Writing executable Python code snippets to test ideas, explore the problem, and verify solutions
    #     # 3. Wrapping each code snippet in <code> tags: <code>\n```python\n[your code]\n```\n</code>
    #     # 4. Including multiple code blocks throughout your thinking as you explore and verify
    #     # 5. Python code should be executable. The returned result is wrapped with <interpreter> execution results \\texttt{{</interpreter>}}
    #     # 5. Providing a final, clean solution in the output

    #     # CRITICAL FORMATTING RULES:
    #     # - Each code block MUST be wrapped as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
    #     # - Code blocks can appear MULTIPLE times in your thinking (typically 2-4 times)
    #     # - Each code block should be a complete, executable Python script
    #     # - Include necessary imports in each code block
    #     # - The thinking should flow naturally with code blocks embedded where you test ideas

    #     # Given this thinking text, solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox."""

    
    #     # Part 1: Goal
    #     prompt = system_prompt


    #     # Part 2: User Question + Original Thinking Process placeholders
    #     prompt += (
    #         "User Question:\n"
    #         "{question}\n"
    #         "Original Thinking Process (without code interpreter's support):\n"
    #         "<original_thinking_process> {original_response} </original_thinking_process>\n\n"
    #     )

    #     # Part 5: Details section (extended with points 5–7)
    #     prompt += (
    #     "Details:\n"
    #     "1. Outputs in the code snippets must explicitly call the print function.\n"
    #     "2. If the Original Thinking Process does not include an <answer> section at the end, please add it in the Revised Thinking Process:\n"
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

    # def format_problem_prompt(self, problem: Dict) -> str:
    #     """
    #     Format a problem into a prompt that encourages step-by-step thinking
    #     with executable code verification.
        
    #     Args:
    #         problem: Dictionary containing:
    #             - description: Problem statement
    #             - original_response: Initial solution attempt (optional)
        
    #     Returns:
    #         Formatted prompt string
    #     """
    #     # Clean up the problem description
    #     filtered_lines = [
    #         line.strip() 
    #         for line in problem.get('description', '').splitlines() 
    #         if line.strip()
    #     ]
    #     question = '\n'.join(filtered_lines)
    #     original_response = problem.get("original_response", "").strip()

    #     system_prompt = """You are an expert problem solver who uses step-by-step reasoning enhanced by executable Python code.

    # YOUR TASK:
    # Solve the problem using a thinking-first approach where you:
    # 1. Reason through the problem step-by-step in natural language
    # 2. Write small Python code snippets (2-4 throughout your solution) to TEST and VERIFY your ideas
    # 3. Use code to explore edge cases, validate assumptions, and confirm your reasoning
    # 4. Provide a final, complete solution at the end

    # CRITICAL WORKFLOW:
    # - Start by breaking down the problem conceptually (no code yet)
    # - As you develop ideas, write code snippets to TEST them
    # - After each code block, interpret the results and continue reasoning
    # - Build toward the final solution incrementally

    # CODE FORMATTING RULES:
    # - Wrap each code snippet as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
    # - Each code block should be COMPLETE and EXECUTABLE (include imports)
    # - Use print() for all outputs you want to see
    # - Include all necessary imports and define all variables.
    # - Code blocks should appear 2-4 times during your thinking process
    # - Execution results will be wrapped in <interpreter> tags

    # WHEN TO USE CODE:
    # ✓ Testing a small example to understand the problem
    # ✓ Verifying an algorithm idea works correctly
    # ✓ Checking edge cases or special conditions
    # ✓ Validating intermediate steps in your reasoning
    # ✓ Exploring different approaches with simple tests
    # ✗ NOT as your first step - think first!
    # ✗ NOT to replace reasoning - code should verify it

    # STRUCTURE YOUR RESPONSE LIKE THIS:

    # 1. **Understanding the Problem**
    # - Explain what the problem is asking in your own words
    # - Identify key constraints and requirements
    # - Note any edge cases to consider

    # 2. **Initial Approach**
    # - Describe your first idea for solving the problem
    # - Explain the intuition behind this approach

    # 3. **Testing the Idea** (CODE BLOCK 1)
    # <code>
    # ```python
    # # Test with a simple example to verify the approach
    # # Include print statements to see results
    # ```
    # </code>

    # 4. **Analysis of Results**
    # - What did the test reveal?
    # - Does the approach work? Any issues?
    # - What adjustments are needed?

    # 5. **Refined Approach**
    # - Update your strategy based on testing
    # - Address any issues discovered

    # 6. **Edge Case Verification** (CODE BLOCK 2)
    # <code>
    # ```python
    # # Test edge cases and special conditions
    # # Verify the refined approach handles all scenarios
    # ```
    # </code>

    # 7. **Final Solution Development**
    # - Explain the complete algorithm
    # - Walk through the implementation logic

    # 8. **Solution Testing** (CODE BLOCK 3-4)
    # <code>
    # ```python
    # # Implement and test the final solution
    # # Run against provided examples
    # ```
    # </code>

    # 9. **Final Answer**
    # <answer>
    # ```python
    # # Complete, clean solution
    # def solution():
    #     # Implementation
    #     pass

    # # Test cases
    # print('\\boxed{"Final Answer"}')
    # ```
    # </answer>

    # IMPORTANT NOTES:
    # - The code interpreter will execute your Python code in a sandbox environment
    # - Results will be returned wrapped in <interpreter> execution results </interpreter> tags
    # - Always include necessary imports in each code block
    # - Make each code block self-contained and executable
    # - Use descriptive print statements to understand what's happening
    # """

    #     prompt = system_prompt
        
    #     # Add the user's question and context
    #     prompt += f"""

    # USER QUESTION:
    # {question}
    # """

    #     # Add original thinking process if available
    #     if original_response:
    #         prompt += f"""
    # ORIGINAL THINKING PROCESS (without code support):
    # <original_thinking_process>
    # {original_response}
    # </original_thinking_process>
    # """

    #     # Add specific instructions
    #     prompt += """
    # DETAILED INSTRUCTIONS:

    # 1. **Start with Understanding**: Begin by explaining the problem in your own words before writing any code.

    # 2. **Use Code Strategically**: Include 2-4 code blocks throughout your thinking process:
    # - First code block: Test basic understanding with a simple example
    # - Middle code blocks: Verify your approach, test edge cases, explore alternatives
    # - Final code block: Implement and test the complete solution

    # 3. **Outputs**: All code outputs must explicitly use print() statements to be visible.

    # 4. **Incremental Building**: Don't jump straight to the final solution. Build up your understanding through testing.

    # 5. **Analysis**: After each code execution, analyze what you learned and how it affects your approach.

    # 6. **Final Answer Format**: Always end with an <answer> section containing your final, complete solution:

    # <answer>
    # ```python
    # # Your final solution here
    # def solve_problem():
    # # Complete implementation
    # pass

    # # Test with examples
    # # Include test cases to verify correctness
    # print('\\boxed{"Your Final Answer"}')
    # ```
    # </answer>

    # Now, solve the problem step by step with code verification:

    # REVISED THINKING PROCESS (with code interpreter support):
    # """

    #     return prompt

#     def format_problem_prompt(self, problem: Dict) -> str:
#         """
#         Format a problem into a prompt that encourages step-by-step thinking
#         with executable code verification.
        
#         Args:
#             problem: Dictionary containing:
#                 - description: Problem statement
#                 - original_response: Initial solution attempt (optional)
        
#         Returns:
#             Formatted prompt string
#         """
#         # Clean up the problem description
#         filtered_lines = [
#             line.strip() 
#             for line in problem.get('description', '').splitlines() 
#             if line.strip()
#         ]
#         question = '\n'.join(filtered_lines)
#         original_response = problem.get("original_response", "").strip()

#         system_prompt = """You are an expert problem solver who uses step-by-step reasoning enhanced by executable Python code.

# YOUR TASK:
# Solve the problem using a thinking-first approach where you:
# 1. Reason through the problem step-by-step in natural language
# 2. Write small Python code snippets (2-4 throughout your solution) to TEST and VERIFY your ideas
# 3. Use code to explore edge cases, validate assumptions, and confirm your reasoning
# 4. Provide a final, complete solution at the end

# CRITICAL WORKFLOW:
# - Start by breaking down the problem conceptually (no code yet)
# - As you develop ideas, write code snippets to TEST them
# - After each code block, interpret the results and continue reasoning
# - Build toward the final solution incrementally

# CODE FORMATTING RULES:
# - Wrap each code snippet as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
# - Each code block MUST be SELF-CONTAINED and EXECUTABLE:
#   * Include ALL necessary imports at the top of EVERY code block
#   * Define ALL variables and data needed within the code block
#   * Don't reference variables from previous code blocks - redefine them if needed
#   * Test data, input examples, and helper functions must be included in each block
# - Use print() for all outputs you want to see
# - Code blocks should appear 2-4 times during your thinking process
# - Execution results will be wrapped in <interpreter> tags

# CRITICAL: Each code block runs independently in a fresh environment. You MUST include:
# ✓ All imports (e.g., import heapq, from collections import defaultdict)
# ✓ All variable definitions (e.g., test_input = [...])
# ✓ All helper functions used in that block
# ✗ Never assume variables from previous blocks exist

# WHEN TO USE CODE:
# ✓ Testing a small example to understand the problem
# ✓ Verifying an algorithm idea works correctly
# ✓ Checking edge cases or special conditions
# ✓ Validating intermediate steps in your reasoning
# ✓ Exploring different approaches with simple tests
# ✗ NOT as your first step - think first!
# ✗ NOT to replace reasoning - code should verify it

# STRUCTURE YOUR RESPONSE LIKE THIS:

# 1. **Understanding the Problem**
#    - Explain what the problem is asking in your own words
#    - Identify key constraints and requirements
#    - Note any edge cases to consider

# 2. **Initial Approach**
#    - Describe your first idea for solving the problem
#    - Explain the intuition behind this approach

# 3. **Testing the Idea** (CODE BLOCK 1)
#    <code>
#    ```python
#    # IMPORTANT: Include ALL imports and variable definitions
#    # This code block must run independently
   
#    # Example:
#    # import math
#    # test_input = [1, 2, 3]
   
#    # Test with a simple example to verify the approach
#    # Include print statements to see results
#    ```
#    </code>

# 4. **Analysis of Results**
#    - What did the test reveal?
#    - Does the approach work? Any issues?
#    - What adjustments are needed?

# 5. **Refined Approach**
#    - Update your strategy based on testing
#    - Address any issues discovered

# 6. **Edge Case Verification** (CODE BLOCK 2)
#    <code>
#    ```python
#    # IMPORTANT: This is a new code block - include ALL imports and definitions again
#    # Don't assume anything from the previous code block exists
   
#    # Test edge cases and special conditions
#    # Verify the refined approach handles all scenarios
#    ```
#    </code>

# 7. **Final Solution Development**
#    - Explain the complete algorithm
#    - Walk through the implementation logic

# 8. **Solution Testing** (CODE BLOCK 3-4)
#    <code>
#    ```python
#    # IMPORTANT: Fresh code block - redefine everything needed
#    # Include: imports, helper functions, test data
   
#    # Implement and test the final solution
#    # Run against provided examples
#    ```
#    </code>

# 9. **Final Answer**
#    <answer>
#    ```python
#    # Complete, clean solution
#    def solution():
#        # Implementation
#        pass
   
#    # Test cases
#    print('\\boxed{"Final Answer"}')
#    ```
#    </answer>

# IMPORTANT NOTES:
# - The code interpreter will execute your Python code in a sandbox environment
# - Results will be returned wrapped in <interpreter> execution results </interpreter> tags
# - Always include necessary imports in each code block
# - Make each code block self-contained and executable
# - Use descriptive print statements to understand what's happening
# """

#         prompt = system_prompt
        
#         # Add the user's question and context
#         prompt += f"""

# USER QUESTION:
# {question}
# """

#         # Add original thinking process if available
#         if original_response:
#             prompt += f"""
# ORIGINAL THINKING PROCESS (without code support):
# <original_thinking_process>
# {original_response}
# </original_thinking_process>
# """

#         # Add specific instructions
#         prompt += """
# DETAILED INSTRUCTIONS:

# 1. **Start with Understanding**: Begin by explaining the problem in your own words before writing any code.

# 2. **Use Code Strategically**: Include 2-4 code blocks throughout your thinking process:
#    - First code block: Test basic understanding with a simple example
#    - Middle code blocks: Verify your approach, test edge cases, explore alternatives
#    - Final code block: Implement and test the complete solution

# 3. **Self-Contained Code Blocks** (CRITICAL):
#    - Each code block runs in a FRESH, INDEPENDENT environment
#    - ALWAYS include at the start of each code block:
#      * ALL imports needed (import math, from collections import deque, etc.)
#      * ALL test data/input variables
#      * ALL helper functions that will be called
#    - Example of a proper code block:
#      ```python
#      # Correct - Self-contained
#      import heapq
#      from collections import defaultdict
     
#      def helper_func(x):
#          return x * 2
     
#      test_data = [1, 2, 3]
#      result = helper_func(test_data[0])
#      print(result)
#      ```
#    - Example of WRONG code block:
#      ```python
#      # WRONG - Missing imports and assumes 'test_data' exists from before
#      result = helper_func(test_data[0])  # ERROR: helper_func and test_data undefined!
#      print(result)
#      ```

# 4. **Outputs**: All code outputs must explicitly use print() statements to be visible.

# 5. **Incremental Building**: Don't jump straight to the final solution. Build up your understanding through testing.

# 6. **Analysis**: After each code execution, analyze what you learned and how it affects your approach.

# 6. **Final Answer Format**: Always end with an <answer> section containing your final, complete solution:

# <answer>
# ```python
# # Your final solution here
# def solve_problem():
#     # Complete implementation
#     pass

# # Test with examples
# # Include test cases to verify correctness
# print('\\boxed{"Your Final Answer"}')
# ```
# </answer>

# CRITICAL REMINDERS:
# - Every code block is executed independently - include ALL imports and variable definitions
# - If you need data from the problem, redefine it in each code block
# - Common mistakes to avoid:
#   * Forgetting imports like 'import heapq', 'from collections import defaultdict'
#   * Using variables defined in previous code blocks (they won't exist!)
#   * Calling functions not defined in the current code block

# Now, solve the problem step by step with code verification:

# REVISED THINKING PROCESS (with code interpreter support):
# """

#         return prompt


    def format_problem_prompt(self, problem: Dict) -> str:
        """
        Format a problem into a prompt that encourages step-by-step thinking
        with executable code verification.
        
        Args:
            problem: Dictionary containing:
                - description: Problem statement
                - original_response: Initial solution attempt (optional)
        
        Returns:
            Formatted prompt string
        """
        # Clean up the problem description
        filtered_lines = [
            line.strip() 
            for line in problem.get('description', '').splitlines() 
            if line.strip()
        ]
        question = '\n'.join(filtered_lines)
        original_response = problem.get("original_response", "").strip()

        system_prompt = """You are an expert problem solver who uses step-by-step reasoning enhanced by executable Python code.

YOUR TASK:
Solve the problem using a thinking-first approach where you:
1. Reason through the problem step-by-step in natural language
2. Write small Python code snippets (2-4 throughout your solution) to TEST and VERIFY your ideas
3. Use code to explore edge cases, validate assumptions, and confirm your reasoning
4. Provide a final, complete solution at the end

CRITICAL WORKFLOW:
- Start by breaking down the problem conceptually (no code yet)
- As you develop ideas, write code snippets to TEST them
- After each code block, interpret the results and continue reasoning
- Build toward the final solution incrementally

CODE FORMATTING RULES:
- Wrap each code snippet as: <code>\n```python\nYOUR_CODE_HERE\n```\n</code>
- Each code block MUST be SELF-CONTAINED and EXECUTABLE:
  * Include ALL necessary imports at the top of EVERY code block
  * Define ALL variables and data needed within the code block
  * Don't reference variables from previous code blocks - redefine them if needed
  * Test data, input examples, and helper functions must be included in each block
- Use print() for all outputs you want to see
- Code blocks should appear 2-4 times during your thinking process
- Execution results will be wrapped in <interpreter> tags

CRITICAL: Each code block runs independently in a fresh environment. You MUST include:
✓ All imports (e.g., import heapq, from collections import defaultdict)
✓ All variable definitions (e.g., test_input = [...])
✓ All helper functions used in that block
✗ Never assume variables from previous blocks exist

WHEN TO USE CODE:
✓ Testing a small example to understand the problem
✓ Verifying an algorithm idea works correctly
✓ Checking edge cases or special conditions
✓ Validating intermediate steps in your reasoning
✓ Exploring different approaches with simple tests
✗ NOT as your first step - think first!
✗ NOT to replace reasoning - code should verify it

STRUCTURE YOUR RESPONSE LIKE THIS:

1. **Understanding the Problem**
   - Explain what the problem is asking in your own words
   - Identify key constraints and requirements
   - Note any edge cases to consider

2. **Initial Approach**
   - Describe your first idea for solving the problem
   - Explain the intuition behind this approach

3. **Testing the Idea** (CODE BLOCK 1)
   <code>
   ```python
   # IMPORTANT: Include ALL imports and variable definitions
   # This code block must run independently
   
   # Example:
   # import math
   # test_input = [1, 2, 3]
   
   # Test with a simple example to verify the approach
   # Include print statements to see results
   ```
   </code>

4. **Analysis of Results**
   - What did the test reveal?
   - Does the approach work? Any issues?
   - What adjustments are needed?

5. **Refined Approach**
   - Update your strategy based on testing
   - Address any issues discovered

6. **Edge Case Verification** (CODE BLOCK 2)
   <code>
   ```python
   # IMPORTANT: This is a new code block - include ALL imports and definitions again
   # Don't assume anything from the previous code block exists
   
   # Test edge cases and special conditions
   # Verify the refined approach handles all scenarios
   ```
   </code>

7. **Final Solution Development**
   - Explain the complete algorithm
   - Walk through the implementation logic

8. **Solution Testing** (CODE BLOCK 3-4)
   <code>
   ```python
   # IMPORTANT: Fresh code block - redefine everything needed
   # Include: imports, helper functions, test data
   
   # Implement and test the final solution
   # Run against provided examples
   ```
   </code>

9. **Conclusion**
   - Summarize your findings
   - Confirm the approach works for all cases
   - Note any important insights or edge cases discovered

IMPORTANT NOTES:
- The code interpreter will execute your Python code in a sandbox environment
- Results will be returned wrapped in <interpreter> execution results </interpreter> tags
- Always include necessary imports in each code block
- Make each code block self-contained and executable
- Use descriptive print statements to understand what's happening
"""

        prompt = system_prompt
        
        # Add the user's question and context
        prompt += f"""

USER QUESTION:
{question}
"""

        # Add original thinking process if available
        if original_response:
            prompt += f"""
ORIGINAL THINKING PROCESS (without code support):
<original_thinking_process>
{original_response}
</original_thinking_process>
"""

        # Add specific instructions
        prompt += """
DETAILED INSTRUCTIONS:

1. **Start with Understanding**: Begin by explaining the problem in your own words before writing any code.

2. **Use Code Strategically**: Include 2-4 code blocks throughout your thinking process:
   - First code block: Test basic understanding with a simple example
   - Middle code blocks: Verify your approach, test edge cases, explore alternatives
   - Final code block: Implement and test the complete solution

3. **Self-Contained Code Blocks** (CRITICAL):
   - Each code block runs in a FRESH, INDEPENDENT environment
   - ALWAYS include at the start of each code block:
     * ALL imports needed (import math, from collections import deque, etc.)
     * ALL test data/input variables
     * ALL helper functions that will be called
   - Example of a proper code block:
     ```python
     # Correct - Self-contained
     import heapq
     from collections import defaultdict
     
     def helper_func(x):
         return x * 2
     
     test_data = [1, 2, 3]
     result = helper_func(test_data[0])
     print(result)
     ```
   - Example of WRONG code block:
     ```python
     # WRONG - Missing imports and assumes 'test_data' exists from before
     result = helper_func(test_data[0])  # ERROR: helper_func and test_data undefined!
     print(result)
     ```

4. **Outputs**: All code outputs must explicitly use print() statements to be visible.

5. **Incremental Building**: Build up your understanding through testing and verification.

6. **Analysis**: After each code execution, analyze what you learned and how it affects your approach.

CRITICAL REMINDERS:
- Every code block is executed independently - include ALL imports and variable definitions
- If you need data from the problem, redefine it in each code block
- Common mistakes to avoid:
  * Forgetting imports like 'import heapq', 'from collections import defaultdict'
  * Using variables defined in previous code blocks (they won't exist!)
  * Calling functions not defined in the current code block

Now, solve the problem step by step with code verification:

REVISED THINKING PROCESS (with code interpreter support):
"""

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

        # debug
        print(initial_prompt)

        try:
            # Turn 1: Get initial solution
            # print("[Turn 1] Generating initial solution...")
            messages = [{'role': 'user', 'content': initial_prompt}]
            
            responses = []
            for response in self.agent.run(messages=messages):
                responses.append(response)          
            
            
            # Extract assistant's response
            assistant_response = responses[-1][-1]['content']

            # Execute code blocks
            assistant_response = execute_code_blocks(assistant_response)


            # debug
            print("**************** HM RESPONSE")
            print(assistant_response)

            # print("Hossam after exectution", assistant_response)

            if assistant_response:
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

                trajectory["final_solution"] = current_python_code
                
                print(f"Response length: {len(assistant_response)} chars")
                trajectory["success"] = True
            else:
                trajectory["final_solution"] = assistant_response
                trajectory["success"] = True
            
            print("✓ Trajectory generated successfully")
            
        except Exception as e:
            print(f"✗ Error generating trajectory: {e}")
            trajectory["error"] = str(e)
            trajectory["success"] = False
        
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
            
            # Leave this part for debugging
            # debug
            # if problem["title"] != "shortest-distance-after-road-addition-queries-ii":
            #     trajectory = {
            #     "problem_id": problem.get("id", "unknown"),
            #     "problem_title": problem["title"],
            #     "difficulty": problem.get("difficulty", "Unknown"),
            #     "conversation": [],
            #     # "final_solution": current_python_code,
            #     "success": False
            # }
            # else:
            #     trajectory = self.generate_trajectory(problem)

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
    # MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Small model for testing
    # MODEL_NAME = "Qwen/Qwen3-1.7B"  # Small model for testing
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Small model for testing
    # MAX_SAMPLES = 10  # Number of problems to process
    # MAX_SAMPLES = 5  # Number of problems to process
    # MAX_SAMPLES = 2  # Number of problems to process

    MAX_SAMPLES = 1500
    OUTPUT_DIR = "./leetcode_trajectories/"
    
    # clear the log file
    with open("code_exec.txt", "w") as f:
        pass

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
    problems = generator.load_leetcode_dataset(split="train", max_samples=MAX_SAMPLES)
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