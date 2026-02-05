"""
Generate LeetCode trajectories using Qwen-Agent with local Qwen-1.5B model
This script generates multi-turn interaction trajectories for LeetCode problems
Uses local model inference - no API keys or Docker required
Saves output in parquet format matching the message structure
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
import pandas as pd


# Install required packages if needed
def install_requirements():
    """Install required packages"""
    import subprocess
    packages = [
        "transformers",
        "torch",
        "qwen-agent",
        "accelerate",
        "datasets",
        "pyarrow",
        "pandas"
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
        
        # 🔒 Log code + output together
        log_execution(pre_text, generated_code, output)

        return match.group(0) + f"\n<interpreter>\n{output}\n</interpreter>\n"
        
    return CODE_BLOCK_RE.sub(run_block, text)


def convert_trajectory_to_messages(trajectory: Dict) -> Dict:
    """
    Convert trajectory format to messages format matching the parquet structure.
    
    Args:
        trajectory: Dictionary with conversation history
        
    Returns:
        Dictionary with 'messages' and 'tools' keys
    """
    messages = []
    
    # Extract tool calls from assistant responses
    def extract_tool_calls(content: str) -> List[Dict]:
        """Extract code blocks as tool calls"""
        tool_calls = []
        
        # Find all code blocks
        code_pattern = r'<code>\s*```python\s*(.*?)```\s*</code>'
        matches = re.finditer(code_pattern, content, re.DOTALL)
        
        for match in matches:
            code = match.group(1).strip()
            tool_calls.append({
                "function": {
                    "arguments": {"code": code},
                    "name": "code_interpreter"
                },
                "type": "function"
            })
        
        return tool_calls if tool_calls else None
    
    def extract_tool_results(content: str) -> List[str]:
        """Extract interpreter outputs"""
        results = []
        interpreter_pattern = r'<interpreter>\s*(.*?)\s*</interpreter>'
        matches = re.finditer(interpreter_pattern, content, re.DOTALL)
        
        for match in matches:
            results.append(match.group(1).strip())
        
        return results
    
    # Process conversation turns
    for turn_data in trajectory.get("conversation", []):
        role = turn_data.get("role")
        content = turn_data.get("content", "")
        
        if role == "user":
            # User message
            messages.append({
                "content": content,
                "role": "user",
                "tool_calls": None
            })
        
        elif role == "assistant":
            # Extract tool calls from the content
            tool_calls = extract_tool_calls(content)
            
            # Remove code blocks and interpreter outputs from content for display
            clean_content = re.sub(r'<code>\s*```python\s*.*?```\s*</code>', '', content, flags=re.DOTALL)
            clean_content = re.sub(r'<interpreter>\s*.*?\s*</interpreter>', '', clean_content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            
            # Add assistant message with tool calls
            messages.append({
                "content": clean_content,
                "role": "assistant",
                "tool_calls": tool_calls
            })
            
            # If there were tool calls, add tool result messages
            tool_results = extract_tool_results(content)
            for result in tool_results:
                # Format the result to match the expected structure
                formatted_result = f"execute_result:\n\n```\n{result}\n```"
                
                messages.append({
                    "content": formatted_result,
                    "role": "tool",
                    "tool_calls": None
                })
    
    # Define the tools structure
    tools = [{
        "function": {
            "description": "A tool for executing code.",
            "name": "code_interpreter",
            "parameters": {
                "properties": {
                    "code": {
                        "description": "The code to execute.",
                        "type": "string"
                    }
                },
                "required": ["code"],
                "type": "object"
            }
        },
        "type": "function"
    }]
    
    return {
        "messages": messages,
        "tools": tools
    }


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
                problem = {
                    "id": item.get("question_id") or item.get("task_id"),
                    "title": f"{item.get('task_id', 'Unknown')}",
                    "difficulty": item.get("difficulty", "Unknown"),
                    "description": item.get("problem_description", ""),
                    "starter_code": item.get("starter_code", ""),
                    "tags": item.get("tags", []),
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
        ]

    def format_problem_prompt(self, problem: Dict) -> str:
        """
        Format a problem into a prompt that encourages step-by-step thinking
        with executable code verification.
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

        try:
            # Turn 1: Get initial solution
            messages = [{'role': 'user', 'content': initial_prompt}]
            
            responses = []
            for response in self.agent.run(messages=messages):
                responses.append(response)          
            
            # Extract assistant's response
            assistant_response = responses[-1][-1]['content']

            # Execute code blocks
            assistant_response = execute_code_blocks(assistant_response)

            print("**************** RESPONSE")
            print(assistant_response)

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
        output_path: str = "trajectories.parquet"
    ):
        """
        Generate trajectories for a batch of problems and save as parquet
        
        Args:
            problems: List of LeetCode problems
            output_path: Path to save trajectories (parquet format)
        """
        all_data = []
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{'#'*60}")
            print(f"Problem {i}/{len(problems)}")
            print(f"{'#'*60}")
            
            trajectory = self.generate_trajectory(problem)
            
            # Convert trajectory to messages format
            if trajectory.get("success", False):
                messages_data = convert_trajectory_to_messages(trajectory)
                all_data.append(messages_data)
            
            # Small delay between problems
            time.sleep(0.5)
        
        # Convert to DataFrame and save as parquet
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_parquet(output_path, engine='pyarrow', index=False)
            print(f"\n✓ Saved {len(all_data)} trajectories to {output_path}")
        
        # Print summary
        successful = len(all_data)
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {len(problems)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(problems) - successful}")
        print(f"Success rate: {successful/len(problems)*100:.1f}%")
        print(f"Output file: {output_path}")
        print(f"{'='*60}")
        
        return all_data


def main():
    """Main execution function"""
    
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MAX_SAMPLES = 2
    OUTPUT_DIR = "./parquet_leetcode_trajectories/"
    
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
        use_local=True
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
    output_path = os.path.join(OUTPUT_DIR, "test_trajectories.parquet")
    print(f"\nStarting trajectory generation...")
    print(f"This may take a while depending on model size and hardware...")
    
    trajectories = generator.generate_trajectories_batch(
        problems=problems,
        output_path=output_path
    )
    
    print(f"\n✓ All done! Check {output_path} for results.")


if __name__ == "__main__":
    main()