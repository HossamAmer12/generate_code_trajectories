"""
Process LeetCode trajectories from JSONL format
- Convert to messages format (JSON and Parquet)
- Generate statistics on tool calls and interpreter outputs
- Handle errors and replace with original responses
- Fix empty interpreter outputs
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict


def extract_code_blocks(content: str) -> List[str]:
    """Extract all code blocks from content"""
    code_pattern = r'<code>\s*```python\s*(.*?)```\s*</code>'
    matches = re.findall(code_pattern, content, re.DOTALL)
    return [match.strip() for match in matches]


def extract_interpreter_outputs(content: str) -> List[str]:
    """Extract all interpreter outputs from content"""
    interpreter_pattern = r'<interpreter>\s*(.*?)\s*</interpreter>'
    matches = re.findall(interpreter_pattern, content, re.DOTALL)
    return [match.strip() for match in matches]


def fix_empty_interpreter_outputs(content: str) -> Tuple[str, int]:
    """
    Fix cases where code output is shown but interpreter tag is empty.
    Example: ```python\n[3, -1, 3]\n```\n<interpreter>\n\n</interpreter>
    Should become: <interpreter>\n[3, -1, 3]\n</interpreter>
    
    Returns: (fixed_content, num_fixes)
    """
    num_fixes = 0
    
    # Pattern to find code followed by empty interpreter
    pattern = r'```python\s*(.*?)```\s*<interpreter>\s*</interpreter>'
    
    def replacer(match):
        nonlocal num_fixes
        code_output = match.group(1).strip()
        
        # Check if this looks like output (not actual code)
        # Simple heuristic: if it's very short or looks like a list/number
        if code_output and (
            len(code_output) < 100 or 
            re.match(r'^\[.*\]$', code_output) or
            re.match(r'^-?\d+(\.\d+)?$', code_output) or
            re.match(r'^(True|False|None)$', code_output)
        ):
            num_fixes += 1
            return f'```python\n{code_output}\n```\n<interpreter>\n{code_output}\n</interpreter>'
        
        return match.group(0)
    
    fixed_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    return fixed_content, num_fixes


def check_for_errors(interpreter_outputs: List[str]) -> List[str]:
    """Check if any interpreter outputs contain errors"""
    error_outputs = []
    error_keywords = ['ERROR', 'Error', 'error', 'Traceback', 'Exception', 'SyntaxError', 
                      'NameError', 'TypeError', 'ValueError', 'AttributeError']
    
    for output in interpreter_outputs:
        if any(keyword in output for keyword in error_keywords):
            error_outputs.append(output)
    
    return error_outputs


def convert_trajectory_to_messages(trajectory: Dict, original_problem: Dict = None) -> Dict:
    """
    Convert trajectory format to messages format.
    
    Args:
        trajectory: Dictionary with conversation history
        original_problem: Optional original problem data for error recovery
        
    Returns:
        Dictionary with 'messages' and 'tools' keys, plus metadata
    """
    messages = []
    stats = {
        'num_tool_calls': 0,
        'num_tool_results': 0,
        'num_empty_outputs': 0,
        'num_error_outputs': 0,
        'num_untagged_code_blocks': 0,
        'has_errors': False,
        'used_original_response': False,
        'num_fixes': 0
    }
    
    # Process conversation turns
    for turn_idx, turn_data in enumerate(trajectory.get("conversation", [])):
        role = turn_data.get("role")
        content = turn_data.get("content", "")
        
        if role == "user":
            # For user message, use the problem title and description from the dataset
            # NOT the prompt that was sent to the model
            problem_title = trajectory.get("problem_title", "")
            problem_description = original_problem.get("description", "") if original_problem else ""
            
            # If we don't have the original description, try to extract from content
            if not problem_description:
                # The content might have the problem description in it
                # Try to extract just the problem part (before all the instructions)
                user_question_match = re.search(r'USER QUESTION:\s*(.*?)\s*(?:ORIGINAL THINKING PROCESS|$)', content, re.DOTALL)
                if user_question_match:
                    problem_description = user_question_match.group(1).strip()
                else:
                    # Fall back to using content as-is
                    problem_description = content
            
            # Format user message with title and description
            user_content = f"Title:{problem_title}\nContent:{problem_description}"
            
            messages.append({
                "content": user_content,
                "role": "user",
                "tool_calls": None
            })
        
        elif role == "assistant":
            # First, fix empty interpreter outputs
            fixed_content, num_fixes = fix_empty_interpreter_outputs(content)
            stats['num_fixes'] += num_fixes
            
            # Extract code blocks wrapped in <code> tags
            code_blocks_tagged = extract_code_blocks(fixed_content)
            
            # Also check for code blocks that aren't wrapped in <code> tags
            # Pattern: ```python\n...\n``` NOT preceded by <code>
            untagged_code_pattern = r'(?<!<code>\s)```python\s*(.*?)```(?!\s*</code>)'
            untagged_matches = re.findall(untagged_code_pattern, fixed_content, re.DOTALL)
            
            # Combine all code blocks
            all_code_blocks = code_blocks_tagged + [match.strip() for match in untagged_matches]
            
            if untagged_matches:
                stats['num_untagged_code_blocks'] = len(untagged_matches)
            
            # Extract interpreter outputs
            interpreter_outputs = extract_interpreter_outputs(fixed_content)
            
            # Check if we have untagged code blocks without interpreter outputs
            if len(untagged_matches) > 0 and len(untagged_matches) > len(interpreter_outputs):
                # Some code blocks don't have interpreter outputs
                print(f"  ⚠️  Found {len(untagged_matches)} untagged code blocks but only {len(interpreter_outputs)} interpreter outputs in: {trajectory.get('problem_title')}")
            
            # Check for errors
            error_outputs = check_for_errors(interpreter_outputs)
            if error_outputs:
                stats['has_errors'] = True
                stats['num_error_outputs'] = len(error_outputs)
            
            # Count empty outputs
            empty_outputs = sum(1 for output in interpreter_outputs if not output.strip())
            stats['num_empty_outputs'] = empty_outputs
            
            # If there are errors and we have original response, use it instead
            if stats['has_errors'] and original_problem and original_problem.get('original_response'):
                print(f"  ⚠️  Using original response due to errors in: {trajectory.get('problem_title')}")
                fixed_content = original_problem['original_response']
                all_code_blocks = extract_code_blocks(fixed_content)
                interpreter_outputs = extract_interpreter_outputs(fixed_content)
                stats['used_original_response'] = True
            
            # Create tool calls from all code blocks
            tool_calls = []
            for code in all_code_blocks:
                tool_calls.append({
                    "function": {
                        "arguments": {"code": code},
                        "name": "code_interpreter"
                    },
                    "type": "function"
                })
            
            stats['num_tool_calls'] = len(tool_calls)
            stats['num_tool_results'] = len(interpreter_outputs)
            
            # Remove code blocks and interpreter outputs from content for clean display
            clean_content = re.sub(r'<code>\s*```python\s*.*?```\s*</code>', '', fixed_content, flags=re.DOTALL)
            clean_content = re.sub(r'```python\s*.*?```', '', clean_content, flags=re.DOTALL)
            clean_content = re.sub(r'<interpreter>\s*.*?\s*</interpreter>', '', clean_content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            
            # If this is the last assistant turn, add the final solution
            is_last_turn = (turn_idx == len(trajectory.get("conversation", [])) - 1)
            if is_last_turn and trajectory.get("final_solution"):
                final_solution = trajectory.get("final_solution", "")
                clean_content += f"\n\nHere's the final solution:\n\n```python\n{final_solution}\n```"
            
            # Add assistant message with tool calls
            messages.append({
                "content": clean_content,
                "role": "assistant",
                "tool_calls": tool_calls if tool_calls else None
            })
            
            # Add tool result messages
            for output in interpreter_outputs:
                # Format the result to match expected structure
                if output.strip():
                    formatted_result = f"execute_result:\n\n```\n{output}\n```"
                else:
                    formatted_result = "execute_result:\n\n```\n\n```"
                
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
        "tools": tools,
        "problem_id": trajectory.get("problem_id"),
        "problem_title": trajectory.get("problem_title"),
        "difficulty": trajectory.get("difficulty"),
        "final_solution": trajectory.get("final_solution"),
        "stats": stats
    }


def load_original_problems(dataset_name: str = "newfacade/LeetCodeDataset", 
                          split: str = "train", 
                          max_samples: int = 2000) -> Dict:
    """Load original LeetCode dataset for error recovery"""
    try:
        from datasets import load_dataset
        print(f"Loading original dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split=split)
        
        # Create lookup by task_id
        problems_lookup = {}
        for item in list(dataset)[:max_samples]:
            task_id = item.get('task_id', '')
            problems_lookup[task_id] = {
                'original_response': item.get('response', ''),
                'description': item.get('problem_description', ''),
                'task_id': task_id
            }
        
        print(f"Loaded {len(problems_lookup)} original problems for lookup")
        return problems_lookup
        
    except Exception as e:
        print(f"Warning: Could not load original dataset: {e}")
        return {}


def process_trajectories(
    input_jsonl_path: str,
    output_dir: str = "./processed_trajectories/",
    load_originals: bool = True,
    max_samples: int = 100
):
    """
    Process trajectories from JSON/JSONL and convert to messages format.
    
    Args:
        input_path: Path to input file (JSON array or JSONL)
        output_dir: Directory for output files
        load_originals: Whether to load original dataset for error recovery
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original problems if requested
    original_problems = {}
    if load_originals:
        original_problems = load_original_problems()
    
    # Statistics
    global_stats = {
        'total_problems': 0,
        'total_tool_calls': 0,
        'total_tool_results': 0,
        'total_empty_outputs': 0,
        'total_error_outputs': 0,
        'total_untagged_code_blocks': 0,
        'problems_with_errors': 0,
        'problems_using_original': 0,
        'total_fixes': 0,
        'tool_calls_distribution': defaultdict(int),
        'errors_by_problem': [],
        'untagged_code_problems': []
    }
    
    # Process trajectories
    all_messages_data = []
    error_log = []
    
    print(f"\nProcessing trajectories from: {input_jsonl_path}")
    print("="*60)
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            # Stop at maximum number of sample
            if line_num > max_samples:
                break
                
            try:
                trajectory = json.loads(line)
                global_stats['total_problems'] += 1
                
                # Get original problem data if available
                problem_title = trajectory.get('problem_title', '')
                original_problem = original_problems.get(problem_title, {})
                
                # Convert to messages format
                messages_data = convert_trajectory_to_messages(trajectory, original_problem)
                
                # Update statistics
                stats = messages_data.get('stats', {})
                global_stats['total_tool_calls'] += stats.get('num_tool_calls', 0)
                global_stats['total_tool_results'] += stats.get('num_tool_results', 0)
                global_stats['total_empty_outputs'] += stats.get('num_empty_outputs', 0)
                global_stats['total_error_outputs'] += stats.get('num_error_outputs', 0)
                global_stats['total_fixes'] += stats.get('num_fixes', 0)
                global_stats['total_untagged_code_blocks'] += stats.get('num_untagged_code_blocks', 0)
                
                if stats.get('num_untagged_code_blocks', 0) > 0:
                    global_stats['untagged_code_problems'].append({
                        'problem_id': trajectory.get('problem_id'),
                        'problem_title': problem_title,
                        'num_untagged': stats.get('num_untagged_code_blocks', 0)
                    })
                
                if stats.get('has_errors'):
                    global_stats['problems_with_errors'] += 1
                    error_info = {
                        'problem_id': trajectory.get('problem_id'),
                        'problem_title': problem_title,
                        'num_errors': stats.get('num_error_outputs', 0),
                        'used_original': stats.get('used_original_response', False)
                    }
                    global_stats['errors_by_problem'].append(error_info)
                    error_log.append(error_info)
                
                if stats.get('used_original_response'):
                    global_stats['problems_using_original'] += 1
                
                # Track distribution of tool calls
                num_calls = stats.get('num_tool_calls', 0)
                global_stats['tool_calls_distribution'][num_calls] += 1
                
                # Remove stats from final output
                messages_data_clean = {
                    'messages': messages_data['messages'],
                    'tools': messages_data['tools']
                }
                
                all_messages_data.append(messages_data_clean)
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} trajectories...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"✓ Processed {global_stats['total_problems']} trajectories")
    
    # Save as JSON
    json_output_path = os.path.join(output_dir, "trajectories_messages.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_messages_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved JSON to: {json_output_path}")
    
    # Save as JSONL
    jsonl_output_path = os.path.join(output_dir, "trajectories_messages.jsonl")
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for item in all_messages_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved JSONL to: {jsonl_output_path}")
    
    # Save as Parquet
    parquet_output_path = os.path.join(output_dir, "trajectories_messages.parquet")
    df = pd.DataFrame(all_messages_data)
    df.to_parquet(parquet_output_path, engine='pyarrow', index=False)
    print(f"✓ Saved Parquet to: {parquet_output_path}")
    
    # Save error log
    if error_log:
        error_log_path = os.path.join(output_dir, "error_log.json")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_problems_with_errors': len(error_log),
                    'problems_using_original_response': sum(1 for e in error_log if e['used_original'])
                },
                'errors': error_log
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved error log to: {error_log_path}")
    
    # Save untagged code blocks log
    if global_stats['untagged_code_problems']:
        untagged_log_path = os.path.join(output_dir, "untagged_code_blocks_log.json")
        with open(untagged_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_problems_with_untagged_code': len(global_stats['untagged_code_problems']),
                    'total_untagged_blocks': global_stats['total_untagged_code_blocks']
                },
                'problems': global_stats['untagged_code_problems']
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved untagged code blocks log to: {untagged_log_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f"Total problems processed: {global_stats['total_problems']}")
    print(f"Total tool calls: {global_stats['total_tool_calls']}")
    print(f"Total tool results: {global_stats['total_tool_results']}")
    print(f"Total empty outputs: {global_stats['total_empty_outputs']}")
    print(f"Total error outputs: {global_stats['total_error_outputs']}")
    print(f"Total untagged code blocks: {global_stats['total_untagged_code_blocks']}")
    print(f"Problems with errors: {global_stats['problems_with_errors']}")
    print(f"Problems using original response: {global_stats['problems_using_original']}")
    print(f"Total fixes applied (empty outputs): {global_stats['total_fixes']}")
    print(f"Problems with untagged code blocks: {len(global_stats['untagged_code_problems'])}")
    
    print("\nTool Calls Distribution:")
    for num_calls in sorted(global_stats['tool_calls_distribution'].keys()):
        count = global_stats['tool_calls_distribution'][num_calls]
        percent = 100.0* global_stats['tool_calls_distribution'][num_calls] / global_stats['total_problems']
        bar = "█" * int(percent / 2)
        print(f"  {num_calls} tool calls: {count} problems ({percent:.2f})% {bar}")
    
    
    if global_stats['problems_with_errors'] > 0:
        print(f"\n⚠️  Found {global_stats['problems_with_errors']} problems with errors")
        print(f"   {global_stats['problems_using_original']} were replaced with original responses")
        print(f"   See error_log.json for details")
    
    if len(global_stats['untagged_code_problems']) > 0:
        print(f"\n⚠️  Found {len(global_stats['untagged_code_problems'])} problems with untagged code blocks")
        print(f"   (Code blocks using ```python``` without <code> tags)")
        print(f"   Total untagged blocks: {global_stats['total_untagged_code_blocks']}")
    
    print("="*60)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        # Convert defaultdict to regular dict for JSON serialization
        stats_to_save = {
            **global_stats,
            'tool_calls_distribution': dict(global_stats['tool_calls_distribution'])
        }
        json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved statistics to: {stats_path}")
    
    return all_messages_data, global_stats


def main():
    """Main execution function"""
    
    # Configuration
    INPUT_FILE = "./leetcode_trajectories/test_trajectories.jsonl"  # Can be .json or .jsonl
    OUTPUT_DIR = "./processed_trajectories/"
    MAX_SAMPLES = 1
    
    print("="*60)
    print("LeetCode Trajectories Processor")
    print("="*60)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please update INPUT_FILE path in the script")
        return
    
    # Process trajectories
    messages_data, stats = process_trajectories(
        input_jsonl_path=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        load_originals=True,
        max_samples=MAX_SAMPLES
    )
    
    print(f"\n✓ All done! Check {OUTPUT_DIR} for results.")
    print(f"\nOutput files:")
    print(f"  - trajectories_messages.json (JSON format)")
    print(f"  - trajectories_messages.jsonl (JSONL format)")
    print(f"  - trajectories_messages.parquet (Parquet format)")
    print(f"  - statistics.json (Processing statistics)")
    if stats['problems_with_errors'] > 0:
        print(f"  - error_log.json (Error details)")


if __name__ == "__main__":
    main()