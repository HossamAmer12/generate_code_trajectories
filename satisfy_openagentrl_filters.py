import json
import re
from typing import Dict, List, Any

# Compile regex pattern for Python code blocks
py_block = re.compile(r"```python[\s\S]*?```", re.IGNORECASE)

def is_code_trace(ex: Dict[str, Any]) -> bool:
    """
    Check if an example satisfies the code trace filters:
    1. Must have at least one tool call (assistant message with tool_calls)
    2. Must have at least one tool response (role == "tool")
    3. Last assistant message must contain Python code block
    4. Last assistant message must NOT contain \\boxed
    """
    msgs = ex.get("messages", [])
    
    # Check 1: Must have tool call (assistant with tool_calls)
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls") 
        for m in msgs
    )
    
    # Check 2: Must have tool response (role == "tool")
    has_tool_resp = any(m.get("role") == "tool" for m in msgs)
    
    # Check 3 & 4: Last assistant message checks
    last_a = next(
        (m for m in reversed(msgs) if m.get("role") == "assistant"), 
        None
    )
    
    if not last_a:
        return False
    
    content = last_a.get("content", "")
    has_final_python = bool(py_block.search(content))
    not_boxed = ("\\boxed" not in content)
    
    return has_tool_call and has_tool_resp and has_final_python and not_boxed

def analyze_and_filter_jsonl(input_file: str, output_file: str = None):
    """
    Analyze JSONL file and optionally filter it.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output filtered JSONL file (optional)
    """
    total_count = 0
    passing_count = 0
    failing_indices = []
    passing_examples = []
    
    print(f"Analyzing file: {input_file}\n")
    print("=" * 80)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            total_count += 1
            try:
                example = json.loads(line.strip())
                
                if is_code_trace(example):
                    passing_count += 1
                    passing_examples.append(example)
                else:
                    failing_indices.append(idx)
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}")
                failing_indices.append(idx)
    
    # Print results
    print(f"\nRESULTS:")
    print(f"  Total examples: {total_count}")
    print(f"  Passing filters: {passing_count}")
    print(f"  Failing filters: {len(failing_indices)}")
    print(f"  Pass rate: {passing_count/total_count*100:.2f}%")
    print("=" * 80)
    
    if failing_indices:
        print(f"\n❌ File does NOT satisfy all filters")
        print(f"\nFailing example indices (0-based): {failing_indices[:20]}")
        if len(failing_indices) > 20:
            print(f"... and {len(failing_indices) - 20} more")
        
        # Show detailed analysis of first failing example
        if failing_indices:
            print("\n" + "=" * 80)
            print("DETAILED ANALYSIS OF FIRST FAILING EXAMPLE:")
            print("=" * 80)
            analyze_failing_example(input_file, failing_indices[0])
    else:
        print(f"\n✅ All examples satisfy the filters!")
    
    # Write filtered output if requested
    if output_file and failing_indices:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in passing_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"\n✅ Filtered data written to: {output_file}")
        print(f"   ({passing_count} examples)")
    
    return {
        'total': total_count,
        'passing': passing_count,
        'failing': len(failing_indices),
        'failing_indices': failing_indices
    }

def analyze_failing_example(input_file: str, index: int):
    """Analyze why a specific example failed the filters."""
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == index:
                example = json.loads(line.strip())
                msgs = example.get("messages", [])
                
                # Check each condition
                has_tool_call = any(
                    m.get("role") == "assistant" and m.get("tool_calls") 
                    for m in msgs
                )
                has_tool_resp = any(m.get("role") == "tool" for m in msgs)
                
                last_a = next(
                    (m for m in reversed(msgs) if m.get("role") == "assistant"), 
                    None
                )
                
                if last_a:
                    content = last_a.get("content", "")
                    has_final_python = bool(py_block.search(content))
                    not_boxed = ("\\boxed" not in content)
                else:
                    has_final_python = False
                    not_boxed = False
                
                print(f"Example index: {index}")
                print(f"  ✓ Has tool call (assistant with tool_calls): {has_tool_call}")
                print(f"  ✓ Has tool response (role='tool'): {has_tool_resp}")
                print(f"  ✓ Last assistant has Python code block: {has_final_python}")
                print(f"  ✓ Last assistant does NOT have \\boxed: {not_boxed}")
                
                # Show message structure
                print(f"\nMessage structure:")
                for i, msg in enumerate(msgs):
                    role = msg.get("role", "unknown")
                    has_tc = "tool_calls" in msg
                    content_preview = str(msg.get("content", ""))[:100]
                    print(f"  [{i}] role={role}, has_tool_calls={has_tc}")
                    if content_preview:
                        print(f"      content preview: {content_preview}...")
                
                break

if __name__ == "__main__":
    import sys
    
    # Get input file from command line or use default
    # input_file = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/document.jsonl"
    # output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # input_file = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/document.jsonl"
    # output_file = sys.argv[2] if len(sys.argv) > 2 else None

    input_file = "/home/hossamamer/generate_code_trajectories/processed_trajectories/trajectories_messages.jsonl"
    output_file = "/home/hossamamer/generate_code_trajectories/processed_trajectories/trajectories_messages_satisfy.jsonl"
    
    # If no output file specified, ask user
    if output_file is None:
        results = analyze_and_filter_jsonl(input_file)
        if results['failing'] > 0:
            print("\n" + "=" * 80)
            print("To create a filtered output file, run:")
            print(f"  python filter_jsonl.py {input_file} filtered_output.jsonl")
    else:
        analyze_and_filter_jsonl(input_file, output_file)