#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_tool_calls.py

Count the number of tool calls per example in a dataset.
Works with both the original conversation format and the converted ReTool-SFT format.

Usage:
  python count_tool_calls.py --input data.json
  python count_tool_calls.py --input data.parquet
  python count_tool_calls.py --input data.jsonl
"""

import argparse
import json
import re
from typing import Dict, List, Any
import datasets


EXECUTE_TAG_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)


def count_tool_calls_conversations(conversations: List[Dict[str, str]]) -> int:
    """Count tool calls in original conversation format (count <execute> tags)."""
    count = 0
    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", "")
            # Count <execute> tags
            count += len(EXECUTE_TAG_RE.findall(content))
    return count


def count_tool_calls_messages(messages: List[Dict[str, Any]]) -> int:
    """Count tool calls in ReTool-SFT messages format."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                count += len(tool_calls)
    return count


def analyze_dataset(dataset):
    """Analyze tool call statistics for the dataset."""
    tool_call_counts = []
    
    for example in dataset:
        # Check which format we have
        if "conversations" in example:
            # Original format
            count = count_tool_calls_conversations(example["conversations"])
        elif "messages" in example:
            # Converted format
            count = count_tool_calls_messages(example["messages"])
        else:
            print(f"Warning: Unknown format for example, skipping")
            continue
        
        tool_call_counts.append(count)
    
    if not tool_call_counts:
        print("No examples found!")
        return
    
    total_examples = len(tool_call_counts)
    total_tool_calls = sum(tool_call_counts)
    avg_tool_calls = total_tool_calls / total_examples
    max_tool_calls = max(tool_call_counts)
    min_tool_calls = min(tool_call_counts)
    
    # Count distribution
    from collections import Counter
    distribution = Counter(tool_call_counts)
    
    print("\n" + "="*60)
    print("TOOL CALL STATISTICS")
    print("="*60)
    print(f"Total examples: {total_examples:,}")
    print(f"Total tool calls: {total_tool_calls:,}")
    print(f"Average tool calls per example: {avg_tool_calls:.2f}")
    print(f"Min tool calls: {min_tool_calls}")
    print(f"Max tool calls: {max_tool_calls}")
    print(f"Median tool calls: {sorted(tool_call_counts)[total_examples//2]}")
    
    print("\n" + "-"*60)
    print("DISTRIBUTION:")
    print("-"*60)
    for num_calls in sorted(distribution.keys()):
        count = distribution[num_calls]
        pct = (count / total_examples) * 100
        bar = "█" * int(pct / 2)
        print(f"{num_calls:2d} tool calls: {count:5d} examples ({pct:5.1f}%) {bar}")
    
    print("="*60 + "\n")
    
    return {
        "total_examples": total_examples,
        "total_tool_calls": total_tool_calls,
        "average": avg_tool_calls,
        "min": min_tool_calls,
        "max": max_tool_calls,
        "distribution": dict(distribution)
    }


def main():
    parser = argparse.ArgumentParser(description="Count tool calls in dataset")
    parser.add_argument("--input", required=True, help="Input file (json, jsonl, or parquet)")
    parser.add_argument("--output", help="Optional: save statistics to JSON file")
    
    args = parser.parse_args()
    
    # Determine file type and load
    input_path = args.input
    
    print(f"Loading dataset from: {input_path}")
    
    if input_path.endswith('.parquet'):
        ds = datasets.load_dataset("parquet", data_files=input_path, split="train")
    elif input_path.endswith('.jsonl'):
        ds = datasets.load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith('.json'):
        ds = datasets.load_dataset("json", data_files=input_path, split="train")
    else:
        # Try to auto-detect
        try:
            ds = datasets.load_dataset("json", data_files=input_path, split="train")
        except:
            try:
                ds = datasets.load_dataset("parquet", data_files=input_path, split="train")
            except:
                print(f"Error: Could not load {input_path}")
                return
    
    print(f"Loaded {len(ds)} examples")
    
    # Analyze
    stats = analyze_dataset(ds)
    
    # Optionally save stats
    if args.output and stats:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {args.output}")


if __name__ == "__main__":
    main()


#  python .\count_toolcalls_retool.py --input .\my_local_zhentao_dataset_folder\Gen-Verse___open-agent_rl-sft-3_k\Open-AgentRL-SFT-3K.code_only.parquet
# Zhentao dataset:
# ============================================================
# TOOL CALL STATISTICS
# ============================================================
# Total examples: 144
# Total tool calls: 306
# Average tool calls per example: 2.12
# Min tool calls: 1
# Max tool calls: 8
# Median tool calls: 2

# ------------------------------------------------------------
# DISTRIBUTION:
# ------------------------------------------------------------
#  1 tool calls:    66 examples ( 45.8%) ██████████████████████
#  2 tool calls:    49 examples ( 34.0%) █████████████████
#  4 tool calls:    14 examples (  9.7%) ████
#  5 tool calls:     7 examples (  4.9%) ██
#  6 tool calls:     6 examples (  4.2%) ██
#  7 tool calls:     1 examples (  0.7%)
#  8 tool calls:     1 examples (  0.7%)
# ============================================================


# Hossam dataset:
#  python count_toolcalls_retool_sft.py --input ./processed_trajectories/merged_trajectories_messages_sft.parquet
# Loaded 318 examples

# ============================================================
# TOOL CALL STATISTICS
# ============================================================
# Total examples: 318
# Total tool calls: 785
# Average tool calls per example: 2.47
# Min tool calls: 0
# Max tool calls: 8
# Median tool calls: 2

# ------------------------------------------------------------
# DISTRIBUTION:
# ------------------------------------------------------------
#  0 tool calls:     1 examples (  0.3%) 
#  1 tool calls:    90 examples ( 28.3%) ██████████████
#  2 tool calls:    70 examples ( 22.0%) ███████████
#  3 tool calls:    99 examples ( 31.1%) ███████████████
#  4 tool calls:    43 examples ( 13.5%) ██████
#  5 tool calls:     7 examples (  2.2%) █
#  6 tool calls:     6 examples (  1.9%) 
#  7 tool calls:     1 examples (  0.3%) 
#  8 tool calls:     1 examples (  0.3%) 