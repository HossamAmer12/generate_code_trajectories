import json

# This file to view one example from the training data
# Helpful to see the turns happening the processed data

# Define your file names
input_file = './processed_trajectories/trajectories_messages.jsonl'
output_file = './processed_trajectories/one_example.json'

# Set an example index
example_index = 6

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == example_index:
                data = json.loads(line)
                break
        else:
            raise IndexError("Index out of range")

    # 3. Save the dictionary as a standard JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        # indent=4 makes the file human-readable (pretty-printed)
        json.dump(data, f, indent=4)
        
    print(f"Success! The first example has been saved to {output_file}")

except FileNotFoundError:
    print("Error: The .jsonl file was not found. Check your file path!")


# 1. Load the data
with open(output_file, "rb") as f:
    data = json.load(f)
messages = data.get("messages", [])

# 2. Print high-level summary
print(f"--- Conversation Summary ---")
print(f"Total Messages: {len(messages)}")
print("-" * 30)
num_tool_calls = 0
# 3. Iterate and print details for each message
for i, msg in enumerate(messages):
    role = msg.get("role", "Unknown").upper()
    content = msg.get("content", "")
    
    # Snippet for visibility (first 80 characters)
    snippet = content.replace('\n', ' ')[:80] + "..."

    has_tool_calls = msg.get("tool_calls", "")
    if has_tool_calls == None:
        has_tool_calls = False
    else:
        has_tool_calls = True
        num_tool_calls = num_tool_calls +  1
    
    print(f"Message #{i}")
    print(f"  Role   : {role}")
    print(f"  Length : {len(content)} characters")
    print(f"  Has tool calls   : {has_tool_calls}")
    print(f"  Preview: {snippet}")
    print("-" * 30)

print("Tools defined:")
tools = data.get("tools", [])
print(tools)

print(f"--- Conversation Summary Index#{example_index} ---")
print(f"Total Messages: {len(messages)}")
print(f"Total Total calls: {num_tool_calls}")