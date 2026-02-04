import requests
import re
from qwen_agent.agents import Assistant

# -------------------------------
# Config
# -------------------------------
N = 10
NUM_SAMPLES = 4

initial_prompt = f"""
You are a Python coding assistant.
Write Python code to sum numbers from 1 to {N}. Print the result at the end.
Return only working Python code, nothing else.
"""

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

llm_cfg = {
    'model': model_name,
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'top_p': 0.8,
        'temperature': 0.7,
        'max_tokens': 2048,
    }
}

agent = Assistant(
    llm=llm_cfg,
    name='LeetCode Solver',
    description='An AI assistant that solves LeetCode problems',
    function_list=[]
)

# -------------------------------
# Generate multiple code samples
# -------------------------------
all_generated_codes = []

for i in range(NUM_SAMPLES):
    # Run the agent
    responses = list(agent.run(messages=[{'role': 'user', 'content': initial_prompt}]))
    
    # Extract the last message from the agent
    raw_content = responses[-1][-1]['content']
    
    # Try to extract code from ```python``` blocks first
    match = re.search(r"```python\s*(.*?)\s*```", raw_content, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        # Fallback: remove leading/trailing whitespace
        code = raw_content.strip()
    
    all_generated_codes.append(code)

# -------------------------------
# Print all generated codes
# -------------------------------
print("Generated Code Samples:")
for idx, code in enumerate(all_generated_codes, 1):
    print(f"\n--- Sample {idx} ---")
    print(code)

# -------------------------------
# Execute each code in SandboxFusion
# -------------------------------
print("\nSandbox Outputs:")
for idx, code in enumerate(all_generated_codes, 1):
    sandbox_payload = {"code": code, "language": "python"}
    
    sandbox_response = requests.post(
        "http://localhost:8080/run_code",
        headers={"Content-Type": "application/json"},
        json=sandbox_payload
    )
    
    sandbox_result = sandbox_response.json()
    output = sandbox_result.get("run_result", {}).get("stdout", "").strip()
    print(f"Sample {idx} Output: {output}")
