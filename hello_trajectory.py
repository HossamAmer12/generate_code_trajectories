import requests
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
import re

# --- Step 1: Define the problem prompt ---
N = 10
initial_prompt = f"""
You are a Python coding assistant.
Write Python code to sum numbers from 1 to {N}. Print the result at the end.
Return only working Python code, nothing else.
"""

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Small model for testing

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

agent = Assistant(
            llm=llm_cfg,
            name='LeetCode Solver',
            description='An AI assistant that solves LeetCode problems',
            function_list=[]  # Empty list - no tools to avoid Docker
        )

# --- Step 2: Ask Qwen-Coder via vLLM ---
messages = [{'role': 'user', 'content': initial_prompt}]
responses = []

# Extract the generated code
# vLLM responses are usually a list of messages, take the first one
for response in agent.run(messages=messages):
    responses.append(response)          
            

# Extract assistant's response
generated_code = responses[-1][-1]['content']

match = re.search(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
generated_code = match.group(1) if match else None


print("Generated Code:")
print(generated_code)
print("******************")

# --- Step 3: Send the generated code to SandboxFusion ---
sandbox_payload = {
    "code": generated_code,
    "language": "python"
}

sandbox_response = requests.post(
    "http://localhost:8080/run_code",  # Your sandbox endpoint
    headers={"Content-Type": "application/json"},
    json=sandbox_payload
)

print("Sandbox response: ")
print(sandbox_response.json())

sandbox_result = sandbox_response.json()
output = sandbox_result.get("run_result", {}).get("stdout", "").strip()
print("Sandbox Output:", output)
