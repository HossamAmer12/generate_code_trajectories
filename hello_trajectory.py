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
# generated_code = responses[-1][-1]['content']

# generated_code = ()
# match = re.search(r"```python\s*(.*?)\s*```", generated_code, re.DOTALL)
# generated_code = match.group(1) if match else None


# code_str = """# Final solution
# def solve_problem(n: int, queries: List[List[int]]) -> List[int]:
#     def dijkstra(graph, start, end):
#         distances = [float('inf')] * n
#         distances[start] = 0
#         priority_queue = [(0, start)]

#         while priority_queue:
#             current_distance, current_node = heapq.heappop(priority_queue)

#             if current_distance > distances[current_node]:
#                 continue

#             for neighbor, weight in graph[current_node]:
#                 distance = current_distance + weight

#                 if distance < distances[neighbor]:
#                     distances[neighbor] = distance
#                     heapq.heappush(priority_queue, (distance, neighbor))

#         return distances[end]

#     # Initialize the graph with the default roads
#     graph = {i: [(i + 1, 1)] for i in range(n - 1)}
#     graph[n - 1] = []  # The last city has no outgoing roads

#     # Process each query
#     results = []
#     for u, v in queries:
#         if u not in graph:
#             graph[u] = []
#         graph[u].append((v, 1))
#         results.append(dijkstra(graph, 0, n - 1))

#     return results


# # Test cases
# print(solve_problem(5, [[2, 4], [0, 2], [0, 4]]))  # Output: [3, 2, 1]
# print(solve_problem(4, [[0, 3], [0, 2]]))  # Output: [1, 1]
# """

code_str = """from typing import List
import heapq

# Final solution
def solve_problem(n: int, queries: List[List[int]]) -> List[int]:
    def dijkstra(graph, start, end):
        distances = [float('inf')] * n
        distances[start] = 0
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[end]

    # Initialize the graph with the default roads
    graph = {i: [(i + 1, 1)] for i in range(n - 1)}
    graph[n - 1] = []  # The last city has no outgoing roads

    # Process each query
    results = []
    for u, v in queries:
        if u not in graph:
            graph[u] = []
        graph[u].append((v, 1))
        results.append(dijkstra(graph, 0, n - 1))

    return results


# Test cases
print(solve_problem(5, [[2, 4], [0, 2], [0, 4]]))  # Output: [3, 2, 1]
print(solve_problem(4, [[0, 3], [0, 2]]))  # Output: [1, 1]
"""


# Debug
generated_code = code_str

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
