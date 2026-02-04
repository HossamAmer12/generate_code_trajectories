# import requests

# resp = requests.post(
#     "http://localhost:8080/run_code",
#     json={
#         "language": "python",
#         "code": "print('sandbox works')"
#     },
#     timeout=10,
# )

# print(resp.json())


import requests

code = """
def sum_1_to_3():
    total = 0
    for i in range(1, 4):
        total += i
    return total

print(sum_1_to_3())
"""

resp = requests.post(
    "http://localhost:8080/run_code",
    json={
        "language": "python",
        "code": code
    },
    timeout=10,
)

print("Status:", resp.status_code)
print("Response:", resp.json())
