# from collections import deque
# from typing import List

# class Solution:
#     def shortestPathQueries(self, n: int, queries: List[List[int]]) -> List[int]:
#         # Build initial graph: i -> i+1
#         graph = [[] for _ in range(n)]
#         for i in range(n - 1):
#             graph[i].append(i + 1)

#         def bfs():
#             dist = [-1] * n
#             dist[0] = 0
#             q = deque([0])

#             while q:
#                 u = q.popleft()
#                 if u == n - 1:
#                     return dist[u]
#                 for v in graph[u]:
#                     if dist[v] == -1:
#                         dist[v] = dist[u] + 1
#                         q.append(v)

#             return -1

#         ans = []
#         for u, v in queries:
#             graph[u].append(v)
#             ans.append(bfs())

#         return ans


# # Example usage
# n = 5
# queries = [[2, 4], [0, 2], [0, 4]]

# solution = Solution()
# print(solution.shortestPathQueries(n, queries))  # [3, 2, 1]


# Import necessary libraries
# import heapq
# n = 5

# # Define the graph with default roads
# def init_graph(n):
#     graph = {i: [(i + 1, 1)] for i in range(n - 1)}
#     graph[n - 1] = []  # The last city has no outgoing roads
#     return graph

# # Function to perform Dijkstra's algorithm
# def dijkstra(graph, start, end):
#     distances = [float('inf')] * n
#     distances[start] = 0
#     priority_queue = [(0, start)]

#     while priority_queue:
#         current_distance, current_node = heapq.heappop(priority_queue)

#         if current_distance > distances[current_node]:
#             continue

#         for neighbor, weight in graph[current_node]:
#             distance = current_distance + weight

#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(priority_queue, (distance, neighbor))

#     return distances[end]

# # Test the initial approach
# def test_initial_approach():
#     n = 5
#     queries = [[2, 4], [0, 2], [0, 4]]
#     expected_output = [3, 2, 1]

#     graph = init_graph(n)
#     results = []

#     for u, v in queries:
#         if u not in graph:
#             graph[u] = []
#         graph[u].append((v, 1))
#         results.append(dijkstra(graph, 0, n - 1))

#     assert results == expected_output, f"Expected output: {expected_output}, but got: {results}"
#     print("Test passed!")

# test_initial_approach()

# Add more edges to the graph
# graph = {
#     0: [(1, 1), (2, 1)],
#     1: [(3, 1)],
#     2: [(3, 1)],
#     3: [(4, 1)],
#     4: []
# }

# # Function to perform Dijkstra's algorithm
# def dijkstra(graph, start, end):
#     distances = [float('inf')] * len(graph)
#     distances[start] = 0
#     priority_queue = [(0, start)]

#     while priority_queue:
#         current_distance, current_node = heapq.heappop(priority_queue)

#         if current_distance > distances[current_node]:
#             continue

#         for neighbor, weight in graph[current_node]:
#             distance = current_distance + weight

#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 heapq.heappush(priority_queue, (distance, neighbor))

#     return distances[end]

# # Test the Dijkstra's algorithm with the updated graph
# shortest_paths = [dijkstra(graph, 0, n - 1) for _ in range(len(queries))]
# print(shortest_paths)



# from typing import List
# import heapq

# class Solution:
#     def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
#         def dijkstra(graph, start, end):
#             distances = [float('inf')] * n
#             distances[start] = 0
#             priority_queue = [(0, start)]
            
#             while priority_queue:
#                 current_distance, current_node = heapq.heappop(priority_queue)
                
#                 if current_distance > distances[current_node]:
#                     continue
                
#                 for neighbor, weight in graph[current_node]:
#                     distance = current_distance + weight
                    
#                     if distance < distances[neighbor]:
#                         distances[neighbor] = distance
#                         heapq.heappush(priority_queue, (distance, neighbor))
            
#             return distances[end]
        
#         # Initialize the graph with the default roads
#         graph = {i: [(i + 1, 1)] for i in range(n - 1)}
#         graph[n - 1] = []  # The last city has no outgoing roads
        
#         # Process each query
#         results = []
#         for u, v in queries:
#             if u not in graph:
#                 graph[u] = []
#             graph[u].append((v, 1))
#             results.append(dijkstra(graph, 0, n - 1))
        
#         return results


# n=5
# queries=[[2,4],[0,2],[0,4]]
# # Initial graph: 0→1→2→3→4 (path length = 4)
# from collections import deque
# def bfs_shortest_path(graph, start, end, n):
#     """Find shortest path using BFS"""
#     distances = [-1] * n
#     distances[start] = 0
#     queue = deque([start])
#     while queue:
#         node = queue.popleft()
        
#         if node == end:
#             return distances[end]
        
#         for neighbor in graph[node]:
#             if distances[neighbor] == -1:
#                 distances[neighbor] = distances[node] + 1
#                 queue.append(neighbor)

#     return distances[end]

# n = 5
# graph = {i: [] for i in range(n)}

# for i in range(n - 1):
#     graph[i].append(i + 1)
#     print("Initial graph:", graph)
#     print("Initial shortest path from 0 to 4:", bfs_shortest_path(graph, 0, n-1, n))

# graph[2].append(4)
# print("\nAfter adding 2→4:")
# print("Shortest path:", bfs_shortest_path(graph, 0, n-1, n))

# graph[0].append(2)
# print("\nAfter adding 0→2:")
# print("Shortest path:", bfs_shortest_path(graph, 0, n-1, n))

# graph[0].append(4)
# print("\nAfter adding 0→4:")
# print("Shortest path:", bfs_shortest_path(graph, 0, n-1, n))


# Case when the longest_common_prefix is already defined above *****
# Complete solution testing
# test_cases = [
#     (["flower", "flow", "flight"], "fl"),
#     (["dog", "racecar", "car"], ""),
#     (["interspecies", "interstellar", "interstate"], "inters")
# ]

# for strs, expected in test_cases:
#     result = longest_common_prefix(strs)
#     assert result == expected, f"Expected {expected}, but got {result} for input {strs}"
#     print(f"Test passed for input {strs}: {result}")
