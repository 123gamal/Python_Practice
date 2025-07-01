def dfs(v, visited, graph):
    visited[v] = True
    for neighbor in graph[v]:
        if not visited[neighbor]:
            dfs(neighbor, visited, graph)

def count_connected_components(graph, n):
    visited = [False] * n
    component_count = 0
    for v in range(n):
        if not visited[v]:
            dfs(v, visited, graph)
            component_count += 1
    return component_count

n, m = map(int, input().split())

graph = [[] for _ in range(n)]

for _ in range(m):
    u, v = map(int, input().split())
    graph[u-1].append(v-1)
    graph[v-1].append(u-1)

components = count_connected_components(graph, n)
print(components)
