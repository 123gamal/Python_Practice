def dfs(graph, v, visited):
    visited[v] = True
    for neighbor in graph[v]:
        if not visited[neighbor]:
            dfs(graph, neighbor, visited)

def is_connected(graph, n, u, v):
    visited = [False] * n
    dfs(graph, u, visited)
    return visited[v]


n, m = map(int, input("Enter the number of vertices and edges (n m): ").split())

graph = [[] for _ in range(n)]
print("Enter the edges (one pair per line):")
for _ in range(m):
    x, y = map(int, input().split())
    graph[x-1].append(y-1)
    graph[y-1].append(x-1)


u, v = map(int, input("Enter the vertices to check connectivity between (u v): ").split())

if is_connected(graph, n, u-1, v-1):
    print(f"Yes, there is a path between vertices {u} and {v}")
else:
    print(f"No, there is no path between vertices {u} and {v}")
