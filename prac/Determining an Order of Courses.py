from collections import deque


def topological_sort(n, graph):
    in_degree = [0] * n  # Array to track in-degrees of vertices
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1  # Increase the in-degree of the target vertex

    # Initialize a queue with all vertices of in-degree 0
    queue = deque()

    # Push all vertices with in-degree 0 into the queue
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []  # List to store the topological order

    while queue:
        u = queue.popleft()  # Get the vertex from the queue
        topo_order.append(u + 1)  # Append it to the topological order (1-indexed)

        # Decrease the in-degree of neighboring vertices
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)  # Add to the queue if in-degree becomes 0

    return topo_order


# Main function to take input and invoke the topological sort
def main():
    # Input: number of vertices (n) and edges (m)
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]

    # Reading the edges and creating the adjacency list
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u - 1].append(v - 1)  # Directed edge from u to v

    # Perform topological sort
    sorted_order = topological_sort(n, graph)

    # Output the topologically sorted order in a single line
    print(" ".join(map(str, sorted_order)))


# Run the main function
if __name__ == "__main__":
    main()
