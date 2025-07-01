# DFS function to detect cycles in a directed graph
def dfs(v, graph, visited, rec_stack):
    # Mark the current node as visited and add to the recursion stack
    visited[v] = True
    rec_stack[v] = True

    # Visit all neighbors (edges) of this vertex
    for neighbor in graph[v]:
        # If neighbor has not been visited, visit it recursively
        if not visited[neighbor]:
            if dfs(neighbor, graph, visited, rec_stack):
                return True
        # If neighbor is in the recursion stack, a cycle is found
        elif rec_stack[neighbor]:
            return True

    # Remove the vertex from recursion stack after processing
    rec_stack[v] = False
    return False


# Function to check if the graph contains a cycle
def contains_cycle(graph, n):
    visited = [False] * n  # Track visited nodes
    rec_stack = [False] * n  # Track nodes in the recursion stack

    # Perform DFS from each node if it has not been visited
    for node in range(n):
        if not visited[node]:
            if dfs(node, graph, visited, rec_stack):
                return True
    return False


# Main function to take input and invoke cycle detection
def main():
    # Input: number of vertices (n) and edges (m)
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]

    # Reading the edges and creating the adjacency list
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u - 1].append(v - 1)  # Directed edge from u to v

    # Check for cycle and print the result
    if contains_cycle(graph, n):
        print(1)  # Output 1 if there is a cycle
    else:
        print(0)  # Output 0 if no cycle


# Run the main function
if __name__ == "__main__":
    main()
