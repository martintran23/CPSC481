def breadth_first_graph_search(problem, display=False):
    node = Node(problem.initial)
    if problem.goal_test(node.state): return node
    frontier = deque([node])
    explored = {problem.initial}
    expanded = 0  # Counter
    while frontier:
        node = frontier.popleft()
        expanded += 1
        for child in node.expand(problem):
            if child.state not in explored:
                if problem.goal_test(child.state):
                    if display: print(f"BFS nodes expanded: {expanded}")
                    return child
                explored.add(child.state)
                frontier.append(child)
    return None
