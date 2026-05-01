"""Homework 6: 8-puzzle search comparison (A*, BFS, DFS)."""

from collections import deque
import heapq


GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
INITIAL_STATE = (1, 2, 3, 5, 7, 4, 8, 6, 0)


def is_goal(state):
    return state == GOAL_STATE


def misplaced_tiles(state):
    """A* heuristic: number of misplaced tiles (excluding blank)."""
    return sum(
        1 for i, tile in enumerate(state) if tile != 0 and tile != GOAL_STATE[i]
    )


def neighbors(state):
    """Generate (next_state, action) pairs."""
    blank_idx = state.index(0)
    row, col = divmod(blank_idx, 3)

    moves = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
    for dr, dc, action in moves:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            swap_idx = nr * 3 + nc
            next_state = list(state)
            next_state[blank_idx], next_state[swap_idx] = (
                next_state[swap_idx],
                next_state[blank_idx],
            )
            yield tuple(next_state), action


def reconstruct_path(parents, end_state):
    actions = []
    s = end_state
    while parents[s][0] is not None:
        s, action = parents[s]
        actions.append(action)
    actions.reverse()
    return actions


def astar_search_display(initial):
    """
    A* graph search with expanded-node counter.
    Expanded nodes = number of nodes popped from frontier.
    """
    frontier = []
    counter = 0
    g_cost = {initial: 0}
    parents = {initial: (None, None)}
    heapq.heappush(frontier, (misplaced_tiles(initial), counter, initial))

    expanded = 0
    while frontier:
        _, _, state = heapq.heappop(frontier)
        current_g = g_cost[state]
        expanded += 1

        if is_goal(state):
            return reconstruct_path(parents, state), expanded

        for nxt, action in neighbors(state):
            ng = current_g + 1
            if ng < g_cost.get(nxt, 10**9):
                g_cost[nxt] = ng
                parents[nxt] = (state, action)
                counter += 1
                heapq.heappush(frontier, (ng + misplaced_tiles(nxt), counter, nxt))

    return [], expanded


def breadth_first_graph_search_display(initial):
    frontier = deque([initial])
    frontier_set = {initial}
    explored = set()
    parents = {initial: (None, None)}
    expanded = 0

    while frontier:
        state = frontier.popleft()
        frontier_set.discard(state)
        expanded += 1

        if is_goal(state):
            return reconstruct_path(parents, state), expanded

        explored.add(state)
        for nxt, action in neighbors(state):
            if nxt not in explored and nxt not in frontier_set:
                parents[nxt] = (state, action)
                frontier.append(nxt)
                frontier_set.add(nxt)

    return [], expanded


def depth_first_graph_search_display(initial):
    frontier = [initial]
    frontier_set = {initial}
    explored = set()
    parents = {initial: (None, None)}
    expanded = 0

    while frontier:
        state = frontier.pop()
        frontier_set.discard(state)
        expanded += 1

        if is_goal(state):
            return reconstruct_path(parents, state), expanded

        explored.add(state)
        for nxt, action in neighbors(state):
            if nxt not in explored and nxt not in frontier_set:
                if nxt not in parents:
                    parents[nxt] = (state, action)
                frontier.append(nxt)
                frontier_set.add(nxt)

    return [], expanded


if __name__ == "__main__":
    print("Running A* search...")
    astar_solution, astar_expanded = astar_search_display(INITIAL_STATE)
    print("A* solution:", astar_solution)
    print("Nodes expanded (A*):", astar_expanded)

    print("\nRunning BFS...")
    bfs_solution, bfs_expanded = breadth_first_graph_search_display(INITIAL_STATE)
    print("BFS solution:", bfs_solution)
    print("Nodes expanded (BFS):", bfs_expanded)

    print("\nRunning DFS...")
    dfs_solution, dfs_expanded = depth_first_graph_search_display(INITIAL_STATE)
    print("DFS solution:", dfs_solution)
    print("Nodes expanded (DFS):", dfs_expanded)