import heapq
from collections import deque
import numpy as np
from .graph import Cell
from .utils import trace_path

"""
General graph search instructions:

First, define the correct data type to keep track of your visited cells
and add the start cell to it. If you need to initialize any properties
of the start cell, do that too.

Next, implement the graph search function. When you find a path, use the
trace_path() function to return a path given the goal cell and the graph. You
must have kept track of the parent of each node correctly and have implemented
the graph.get_parent() function for this to work. If you do not find a path,
return an empty list.

To visualize which cells are visited in the navigation webapp, save each
visited cell in the list in the graph class as follows:
     graph.visited_cells.append(Cell(cell_i, cell_j))
where cell_i and cell_j are the cell indices of the visited cell you want to
visualize.
"""


def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm. This algorithm is optional for P3.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    if not _endpoints_ok(graph, start, goal):
        return []

    stack = [start]  # LIFO: last cell pushed is the next one explored.
    graph.visited[start.j, start.i] = True

    while stack:
        cur = stack.pop()
        graph.visited_cells.append(Cell(cur.i, cur.j))
        if cur.i == goal.i and cur.j == goal.j:
            return trace_path(goal, graph)
        for n in graph.find_neighbors(cur.i, cur.j):
            if graph.visited[n.j, n.i] or graph.check_collision(n.i, n.j):
                continue
            graph.visited[n.j, n.i] = True
            graph.parent_i[n.j, n.i], graph.parent_j[n.j, n.i] = cur.i, cur.j
            stack.append(n)

    # If no path was found, return an empty list.
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    if not _endpoints_ok(graph, start, goal):
        return []

    # P3.5: FIFO queue. Cells are explored in order of how many steps they
    # are from the start, so the first time we reach the goal is a shortest path.
    queue = deque([start])
    graph.visited[start.j, start.i] = True

    while queue:
        cur = queue.popleft()
        graph.visited_cells.append(Cell(cur.i, cur.j))  # For the web app.
        if cur.i == goal.i and cur.j == goal.j:
            return trace_path(goal, graph)
        for n in graph.find_neighbors(cur.i, cur.j):
            # Skip cells already queued, and cells the robot would hit a wall in.
            if graph.visited[n.j, n.i] or graph.check_collision(n.i, n.j):
                continue
            graph.visited[n.j, n.i] = True
            graph.parent_i[n.j, n.i], graph.parent_j[n.j, n.i] = cur.i, cur.j
            queue.append(n)

    # If no path was found, return an empty list.
    return []


def a_star_search(graph, start, goal):
    """A* Search (BFS) algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    if not _endpoints_ok(graph, start, goal):
        return []

    def h(i, j):
        # Straight-line distance to the goal (never overestimates).
        return np.hypot(goal.i - i, goal.j - j)

    graph.dist[start.j, start.i] = 0.0
    count = 0  # Tie-breaker so the heap never has to compare Cells.
    heap = [(h(start.i, start.j), count, start.i, start.j)]

    while heap:
        _, _, ci, cj = heapq.heappop(heap)
        if graph.visited[cj, ci]:
            continue  # Already expanded with a cheaper cost.
        graph.visited[cj, ci] = True
        graph.visited_cells.append(Cell(ci, cj))
        if ci == goal.i and cj == goal.j:
            return trace_path(goal, graph)
        for n in graph.find_neighbors(ci, cj):
            if graph.visited[n.j, n.i] or graph.check_collision(n.i, n.j):
                continue
            step = 1.4142135 if (n.i != ci and n.j != cj) else 1.0
            new_cost = graph.dist[cj, ci] + step
            if new_cost < graph.dist[n.j, n.i]:
                graph.dist[n.j, n.i] = new_cost
                graph.parent_i[n.j, n.i], graph.parent_j[n.j, n.i] = ci, cj
                count += 1
                heapq.heappush(heap, (new_cost + h(n.i, n.j), count, n.i, n.j))

    # If no path was found, return an empty list.
    return []


def _endpoints_ok(graph, start, goal):
    """Prints a helpful message and returns False if start/goal can't be used."""
    for name, c in (("Start", start), ("Goal", goal)):
        if not graph.is_cell_in_bounds(c.i, c.j):
            print(f"{name} cell ({c.i}, {c.j}) is outside the map.")
            return False
        if graph.check_collision(c.i, c.j):
            print(f"{name} cell ({c.i}, {c.j}) is too close to an obstacle.")
            return False
    return True
