"""Tests your path planner on every map in data/ -- no robot needed.

Run:  python3 test_planner.py

For each map it picks a few start/goal pairs that aren't next to walls and
checks that each path: starts at start, ends at goal, only takes single
steps between neighboring cells, and never goes near a wall.
"""
import glob, random, time
from src.graph import GridGraph, Cell
from src.graph_search import breadth_first_search, depth_first_search, a_star_search

random.seed(0)
failures = 0

def check(graph, start, goal, path, name):
    global failures
    problems = []
    if not path:
        return "no path"
    if (path[0].i, path[0].j) != (start.i, start.j): problems.append("doesn't start at start")
    if (path[-1].i, path[-1].j) != (goal.i, goal.j): problems.append("doesn't end at goal")
    for a, b in zip(path, path[1:]):
        if max(abs(a.i - b.i), abs(a.j - b.j)) != 1:
            problems.append(f"jumps from ({a.i},{a.j}) to ({b.i},{b.j})"); break
    for c in path:
        if graph.check_collision(c.i, c.j):
            problems.append(f"hits a wall at ({c.i},{c.j})"); break
    if problems:
        failures += 1
        return "FAIL: " + "; ".join(problems)
    return f"ok, {len(path)} cells"

for map_file in sorted(glob.glob("data/*.map")):
    graph = GridGraph(map_file)
    free = [(i, j) for i in range(graph.width) for j in range(graph.height)
            if not graph.check_collision(i, j)]
    print(f"\n{map_file}  ({graph.width}x{graph.height}, {len(free)} free cells)")
    if len(free) < 2:
        print("  (no free space at this collision radius, skipping)"); continue
    for _ in range(3):
        (si, sj), (gi, gj) = random.sample(free, 2)
        start, goal = Cell(si, sj), Cell(gi, gj)
        for name, fn in (("BFS", breadth_first_search), ("A* ", a_star_search), ("DFS", depth_first_search)):
            t = time.time()
            path = fn(graph, start, goal)
            print(f"  {name} ({si},{sj})->({gi},{gj}): {check(graph, start, goal, path, name)}"
                  f"  [{time.time()-t:.2f}s]")

print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILED")
