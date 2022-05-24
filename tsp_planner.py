from pyhop_anytime import *
from typing import *
import sys


class TSPState(State):
    def __init__(self, name: str):
        super().__init__(name)
        self.cities = {}
        self.visited = []
        self.at = None
        self.distance_traveled = 0

    def solved(self) -> bool:
        return set(self.visited) == set(self.cities.keys())

    def move_to(self, where):
        self.distance_traveled += self.cities[self.at][where]
        self.at = where
        self.visited.append(where)

    def unvisited(self) -> List[str]:
        return [city for city in self.cities if city not in self.visited]

    def minimal_clone(self) -> 'TSPState':
        cloned = TSPState(self.__name__)
        cloned.cities = self.cities
        cloned.visited = self.visited[:]
        cloned.at = self.at
        cloned.distance_traveled = self.distance_traveled
        return cloned


def value_for_key_in(key: str, tsp_input_lines: List[str]) -> str:
    for line in tsp_input_lines:
        if ':' in line:
            parts = [w.strip() for w in line.split(':')]
            if key.upper() == parts[0].upper():
                return parts[1]


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return sum((x - y)**2 for (x, y) in zip(p1, p2))**0.5


def tsp_from_file(filename: str) -> TSPState:
    with open(filename) as fin:
        contents = [line.strip() for line in fin]
        tsp_type = value_for_key_in("TYPE", contents)
        print("TSP type:", tsp_type)
        if tsp_type == "TSP":
            tsp = TSPState(value_for_key_in("NAME", contents))
            edge_type = value_for_key_in("EDGE_WEIGHT_TYPE", contents)
            print(edge_type)
            if edge_type == "EUC_2D":
                start = contents.index("NODE_COORD_SECTION") + 1
                coords = {}
                for i in range(start, len(contents)):
                    parts = contents[i].split()
                    if len(parts) == 3 and all(p for p in parts if p.isdigit()):
                        node, x, y = parts
                        if not tsp.at:
                            tsp.at = node
                        coords[node] = (float(x), float(y))
                        tsp.cities[node] = {}
                for city1, location1 in coords.items():
                    for city2, location2 in coords.items():
                        if city1 != city2:
                            tsp.cities[city1][city2] = euclidean_distance(location1, location2)
                return tsp


def solve(state: TSPState) -> TaskList:
    if state.solved():
        return TaskList(completed=True)
    else:
        return TaskList(options=[[('go_to', city), ('solve',)] for city in state.unvisited()])


def go_to(state: TSPState, target: str):
    if target != state.at:
        state.move_to(target)
        return state


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"Usage: python3 {sys.argv[0]} -v:[verbosity] -s:[max seconds] -branch_bound:[enable|disable] -queue:["
              f"stack|hybrid] [tsp_problem_file]+")
    else:
        verbosity = find_verbosity(sys.argv)
        max_seconds = find_max_seconds(sys.argv)
        disable_branch_bound = find_tag_value(sys.argv, "branch_bound") == "disable"
        enable_hybrid_queue = find_tag_value(sys.argv, "queue") == "hybrid"
        for filename in sys.argv[1:]:
            if not filename.startswith("-"):
                planner = Planner(copy_func=lambda s: s.minimal_clone(),
                                  cost_func=lambda state, step: state.cities[state.at][step[1]])
                planner.declare_operators(go_to)
                planner.declare_methods(solve)
                tsp = tsp_from_file(filename)
                plans = planner.anyhop(tsp, [('solve',)], max_seconds=max_seconds, verbose=verbosity,
                                       enable_hybrid_queue=enable_hybrid_queue,
                                       disable_branch_bound=disable_branch_bound)
                for (plan, cost, time) in plans:
                    print(plan)
                    print(cost)
                for (plan, cost, time) in plans:
                    print(f"Length: {len(plan)} Cost: {cost} Time: {time}")
                print(len(plans), "total plans generated")
