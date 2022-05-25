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

    def change_start(self, rotation: int):
        cities = list(self.cities.keys())
        self.at = cities[rotation % len(cities)]

    def num_cities(self) -> int:
        return len(self.cities)

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
            name = value_for_key_in("NAME", contents)
            edge_type = value_for_key_in("EDGE_WEIGHT_TYPE", contents)
            print(edge_type)
            if edge_type == "EUC_2D":
                return from_euclidean_2d(name, contents)
            elif edge_type == "EXPLICIT":
                edge_weight_format = value_for_key_in("EDGE_WEIGHT_FORMAT", contents)
                print(edge_weight_format)
                if edge_weight_format == 'FULL_MATRIX':
                    return from_full_matrix(name, contents)
                elif edge_weight_format == 'UPPER_ROW':
                    return from_upper_row(name, contents)


def from_euclidean_2d(name: str, contents: List[str]) -> TSPState:
    tsp = TSPState(name)
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


def from_full_matrix(name: str, contents: List[str]) -> TSPState:
    tsp = TSPState(name)
    start = contents.index("EDGE_WEIGHT_SECTION") + 1
    num_cities = int(value_for_key_in("DIMENSION", contents))
    tsp.at = str(start)
    for matrix_row in range(num_cities):
        tsp.cities[str(matrix_row)] = {str(i):float(n) for i, n in enumerate(contents[start + matrix_row].split())}
    return tsp


def from_upper_row(name: str, contents: List[str]) -> TSPState:
    tsp = TSPState(name)
    start = contents.index("EDGE_WEIGHT_SECTION") + 1
    num_cities = int(value_for_key_in("DIMENSION", contents))
    tsp.at = str(start)
    tsp.cities = {str(city):{} for city in range(num_cities)}
    for matrix_row in range(num_cities - 1):
        city = str(num_cities - matrix_row - 1)
        for i, edge in enumerate(contents[start + matrix_row].split()):
            tsp.cities[city][str(i)] = float(edge)
            tsp.cities[str(i)][city] = float(edge)
    print(tsp)
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


def run_expr(filename, verbosity, max_seconds, disable_branch_bound, enable_hybrid_queue, start):
    planner = Planner(copy_func=lambda s: s.minimal_clone(),
                      cost_func=lambda state, step: state.cities[state.at][step[1]])
    planner.declare_operators(go_to)
    planner.declare_methods(solve)
    tsp = tsp_from_file(filename)
    if start:
        tsp.change_start(start)
    return planner.anyhop(tsp, [('solve',)], max_seconds=max_seconds, verbose=verbosity,
                          enable_hybrid_queue=enable_hybrid_queue,
                          disable_branch_bound=disable_branch_bound)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"Usage: python3 {sys.argv[0]} [-v:verbosity] [-s:max seconds] [-branch_bound:enable|disable] [-queue:"
              f"stack|hybrid] [-rotate:starting city offset] [-expr:all] [tsp_problem_file]+")
    else:
        verbosity = find_verbosity(sys.argv)
        max_seconds = find_max_seconds(sys.argv)
        disable_branch_bound = find_tag_value(sys.argv, "branch_bound") == "disable"
        enable_hybrid_queue = find_tag_value(sys.argv, "queue") == "hybrid"
        for filename in sys.argv[1:]:
            if not filename.startswith("-"):
                if find_tag_value(sys.argv, "expr") == "all":
                    tsp = tsp_from_file(filename)
                    num_cities = tsp.num_cities()
                    best_costs = {True: [], False: []}
                    for start in range(num_cities):
                        for is_hybrid in best_costs:
                            plans = run_expr(filename, verbosity, max_seconds, disable_branch_bound, is_hybrid, start)
                            cost = plans[-1][1] if len(plans) > 0 else None
                            best_costs[is_hybrid].append(cost)
                    print(f'dfs_costs = {best_costs[False]}')
                    print(f'hybrid_costs = {best_costs[True]}')
                else:
                    plans = run_expr(filename, verbosity, max_seconds, disable_branch_bound, enable_hybrid_queue,
                                     find_tag_int(sys.argv, "rotate"))
                    for (plan, cost, time) in plans:
                        print(plan)
                        print(cost)
                    for (plan, cost, time) in plans:
                        print(f"Length: {len(plan)} Cost: {cost} Time: {time}")
                    print(len(plans), "total plans generated")
                    print([(cost, time) for (plan, cost, time) in plans])
