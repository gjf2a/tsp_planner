from pyhop_anytime import *
from typing import *


class TSPState(State):
    def __init__(self, name: str):
        super().__init__(name)
        self.cities = {}
        self.visited = []


def value_for_key_in(key: str, tsp_input_lines: List[str]) -> str:
    for line in tsp_input_lines:
        if ':' in line:
            parts = [w.strip() for w in line.split(':')]
            if key.upper() == parts[0].upper():
                return parts[1]


def euclidean_distance(p1: Tuple[float,float], p2: Tuple[float, float]) -> float:
    return sum((x - y)**2 for (x, y) in zip(p1, p2))**0.5


def tsp_from_file(filename: str) -> TSPState:
    with open(filename) as fin:
        contents = fin.readlines()
        tsp_type = value_for_key_in("TYPE", contents)
        print(tsp_type)
        if tsp_type == "TSP":
            tsp = TSPState(value_for_key_in("NAME", contents))
            edge_type = value_for_key_in("EDGE_WEIGHT_TYPE", contents)
            print(edge_type)
            if edge_type == "EUC_2D":
                start = contents.index("NODE_COORD_SECTION\n") + 1
                coords = {}
                for i in range(start, len(contents)):
                    parts = contents[i].split()
                    if len(parts) == 3 and all(p for p in parts if p.isdigit()):
                        node, x, y = parts
                        coords[node] = (float(x), float(y))
                        tsp.cities[node] = {}
                for city1, location1 in coords.items():
                    for city2, location2 in coords.items():
                        if city1 != city2:
                            tsp.cities[city1][city2] = euclidean_distance(location1, location2)
                return tsp


if __name__ == '__main__':
    tsp_1 = tsp_from_file("tsp_problems/a280.tsp")
    print(tsp_1)