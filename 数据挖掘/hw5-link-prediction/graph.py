from collections import defaultdict
from typing import List, Dict, Set, Tuple


class Graph:
    """An *undirected* graph represented as an adjacency list."""

    def __init__(self, edge_list: List[Tuple[int, int, int]]):
        self.nodes = set()
        self.neighbors: Dict[int, Set[int]] = defaultdict(set)
        self.edges: Dict[Tuple[int, int], float] = defaultdict(int)
        self.degrees: Dict[int, int] = defaultdict(int)

        for src, dst in edge_list:
            self.add_undirected_edge(src, dst)

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph."""
        # add 1 because nodes are 0-indexed
        return max(self.nodes) + 1

    def add_node(self, node: int):
        """Adds a node to the graph."""
        self.nodes.add(node)

    def add_undirected_edge(self, src: int, dst: int):
        """Adds an *undirected* edge to the graph."""
        self.add_node(src)
        self.add_node(dst)

        self.neighbors[src].add(dst)
        self.neighbors[dst].add(src)

    def get_neighbors(self, node: int) -> List[int]:
        """Returns a sorted list of neighbors for the given node."""
        return list(sorted(self.neighbors[node]))

    def has_edge(self, src: int, dst: int) -> bool:
        """Checks if an edge exists between src and dst."""
        return dst in self.neighbors[src]

    def get_degree(self, node: int) -> int:
        """Get the degree of a node."""
        return len(self.neighbors[node])

    def copy(self) -> "Graph":
        """Returns a copy of the graph."""
        return Graph([(src, dst, weight) for (src, dst), weight in self.edges.items()])

    @classmethod
    def from_csv_edges(cls, csv_path: str, has_header: bool = True) -> "Graph":
        """
        Builds a weighted undirected graph from a CSV file containing edges.
        The CSV file should contain two columns: src, dst.
        """
        edge_list = []
        with open(csv_path, "r", encoding="utf-8") as fi:
            if has_header:
                _ = fi.readline()  # skip csv header
            for line in fi.readlines():
                entries = line.strip().split(",")
                if len(entries) == 2:
                    src, dst = map(int, entries)
                else:
                    raise ValueError(f"Invalid edge entry: {entries}")
                edge_list.append((src, dst))

        return cls(edge_list)
