import numpy as np
from typing import Tuple


# XXX: DO NOT MODIFY THIS FUNCTION.
def load_digraph(path: str) -> np.ndarray:
    """
    Loads a directed graph from the given file.
    The file should contain two columns separated by space or tab,
    where the first column denotes source_id and the second column denotes target_id.

    NOTE: (1) Assumes node id starts from 1.
          (2) Duplicated edges are ignored.

    Args:
        path (str): Path to the file containing the graph edges.

    Returns:
        np.ndarray: Adjacency matrix of the directed graph.
    """

    with open(path, "r") as f:
        lines = f.readlines()

    # remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]

    # parse edges
    edges = [tuple(map(int, line.split())) for line in lines]

    # find largest node id
    # assumes the largest node id is the largest number in the file
    max_node_id = max(max(edge) for edge in edges)

    # adj matrix
    adj_matrix = np.zeros((max_node_id, max_node_id), dtype=int)

    for src, tgt in edges:
        adj_matrix[src - 1, tgt - 1] = 1  # convert to 0-based index

    return adj_matrix


# XXX: DO NOT MODIFY THIS FUNCTION.
def find_top_and_bottom_k(array: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find the top-k and bottom-k indices of the given array.

    Args:
        array (np.ndarray): 1D array of scores.
        k (int): Number of top and bottom elements to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the top-k and bottom-k indices.
    """
    # check array shape
    if len(array.shape) != 1:
        raise ValueError("Input array must be 1D.")

    sorted_indices = np.argsort(array)[::-1]  # sort in descending order
    top_k_indices = sorted_indices[:k]
    bottom_k_indices = sorted_indices[-k:][::-1]  # re-arrange bottom-k in ascending order

    return top_k_indices, bottom_k_indices
