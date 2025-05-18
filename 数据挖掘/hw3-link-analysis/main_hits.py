# XXX: DO NOT MODIFY THIS FILE.
# This is the driver file for the HITS algorithm.
# Any updates to this file will be overwritten.
import os
import json
from hits import HITS
from utils import load_digraph, find_top_and_bottom_k


def hits_pipeline(input_path: str, output_path: str) -> None:
    # load graph from file
    graph = load_digraph(input_path)

    # run hits
    solver = HITS()
    hubs, authorities = solver.hits(graph, max_iter=40)

    # find nodes with top-k highest and lowest pagerank scores
    top_k = 5
    top_k_a_nodes, bot_k_a_nodes = find_top_and_bottom_k(authorities, top_k)
    top_k_h_nodes, bot_k_h_nodes = find_top_and_bottom_k(hubs, top_k)

    # output results
    res = {
        "authorities": {
            "top_k": (top_k_a_nodes + 1).tolist(),
            "bottom_k": (bot_k_a_nodes + 1).tolist(),
        },
        "hubs": {
            "top_k": (top_k_h_nodes + 1).tolist(),
            "bottom_k": (bot_k_h_nodes + 1).tolist(),
        },
    }

    with open(output_path, "w") as f:
        json.dump(res, f, indent=4)


def main():
    DATA_DIR = "data"
    RESULT_DIR = "results"

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    GRAPH_FILES = ["graph-small.txt", "graph-full.txt", "graph-randnew.txt"]
    for graph_file in GRAPH_FILES:
        input_path = os.path.join(DATA_DIR, graph_file)
        output_path = os.path.join(RESULT_DIR, f"hits_{os.path.splitext(graph_file)[0]}.json")

        hits_pipeline(input_path, output_path)
        print(f"Processed {graph_file}, results saved to {output_path}.")


if __name__ == "__main__":
    main()
