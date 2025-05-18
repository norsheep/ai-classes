# XXX: DO NOT MODIFY THIS FILE.
# This is the driver file for the PageRank algorithm.
# Any updates to this file will be overwritten.
import os
import json
from pagerank import PageRank
from utils import load_digraph, find_top_and_bottom_k


def pagerank_pipeline(input_path: str, output_path: str) -> None:
    # load graph from file
    graph = load_digraph(input_path)

    # run pagerank
    solver = PageRank()
    ranks = solver.page_rank(graph, beta=0.8, max_iter=40)

    # find nodes with top-k highest and lowest pagerank scores
    top_k = 5
    top_k_nodes, bottom_k_nodes = find_top_and_bottom_k(ranks, top_k)

    top_k_scores = ranks[top_k_nodes]
    bottom_k_scores = ranks[bottom_k_nodes]

    # output results
    res = {"top_k": [], "bottom_k": []}
    for i in range(top_k):
        res["top_k"].append(
            {"node": int(top_k_nodes[i]) + 1, "pagerank": round(float(top_k_scores[i]), 6)}
        )

    for i in range(top_k):
        res["bottom_k"].append(
            {"node": int(bottom_k_nodes[i]) + 1, "pagerank": round(float(bottom_k_scores[i]), 6)}
        )

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
        output_path = os.path.join(RESULT_DIR, f"pagerank_{os.path.splitext(graph_file)[0]}.json")

        pagerank_pipeline(input_path, output_path)
        print(f"Processed {graph_file}, results saved to {output_path}.")


if __name__ == "__main__":
    main()
