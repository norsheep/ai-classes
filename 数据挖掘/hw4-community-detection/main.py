# XXX: DO NOT CHANGE THIS FILE. This is the driver code for the louvain algorithm.
# Changes made in this file WILL BE OVERWRITTEN during grading.
from itertools import permutations
from graph import WeightedDiGraph
from louvain import Louvain


def main():
    GRAPH_DATA_PATH = "./data/p2_edges.csv"
    TRUTH_PATH = "./data/label_reference.csv"
    OUTPUT_CSV_PATH = "./data/p2_prediction.csv"

    with open(TRUTH_PATH, "r", encoding="utf-8") as fi:
        _ = fi.readline()
        gt = [
            tuple(map(int,
                      line.strip().split(","))) for line in fi.readlines()
        ]

    gt_map = {x: y for x, y in gt}

    graph = WeightedDiGraph.from_csv_edges(GRAPH_DATA_PATH)
    louvain = Louvain(graph)

    # run main louvain algorithm
    res = louvain.louvain()

    # merge extra communities
    res = louvain.merge_communities(res, 5, gt_map)

    # reindex community ids based on the best accuracy we can get
    # this is done by enumerating over all possible permutations of 5 community ids
    best_acc = 0.0
    best_reindexer = (0, )
    for reindexer in permutations(range(5)):
        acc = 0.0
        for idx, lbl in gt:
            if reindexer[res[idx]] == lbl:
                acc += 1
        acc /= len(gt)
        if acc > best_acc:
            best_acc = acc
            best_reindexer = reindexer

    print(f"Best ACC: {best_acc}.")

    for x, y in res.items():
        res[x] = best_reindexer[y]

    # output the result to a csv file
    with open(OUTPUT_CSV_PATH, "w") as fo:
        fo.write("id, category\n")
        for x, y in sorted(res.items(), key=lambda x: x[0]):
            fo.write(f"{x}, {y}\n")

    print(f"Result outputted to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
