import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from argparse import ArgumentParser, Namespace

from graph import Graph
from metrics import calc_auc_score
from walker import BiasedRandomWalker
from node2vec_trainer import Node2VecTrainer
from model import Node2Vec, SigmoidPredictionHead
from data_utils import LinkPredictionDataset, LinkPredictionCollator

from typing import List, Dict, Tuple


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/p3_graph.csv")
    parser.add_argument("--testset_path", type=str, default="./data/p3_test.csv")
    parser.add_argument("--output_csv_path", type=str, default="./data/p3_prediction.csv")
    parser.add_argument("--reference_csv_path", type=str, default="./data/label_reference.csv")

    parser.add_argument("--epochs", type=int, default=3, help="Number of trainging epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for each training step"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of node embeddings"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the Node2Vec model"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size for Node2Vec, should be an odd number",
    )
    parser.add_argument("--walk_length", type=int, default=15, help="Length of each random walk")
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        default=10,
        help="Number of negative samples in NegativeSamplingLoss",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.2,
        help="Parameter controlling the probability of returning to the previous node",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=2.0,
        help="Parameter controlling the probability of visiting further neighbor nodes",
    )

    return parser.parse_args()


def load_test_data(path: str) -> List[Tuple[int, int]]:
    """
    Loads test data from a CSV file.

    XXX: Do NOT MODIFY this function.
         This function is used for test data loading.
    """
    with open(path, "r", encoding="utf-8") as fi:
        fi.readline()
        test_data = []
        for line in fi:
            _, src, dst = map(int, line.strip().split(","))
            test_data.append((src, dst))
    return test_data


def write_results(output_csv_path: str, predictions: List[int]):
    """
    Writes the prediction results to a CSV file.

    XXX: Do NOT MODIFY this function.
         This function is used for writing the prediction results.
    """
    with open(output_csv_path, "w", encoding="utf-8") as fo:
        fo.write("id,prediction\n")
        for i, pred in enumerate(predictions):
            fo.write(f"{i},{pred:.4f}\n")


def load_results(result_csv_path: str) -> Dict[int, float]:
    """
    Loads prediction results or reference labels from a CSV file.

    XXX: Do NOT MODIFY this function.
         This function is used for loading the prediction results or reference labels.
    """
    results = {}
    with open(result_csv_path, "r", encoding="utf-8") as fi:
        fi.readline()
        for line in fi:
            idx, truth = line.strip().split(",")
            results[int(idx)] = float(truth)
    return results


def predict(
    model: Node2Vec,
    classifier: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> List[int]:
    """Predicts the link probability for each test sample."""
    predictions = []
    for src, dst, _ in test_loader:
        src = src.to(device)
        dst = dst.to(device)

        src_emb = model(src)
        dst_emb = model(dst)
        pred = classifier(src_emb, dst_emb)

        predictions.extend(pred.tolist())

    return predictions


def train_node2vec(graph: Graph, model: Node2Vec, device: torch.device, args: Namespace):
    """
    Trains the Node2Vec model with the given arguments.
    """

    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    WINDOW_SIZE = args.window_size
    WALK_LENGTH = args.walk_length
    N_NEG_SAMPLES = args.num_neg_samples
    RETURN_PARAM_P = args.p
    IO_PARAM_Q = args.q

    # training Node2Vec
    walker = BiasedRandomWalker(graph, RETURN_PARAM_P, IO_PARAM_Q)
    node2vec_trainer = Node2VecTrainer(
        model,
        graph,
        walker,
        N_NEG_SAMPLES,
        N_EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        device,
        WALK_LENGTH,
        WINDOW_SIZE,
    )
    node2vec_trainer.train()

    return model


def main():
    args = parse_args()
    print(args)

    # data loading
    BATCH_SIZE = args.batch_size
    TRAIN_SET_PATH = args.dataset_path
    TEST_SET_PATH = args.testset_path
    OUTPUT_CSV_PATH = args.output_csv_path
    REFERENCE_CSV_PATH = args.reference_csv_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # build sparse graph
    graph = Graph.from_csv_edges(TRAIN_SET_PATH)
    num_nodes = graph.num_nodes

    # 1. Train Node2Vec node embeddings
    model = Node2Vec(num_nodes, args.embedding_dim).to(device)
    model = train_node2vec(graph, model, device, args)

    # 2. Link Prediction
    # load test data and make it a dataset
    test_data = load_test_data(TEST_SET_PATH)
    test_dataset = LinkPredictionDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=LinkPredictionCollator(),
    )

    # We directly use a dot-product + sigmoid to predict the link probability.
    classifier = SigmoidPredictionHead().to(device)
    predictions = predict(model, classifier, test_loader, device)

    # 3. Write results and compute auc scores on the validation set
    # write prediction results to output_csv_path
    write_results(OUTPUT_CSV_PATH, predictions)
    print(f"Result written to {OUTPUT_CSV_PATH}")

    # verify AUC score on the provided reference labels
    pred_dict = load_results(OUTPUT_CSV_PATH)
    truth_dict = load_results(REFERENCE_CSV_PATH)

    preds, gts = [], []
    for idx in truth_dict:
        preds.append(pred_dict[idx])
        gts.append(truth_dict[idx])

    auc_score = calc_auc_score(preds, gts)
    print(f"AUC Score on validation set: {auc_score:.4f}")


if __name__ == "__main__":
    main()
