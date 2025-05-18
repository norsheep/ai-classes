import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader

from graph import Graph
from model import Node2Vec
from walker import BiasedRandomWalker
from loss import NegativeSamplingLoss


class Node2VecTrainer:
    """A trainer class for training the `Node2Vec` model.

    Args:
        `model` (Node2Vec): A `Node2Vec` model instance to be trained.
        `graph` (Graph): A graph object that contains the graph structure.
        `walker` (BiasedRandomWalker): A random walker.
            This walker should implement:
            1. a `walk(start, length)` method that returns a walk of length `length` from `start`.
            2. a `connected_nodes` attribute that lists all nodes with at least one edge.
        `n_negs` (int): Number of negative samples to be used in negative sampling.
        `n_epochs` (int): Number of epochs to train the model.
        `batch_size` (int): Batch size for training.
        `lr` (float): Learning rate for training.
        `device` (torch.device): Device to run the training.
        `walk_length` (int): Length of each random walk session. Defaults to 15.
        `window_size` (int): Window size for each training sample Defaults to 7.
        `n_walks_per_node` (int): Number of walks to start from each node. Defaults to 1.
    """

    def __init__(
        self,
        model: Node2Vec,
        graph: Graph,
        walker: BiasedRandomWalker,
        n_negs: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        walk_length: int = 15,
        window_size: int = 7,
        n_walks_per_node: int = 1,
    ):
        self.graph = graph
        self.num_nodes = graph.num_nodes

        self.model = model

        self.walker = walker
        self.walk_length = walk_length
        self.n_walks_per_node = n_walks_per_node

        self.n_negs = n_negs
        if window_size % 2 == 0:
            warnings.warn(
                "Window size should be odd. Adding 1 to window size.")
        self.window_size = (window_size // 2) * 2 + 1

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = self.create_optimizer(model, lr)
        self.loss_func = NegativeSamplingLoss()

    def _get_random_walk(self):
        """
        Performs a session of random walk using the `walker`,
        converts the walks into training samples,
        and returns a wrapped `DataLoader` for training.
        """
        walk_len = self.walk_length
        context_sz = self.window_size // 2

        # first perform random walks of length `walk_length`,
        # starting from each node in `connected_nodes`
        # and store the walks in `trajectories`
        trajectories = []
        for node in self.walker.connected_nodes:
            for _ in range(self.n_walks_per_node):
                trajectory = self.walker.walk(node, walk_len)
                trajectories.append(trajectory)

        # then convert the walks into training samples
        # we use a sliding window to extract training samples from each trajectory
        walks = []
        for trajectory in trajectories:
            for cent in range(context_sz, walk_len - context_sz):
                walks.append(trajectory[cent - context_sz:cent + context_sz +
                                        1])

        # finally wrap the training samples into a DataLoader
        walks = torch.LongTensor(walks)
        return DataLoader(walks, batch_size=self.batch_size, shuffle=True)

    def sample_neg_nodes(self, batch_sz: int, window_sz: int, n_negs: int):
        """Returns a batch of negative samples, to be used for NegativeSamplingLoss.

        Args:
            batch_sz (int): Batch size.
            window_sz (int): Window size.
            n_negs (int): Number of negative samples to be used.

        NOTE: We simply randomly sample from all nodes and ignore the fact that
        we might accidentally include positive edges during sampling.
        Since the graph is sparse, this should not cause much trouble.
        """
        return torch.randint(self.num_nodes, (batch_sz, window_sz * n_negs))

    def train_one_epoch(self, eid: int):
        """
        Perform one epoch of training.
        We first perform random walk to generate training samples,
        then train the model using these samples.
        """
        tot_loss = 0

        # sample random walks and convert them into training samples
        dataloader = self._get_random_walk()

        train_iter = tqdm(dataloader)
        for bid, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            B = batch.shape[0]  # batch size
            L = batch.shape[1]  # window size

            # TODO (Task 3)
            # Complete the training loop for Node2Vec.
            #
            # 1. Each batch is a tensor of shape (B, window_sz).
            #    Take one node as the current node and the rest as positive samples.
            # 2. Use `sample_neg_nodes()` to sample negative samples.
            # 3. Use `self.model` to convert node ids to embeddings.
            # 4. Use `self.loss_func` to compute the loss.

            ##################
            # Your Code Here #
            # Get the current node and positive samples
            cur_nodes = batch[:, L // 2]
            pos_nodes = batch[:, :L // 2].reshape(B, -1)
            pos_nodes = torch.cat((pos_nodes, batch[:, L // 2 + 1:]), dim=1)
            # Sample negative nodes
            neg_nodes = self.sample_neg_nodes(B, L, self.n_negs)
            # Get the embeddings
            cur_embs = self.model(cur_nodes)
            pos_embs = self.model(pos_nodes)
            neg_embs = self.model(neg_nodes)
            # Compute the loss
            # The loss function expects the embeddings to be of shape (B, n_pos, h) and (B, n_neg, h)
            ##################

            loss = self.loss_func(cur_embs, pos_embs, neg_embs)

            # End of TODO

            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
            avg_loss = tot_loss / (bid + 1)

            train_iter.set_description(
                f"Epoch: {eid:2d}, Loss: {avg_loss:.4f}")

        print(f"Epoch: {eid:2d}, Loss: {avg_loss:.4f}")

    def create_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        """Create an optimizer for training."""

        # TODO (Task 3)
        # Create an optimizer for the model.
        #
        # You may choose any optimizer you like,
        # and you may also change this function to accept other hyper-parameters.

        ##################
        # Your Code Here #
        """此处, 我探索了多种优化器和参数组合, 发现AdamW+weight_decay=1e-5效果最好"""
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        # weight_decay=1e-7：0.9180/0.9194/0.9251/0.9429/0.9292
        # weight_decay=1e-6：0.9137/0.9277/0.9056/0.9261/0.9150
        # weight_decay=1e-5：0.9172/0.9012/0.9214/0.9057/0.9156(比较稳定)
        # weight_decay = 1e-4：0.9215/0.8892/0.9029 不稳定
        # You can also use other optimizers like SGD, RMSprop, etc.
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # 0.89
        # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)  # 0.9148/0.9194
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=1e-5)  # 0.9301/0.9224
        ##################

        # Construct and return your optimizer.
        return optimizer

        # End of TODO

    def train(self):
        """Train the model for `n_epochs` epochs."""

        self.model.train()
        for eid in range(self.n_epochs):
            self.train_one_epoch(eid)
