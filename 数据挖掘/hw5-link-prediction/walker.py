import random
from abc import ABC, abstractmethod
from graph import Graph
from typing import List, Tuple


class RandomWalkerBase(ABC):

    def __init__(self, graph: Graph):
        self.graph = graph

        self.connected_nodes = self._get_connected_nodes()

    def _get_connected_nodes(self) -> List[int]:
        """
        Returns a list of nodes that have at least one edge connected to them.

        The dataset contains some isolated nodes,
        i.e., nodes with no edges connected to them.

        Since we cannot perform random walk on these isolated nodes,
        we manually exclude them from the list of connected nodes.

        XXX: Do NOT MODIFY THIS FUNCTION.
        """

        connected_nodes = []
        for node in range(self.graph.num_nodes):
            if self.graph.get_degree(node) > 0:
                connected_nodes.append(node)
        return connected_nodes

    def _normalize(self, weights: List[float]) -> List[float]:
        """Normalizes the weights so that the probabilities sum up to 1."""
        tot = sum(weights)
        return [p / tot for p in weights]

    @abstractmethod
    def walk(self, start: int, length: int) -> List[int]:
        """Perform a random walk on the graph of length `length`, starting from node `start`.

        Args:
            start (int): Starting node id
            length (int): Length of the random walk

        Returns:
            List[int]: A list of node ids representing the random walk path.
        """
        pass


class BiasedRandomWalker(RandomWalkerBase):
    """
    A biased random walker for generating random walks on a graph.

    Args:
        graph: A Graph object representing the graph.
        p (float, optional): The return parameter. Defaults to 1.0.
        q (float, optional): The in-out parameter. Defaults to 1.0.
    """

    def __init__(self, graph: Graph, p: float = 1.0, q: float = 1.0):
        super().__init__(graph)
        self.ret_p = p
        self.io_q = q

    def _normalize(self, weights: List[float]) -> List[float]:
        """Normalizes the weights so that the probabilities sum up to 1."""
        tot = sum(weights)
        return [p / tot for p in weights]

    def get_probs_uniform(self, curr: int) -> Tuple[List[int], List[float]]:
        """Returns a normalized uniform probability distribution
        over the neighbors of the current node (i.e., the node pointed by `vit`)

        NOTE: This function returns a tuple of two lists:

        - List of neighbor node IDs
        - List of probabilities corresponding to the neighbor nodes

        XXX: DO NOT MODIFY the signature and return format of this function.
             This function will be used for automated testing.
        """

        neighbors = self.graph.get_neighbors(curr)
        num_neighbors = len(neighbors)

        unnormalized_probs = [1.0] * num_neighbors
        probs = self._normalize(unnormalized_probs)

        return neighbors, probs

    def get_probs_biased(self, prev: int,
                         curr: int) -> Tuple[List[int], List[float]]:
        """Returns a normalized biased probability distribution
        over the neighbors of the current node (i.e., the node pointed by `vit`)

        NOTE: This function returns a tuple of two lists:

        - List of neighbor node IDs
        - List of probabilities corresponding to the neighbor nodes

        XXX: DO NOT MODIFY the signature and return format of this function.
             This function will be used for automated grading.
        """

        # TODO: (Task 1)
        # Get the `curr` node's neighbors
        # 1 line of code expected
        neighbors = self.graph.get_neighbors(curr)

        next_nodes = []
        unnormalized_probs = []
        # 遍历curr所有的邻居节点
        for next in neighbors:
            next_nodes.append(next)

            # TODO: (Task 1)
            # Compute the *unnormalized* transition probs for the biased random walk.
            # For each neighbor node `next` of `curr`, compute the unnormalized probablity of moving to `next` from `curr`, given a previous node `prev`.
            # Add the unnormalized probability to the list `unnormalized_probs`.
            # Hints:
            # 1. The unnormalized probability should take values from [1, 1/p, 1/q] depending on the relationship between `prev`, `curr` and `next`.

            ##################
            # Your Code Here #
            if next == prev:
                # 返回到上一个节点
                unnormalized_probs.append(1 / self.ret_p)
            elif self.graph.has_edge(next, prev):
                # 下一个点是prev的邻居
                unnormalized_probs.append(1)
            else:
                # 下一个点不是prev的邻居，但是是curr的邻居
                unnormalized_probs.append(1 / self.io_q)
            ##################

            # End of TODO

        # normalize the probabilities
        probs = self._normalize(unnormalized_probs)
        return next_nodes, probs

    def walk(self, start: int, length: int) -> List[int]:
        trajectory = [start]
        current_len = 1
        curr = start

        while current_len < length:
            # TODO (Task 1)
            # Implement the biased random walk.
            # Populate the list `trajectory` with the node ids of the random walk trajectory.
            #
            # Hints:
            # 1. For the first node, since there is no previous node, sample uniform at random.
            #    Use `get_probs_uniform`, which we have provided for you.
            # 2. For subsequent nodes, sample based on the biased probabilities.
            #    Use `get_probs_biased`, which you have just implemented.
            # 3. Use `random.choices()` to sample the next node and store it in `target`.

            ##################
            # Your Code Here #
            if current_len == 1:
                # 第一个点
                next_nodes, probs = self.get_probs_uniform(curr)
            else:
                # 其他点
                next_nodes, probs = self.get_probs_biased(trajectory[-2], curr)
            ##################

            # 根据概率权重采样
            target = random.choices(next_nodes, weights=probs, k=1)[0]
            trajectory.append(target)

            # End of TODO

            # move to the next node
            curr = target
            current_len += 1

        return trajectory
