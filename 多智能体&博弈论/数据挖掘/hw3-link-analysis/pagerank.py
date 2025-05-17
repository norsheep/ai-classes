import numpy as np


class PageRank:

    def _init_uniform(self, n_nodes: int) -> np.ndarray:
        """
        Initializes the PageRank vector uniformly.

        Args:
            n_nodes (int): Number of nodes in the graph.

        Returns:
            np.ndarray: Uniformly initialized PageRank vector.
        """

        # TODO (PageRank)
        # 1. Initialize the PageRank vector uniformly.
        init_rank = np.ones(n_nodes) / n_nodes
        return init_rank

    def _build_stochastic_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Initializes the stochastic matrix M.

        M[j, i] = 1 / out_degree(i) if (i, j) in E else 0

        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph.

        Returns:
            np.ndarray: Stochastic matrix M.
        """

        # TODO (PageRank)
        # 2. Initialize the stochastic matrix M.
        # M[j, i] = 1 / out_degree(i) if (i, j) in E else 0
        #
        # Hint:
        # A nested loop would suffice here,
        # but try optimizing the loops with vectorized matrix operations using numpy.
        # Not required, does not count towards your score, but encouraged ;-)

        # `adj_matrix[i, j] == 1` if a directed edge exists from `i` to `j`, and `adj_matrix[i, j] == 0`

        import copy
        tmp = copy.deepcopy(adj_matrix).T  # 复制一份矩阵
        # 计算完之后对每一列归一化,就是出度
        out_degree = np.sum(tmp, axis=0)
        out_degree[out_degree == 0] = 1
        # 归一化, 除以出度
        M = np.array([col / out_degree[i] for i, col in enumerate(tmp.T)]).T
        return M

    def page_rank(
        self,
        adj_matrix: np.ndarray,
        beta: float = 0.8,
        max_iter: int = 40,
    ) -> np.ndarray:
        """
        PageRank algorithm. Compute the PageRank scores of the nodes in the given graph.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph.
            beta (float): Damping factor. Default is 0.8.
            max_iter (int): Maximum number of iterations. Default is 40.

        Returns:
            np.ndarray: 1D array of shape (num_nodes,) containing the PageRank scores.
        """

        num_nodes = adj_matrix.shape[0]

        ranks = self._init_uniform(num_nodes)
        M = self._build_stochastic_matrix(adj_matrix)

        # TODO (PageRank)
        # 3. Complete the power iteration
        # Iteratively update the ranks util max_iter is reached.
        for _ in range(max_iter):
            ranks = beta * M @ ranks + (1 - beta) / num_nodes

        return ranks
