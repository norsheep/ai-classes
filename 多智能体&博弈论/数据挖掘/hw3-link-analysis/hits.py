import numpy as np
from typing import Tuple


class HITS:

    def hits(self,
             adj_matrix: np.ndarray,
             max_iter: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """HITS algorithm for computing hubs and authorities.

        Args:
            adj_matrix (np.ndarray): Adjacency matrix of the graph.
            max_iter (int, optional): Maximum number of iterations. Defaults to 40.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing two 1D arrays:
                - hubs: HITS hub scores.
                - authorities: HITS authority scores.
        """
        # TODO (HITS)
        # Complete the HITS algorithm.
        # 1. Initialize the hub and authority vectors `h` and `a`

        n = adj_matrix.shape[0]
        h = np.ones(n, dtype=float) / np.sqrt(n)  # HITS hub scores
        a = np.ones(n, dtype=float) / np.sqrt(n)

        # 2. Iteratively update and normalize the score vectors
        #    - h = A @ a
        #    - a = A^T @ h
        # A矩阵，i->j有边则A[i,j]=1
        for _ in range(max_iter):
            h_new = adj_matrix @ a  # Update hub scores
            h_new /= np.linalg.norm(h_new, 2)
            a_new = adj_matrix.T @ h
            a_new /= np.linalg.norm(a_new, 2)
            h = h_new
            a = a_new

        return h, a  # hubs, authorities
