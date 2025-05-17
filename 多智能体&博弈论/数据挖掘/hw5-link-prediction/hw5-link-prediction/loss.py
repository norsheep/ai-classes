import torch
import torch.nn as nn
from torch import Tensor


class NegativeSamplingLoss(nn.Module):
    """The negative sampling loss function.

    Args:
        eps (float, optional): For numerical stability. Defaults to 1e-6
    """

    def __init__(self, eps: float = 1e-6):

        super().__init__()
        self.eps = eps

    def forward(
        self,
        cur_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
    ) -> Tensor:
        # current: b,h
        # pos_embeddings: b,n_pos,h
        # neg_embeddings: b,n_neg,h

        B, H = cur_embs.shape
        cur_embs = cur_embs.reshape(B, 1,
                                    H)  # unsqueeze dim 1 for broadcasting
        pos_embs = pos_embs.reshape(B, -1, H)
        neg_embs = neg_embs.reshape(B, -1, H)

        # TODO (Task 2)
        # Implement the negative sampling loss.
        #
        # The implementation of this loss might vary depending on how you deal with
        # the positive and negative samples for each node.
        # We do not require your implementation to be exactly the same as the one
        # presented in the lecture slides, but it should be conceptually similar.
        #
        # Be careful that the loss sometimes goes to NaN.
        # We leave it up to you to figure out how to prevent / mitigate this issue.

        ##################
        # Your Code Here #
        # 对于nan的问题，使用torch.sigmoid来计算正负样本的分数
        # positive scores
        # print(f"cur_embs: {cur_embs.shape}, pos_embs: {pos_embs.shape}, neg_embs: {neg_embs.shape}")
        pos_scores = torch.bmm(cur_embs, pos_embs.transpose(1, 2)).squeeze(1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + self.eps)
        # negative scores
        neg_scores = torch.bmm(cur_embs, neg_embs.transpose(1, 2)).squeeze(1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + self.eps)
        # 对每个Loss求和
        # pos_loss = torch.sum(pos_loss, dim=1).sum(dim=0)
        # neg_loss = torch.sum(neg_loss, dim=1).sum(dim=0)
        # print(f"pos_scores: {pos_scores.shape}, neg_scores: {neg_scores.shape}")
        ##################
        print(f"pos_loss: {pos_loss.shape}, neg_loss: {neg_loss.shape}")

        loss = pos_loss.sum() + neg_loss.sum()

        # End of TODO

        # return loss
        return loss
