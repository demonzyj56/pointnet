"""Implement torch.gather, but faster."""
import torch
import torch.nn as nn

from . import gather_points_cuda


class GatherPointsFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, indices):
        out = torch.zeros(features.size(0), features.size(1), indices.size(-1),
                          dtype=features.dtype, device=features.device)
        gather_points_cuda.forward(features, indices, out)
        ctx.save_for_backward(features, indices)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, indices = ctx.saved_tensors
        grad_feats = torch.zeros_like(features)
        gather_points_cuda.backward(grad_out, indices, grad_feats)
        return grad_feats, None


class GatherPoints(nn.Module):

    def forward(self, features, indices):
        """Gather features from indices.

        Parameters
        ----------
        features: [B, C, N]
            Input features.
        indices: [B, M]
            Input indices. For each batch the index ranges between 0...N-1.

        Returns
        -------
        gathered: [B, C, M]
            Output gathered features.
        """
        return GatherPointsFunction.apply(features, indices)
