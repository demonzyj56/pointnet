"""Implement torch.gather, but faster."""
import torch
import torch.nn as nn

from . import gather_points_cuda


class GatherPointsFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, indices, impl='atomic'):
        assert impl in ('naive', 'reduction', 'atomic')
        out = torch.zeros(features.size(0), features.size(1), indices.size(-1),
                          dtype=features.dtype, device=features.device)
        gather_points_cuda.forward(features, indices, out)
        ctx.save_for_backward(features, indices)
        ctx.impl = impl
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, indices = ctx.saved_tensors
        grad_feats = torch.zeros_like(features)
        if ctx.impl == 'naive':
            gather_points_cuda.backward(grad_out, indices, grad_feats)
        elif ctx.impl == 'reduction':
            gather_points_cuda.backward_reduction(grad_out, indices, grad_feats)
        elif ctx.impl == 'atomic':
            gather_points_cuda.backward_atomicadd(grad_out, indices, grad_feats)
        return grad_feats, None, None


class GatherPoints(nn.Module):

    def __init__(self, impl='atomic'):
        super(GatherPoints, self).__init__()
        self.impl = impl

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
        return GatherPointsFunction.apply(features, indices, self.impl)
