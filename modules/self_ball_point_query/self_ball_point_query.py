import torch
import torch.nn as nn

from . import self_bpq_cuda


class SelfBallPointQueryFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, radius, max_samples):
        group_idx = torch.zeros(pcs.size(0), pcs.size(-1), max_samples,
                                dtype=torch.int64, device=pcs.device)
        self_bpq_cuda.forward(pcs, group_idx, radius, max_samples)
        ctx.mark_non_differentiable(group_idx)
        return group_idx


class SelfBallPointQuery(nn.Module):

    def __init__(self, radius, max_samples):
        super(SelfBallPointQuery, self).__init__()
        self.radius = radius
        self.max_samples = max_samples

    def forward(self, pcs):
        return SelfBallPointQueryFunction.apply(pcs, self.radius,
                                                self.max_samples)

