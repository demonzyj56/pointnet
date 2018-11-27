"""OctantQuery with mask."""
import torch
import torch.nn as nn
from . import octant_query_mask_cuda


class OctantQueryMaskFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, radius, max_samples_per_octant, default_value=-1):
        indices = torch.empty(pcs.size(0), pcs.size(-1), 8,
                              max_samples_per_octant, dtype=torch.int64,
                              device=pcs.device)
        indices.fill_(default_value)
        masks = torch.empty_like(indices, dtype=torch.uint8)
        masks.fill_(0)
        octant_query_mask_cuda.forward(pcs, indices, masks, radius,
                                       max_samples_per_octant)
        ctx.mark_non_differentiable(indices)
        ctx.mark_non_differentiable(masks)
        return indices, masks


class OctantQueryMask(nn.Module):

    def __init__(self, radius, max_samples_per_octant, default_value=-1):
        super(OctantQueryMask, self).__init__()
        self.radius = radius
        self.max_samples_per_octant = max_samples_per_octant
        self.default_value = default_value

    def forward(self, pcs):
        return OctantQueryMaskFunction.apply(pcs, self.radius,
                                             self.max_samples_per_octant,
                                             self.default_value)
