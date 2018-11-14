"""Combin OctantSample and SelfBallPointQuery."""
import torch
import torch.nn as nn
from . import octant_query_cuda


class OctantQueryFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, radius, max_samples_per_octant, default_value=-1):
        indices = torch.empty(pcs.size(0), 8, max_samples_per_octant,
                              dtype=torch.int64, device=pcs.device)
        indices.fill_(default_value)
        octant_query_cuda(pcs, indices, radius, max_samples_per_octant)
        ctx.mark_non_differentiable(indices)
        return indices


class OctantQuery(nn.Module):
    """For input point coordinates, query for each octant maximal n points that
    are within radius r.

    This module combines the previous SelfBallPointQuery, which samples n points
    within radius r globally, and then assign the points to one of eight
    octants.
    """
    def __init__(self, radius, max_samples_per_octant, default_value=-1):
        super(OctantQuery, self).__init__()
        self.radius = radius
        self.max_samples_per_octant = max_samples_per_octant
        self.default_value = default_value

    def forward(self, pcs):
        """
        Parameters
        ----------
        pcs: [batch_size, 3, num_points]
            Input point clouds.

        Returns
        -------
        indices: [batch_size, 8, max_samples_per_octant]
            Index of each point (with respect to the original order) for which
            the points lie at the corresponding octant.  When there are fewer
            than max_samples_per_octant, then the remaining entries are filled
            with default_value.
        """
        return OctantQueryFunction.apply(pcs, self.radius,
                                         self.max_samples_per_octant,
                                         self.default_value)
