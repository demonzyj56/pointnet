"""Combin OctantSample and SelfBallPointQuery."""
import torch
import torch.nn as nn
from . import octant_query_cuda


class OctantQueryFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs, radius, max_samples_per_octant, default_value=-1):
        indices = torch.arange(pcs.size(-1), dtype=torch.int64,
                               device=pcs.device)
        indices = indices.view(1, -1, 1, 1).repeat(
            pcs.size(0), 1, 9, max_samples_per_octant
        )
        octant_query_cuda.forward(pcs, indices, radius, max_samples_per_octant)
        ctx.mark_non_differentiable(indices)
        return indices


class OctantQuery(nn.Module):
    """For input point coordinates, query for each octant maximal n points that
    are within radius r.

    This module combines the previous SelfBallPointQuery, which samples n points
    within radius r globally, and then assign the points to one of eight
    octants.
    """
    def __init__(self, radius, max_samples_per_octant):
        """
        Parameters
        ----------
        radius: float
            The radius of neighborhood of each point.
        max_samples_per_octant: int
            Maximum number for each octant.
        """
        super(OctantQuery, self).__init__()
        self.radius = radius
        self.max_samples_per_octant = max_samples_per_octant

    def forward(self, pcs):
        """
        Parameters
        ----------
        pcs: [batch_size, 3, num_points]
            Input point clouds.

        Returns
        -------
        indices: [batch_size, num_points, 9, max_samples_per_octant]
            Index of each point (with respect to the original order) for which
            the points lie at the corresponding octant. The last (9th)
            octant is the indices of the center point.  If there are fewer than
            max_samples_per_octant number of points, then the remaining entries
            are filled with the index of center point.
        """
        return OctantQueryFunction.apply(pcs, self.radius,
                                         self.max_samples_per_octant)
