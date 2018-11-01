import torch
import torch.nn as nn

from . import octant_sample_cuda


class OctantSampleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pcs):
        octant_idx = torch.zeros(pcs.size(0), 8, pcs.size(-1),
                                 dtype=torch.int64, device=pcs.device)
        octant_sample_cuda.forward(pcs, octant_idx)
        ctx.mark_non_differentiable(octant_idx)
        return octant_idx


class OctantSample(nn.Module):
    """For input point cloud coordinates, assign each point to one of eight
    octants and return the indices.

    Parameters
    ----------
    pcs: [batch_size, 3, max_samples]
        Input point cloud coordinates.  Each batch has number of `max_samples`
        points, and the first point is the center (assumed to be (0,0,0)).

    Returns
    -------
    octant_idx: [batch_size, 8, max_samples]
        Index of each point belonging to one of eight octants.  For each octant
        of each batch, there is `max_samples` number of indices, where the
        nonzero ones are the true indices belonging to that octant.  If only
        zeros appear, that means there is no point belonging to that octant.
    """

    def forward(self, pcs):
        assert pcs.size(1) == 3
        # [batch_size, 8, max_samples]
        octant_idx = OctantSampleFunction.apply(pcs)
        octant_idx = torch.sort(octant_idx, dim=-1, descending=True)[0]
        return octant_idx
