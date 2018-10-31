"""Feature descriptors."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ball_point_query import BallPointQuery

logger = logging.getLogger(__name__)


class PointConv(nn.Module):

    def __init__(self, in_channels, out_channels, radius, max_samples,
                 bias=True):
        super(PointConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.max_samples = max_samples
        self.bpq = BallPointQuery(radius, max_samples)
        self._true_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=[1, 9], bias=bias)
        nn.init.xavier_uniform_(self._true_conv.weight)
        if bias:
            self._true_conv.bias.data.zero_()

    def _sample_each_octant(self, x, pcs):
        """For each input point, sample from each octant a sample point,
        in total nine points."""
        raise NotImplementedError

    def _neighbor_index(self, pcs):
        """For input point cloud coordinates, output the grouped index
        belonging to the neighborhood of each point.  Each neighborhood
        contains the center point as the first index.

        Parameters
        ----------
        pcs: input point cloud coordinates, [B, 3, N]

        Returns
        -------
        group_idx: index of points, [B, N, max_samples]
        """
        # TODO(leoyolo): ensure that center point is always sampled.
        return self.bpq(pcs, pcs)

    def _group_points(self, x, pcs):
        """For input point cloud features and coordinates, output the
        grouped point features and coordinates, where the coordinates are
        normalized with respect to each point."""
        # TODO(leoyolo): gather is slow.
        # (B, N, max_samples)
        group_idx = self._neighbor_index(pcs)
        # (B, 1, N x max_samples)
        group_idx = group_idx.view(group_idx.size(0), -1).unsqueeze(1)
        x_out = x.gather(2, group_idx.repeat(1, x.size(1), 1)).view(
            x.size(0), x.size(1), x.size(2), self.max_samples
        )
        pcs_out = pcs.gather(2, group_idx.repeat(1, 3, 1)).view(
            pcs.size(0), pcs.size(1), pcs.size(2), self.max_samples
        )
        pcs_out.sub_(pcs.unsqueeze(-1).expand_as(pcs_out))
        return x_out, pcs_out

    def forward(self, x, pcs):
        """
        Parameters
        ----------
        x: input features, [B, in_channels, N]
        pcs: input point cloud coordinates, [B, 3, N]

        Returns
        -------
        y: output features, [B, out_channels, N]
        """
        # [B, in_channels, N, 9]
        samples = self._sample_each_octant(x, pcs)
        return self._true_conv(samples)
