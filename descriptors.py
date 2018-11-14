"""Feature descriptors."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.self_ball_point_query import SelfBallPointQuery
from modules.octant_sample import OctantSample
from modules.gather_points import GatherPoints

logger = logging.getLogger(__name__)


class PointConv(nn.Module):

    def __init__(self, in_channels, out_channels, radius, max_samples,
                 bias=True):
        super(PointConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.max_samples = max_samples
        self.sbpq = SelfBallPointQuery(radius, max_samples)
        self.true_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=[1, 9], bias=bias)
        self.reset_parameters()

    def sample_each_octant(self, x, pcs):
        """For each input point, sample from each octant a sample point,
        in total nine points.

        Parameters
        ----------
        x: [batch_size, in_channels, max_samples]
            The input features for each group.
        pcs: [batch_size, 3, max_samples]
            The input point cloud coordinates for each group.

        Returns
        sampled_pcs: [batch_size, in_channels, 9]
        ---
        """
        # [batch_size, 8, max_samples]
        octant_idx_all = OctantSample()(pcs)
        octant_idx = octant_idx_all[..., 0]
        # Octants with no points are set to have no features.
        # The invalid index -1 is handled automatically by GatherPoints.
        octant_idx[octant_idx.eq(0)] = -1
        octant_idx = torch.cat([
            torch.zeros(octant_idx.size(0), 1, dtype=octant_idx.dtype,
                        device=octant_idx.device),
            octant_idx
        ], dim=1)
        out = GatherPoints()(x, octant_idx)
        return out

    def neighbor_index(self, pcs):
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
        return self.sbpq(pcs)

    def group_points(self, x, pcs):
        """For input point cloud features and coordinates, output the
        grouped point features and coordinates, where the coordinates are
        normalized with respect to each point.

        Parameters
        ----------
        x: [B, in_channels, N]
            Input features for each point.
        pcs: [B, 3, N]
            Input point cloud coordinates.

        Returns
        -------
        group_x: [B, in_channels, N, max_samples]
            Grouped features for each point.
        group_pcs: [B, 3, N, max_samples]
            Grouped coordinates for each point.
        """
        # (B, N, max_samples)
        group_idx = self.neighbor_index(pcs)
        group_idx = group_idx.view(group_idx.size(0), -1)
        x_out = GatherPoints()(x, group_idx).view(
            x.size(0), x.size(1), x.size(2), self.max_samples
        )
        pcs_out = GatherPoints()(pcs, group_idx).view(
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
        # [B, 3/in_channels, N, max_samples]
        group_x, group_pcs = self.group_points(x, pcs)
        # [B x N, 3/in_channels, max_samples]
        group_x = group_x.permute(0, 2, 1, 3).contiguous()
        group_x = group_x.view(-1, group_x.size(2), group_x.size(3))
        group_pcs = group_pcs.permute(0, 2, 1, 3).contiguous()
        group_pcs = group_pcs.view(-1, group_pcs.size(2), group_pcs.size(3))

        # [B x N, in_channels, 9]
        samples = self.sample_each_octant(group_x, group_pcs)
        samples = samples.view(x.size(0), x.size(2), samples.size(1),
                               samples.size(2))
        # [B, in_channels, N, 9]
        samples = samples.permute(0, 2, 1, 3)
        out = self.true_conv(samples).squeeze(-1)
        return out

    def reset_parameters(self):
        """Parameter initialization."""
        nn.init.xavier_uniform_(self.true_conv.weight)
        if self.true_conv.bias is not None:
            self.true_conv.bias.data.zero_()
