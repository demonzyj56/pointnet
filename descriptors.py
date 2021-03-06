"""Feature descriptors."""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.self_ball_point_query import SelfBallPointQuery
from modules.octant_sample import OctantSample
from modules.gather_points import GatherPoints
from modules.octant_query import OctantQuery
from modules.octant_query_mask import OctantQueryMask

logger = logging.getLogger(__name__)


class AttnPointConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, mu, radius,
                 max_samples_per_octant, bias=True):
        super(AttnPointConv, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.mu = mu
        self.radius = radius
        self.ms = max_samples_per_octant
        self.octant_query = OctantQueryMask(radius, max_samples_per_octant)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels,
                                        groups=out_channels,
                                        kernel_size=[1, 8], bias=bias)
        self.value_enc = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   bias=False)
        self.query_enc = nn.Conv1d(8*in_channels, 8*mid_channels,
                                   kernel_size=1, groups=8, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for m in (self.value_enc, self.query_enc, self.depthwise_conv):
            nn.init.xavier_uniform_(m.weight)
        if self.depthwise_conv.bias is not None:
            self.depthwise_conv.bias.data.zero_()

    def forward(self, x, pcs):
        batch_size = x.size(0)
        num_points = x.size(-1)
        # [batch_size, num_points, 8, max_samples]
        octant_idx, octant_mask = self.octant_query(pcs)
        octant_mask = octant_mask ^ 1

        # [batch_size, in_channels, (num_points x 8 x max_samples)]
        gathered = GatherPoints()(x, octant_idx.view(octant_idx.size(0), -1))
        # [(batch_size x num_points), (8 x in_channels), max_samples]
        gx = gathered.view(batch_size, self.in_channels, num_points*8, self.ms).permute(0, 2, 1, 3)
        gx = gx.contiguous().view(batch_size*num_points, 8*self.in_channels, self.ms)
        # [(batch_size x num_points), (8 x mid_channels), max_samples]
        query = self.query_enc(gx)
        # [(batch x num_points x 8), mid_channels, max_samples]
        query = query.view(-1, self.mid_channels, self.ms)
        # [(batch x num_points x 8), max_samples, max_samples]
        qqt = query.transpose(1, 2).bmm(query)

        if self.mu > 0:
            # [batch_size, 3, num_points, (8 x max_samples)]
            gp = GatherPoints()(pcs, octant_idx.view(octant_idx.size(0), -1))
            gp = gp.view(batch_size, 3, num_points, 8*self.ms)
            gp.sub_(pcs.unsqueeze(-1).expand_as(gp))
            # [(batch_size x num_points x 8), 3, max_samples]
            gp = gp.view(batch_size, 3, num_points*8, self.ms).permute(0, 2, 1, 3)
            gp = gp.contiguous().view(-1, 3, self.ms)
            # [(batch_size x num_points x 8), max_samples, max_samples]
            xxt = gp.transpose(1, 2).bmm(gp)
            qqt.add_(xxt.mul(self.mu))

        # [(batch_size x num_points x 8), max_samples, max_samples], masked on
        # second dimension.
        qqt.div_(math.sqrt(self.mid_channels))
        qqt.masked_fill_(octant_mask.view(-1, self.ms, 1).expand_as(qqt), -math.inf)
        qqt = qqt.softmax(dim=1)

        # [batch_size, out_channels, (num_points x 8 x max_samples)]
        values = self.value_enc(gathered)
        # [(batch_size x num_points x 8), out_channels, max_samples]
        values = values.view(batch_size, self.out_channels, num_points*8, self.ms).permute(0, 2, 1, 3)
        values = values.contiguous().view(-1, self.out_channels, self.ms)
        # [(batch_size x num_points x 8), out_channels, max_samples]
        octant_feats = values.bmm(qqt)
        # [(batch_size x num_points x 8), out_channels, max_samples], masked on
        # the last dimension.
        octant_feats.masked_fill_(octant_mask.view(-1, 1, self.ms).expand_as(octant_feats), -math.inf)
        # After max pooling all -inf entries will be removed.
        # [(batch_size x num_points x 8), out_channels]
        octant_feats = F.max_pool1d(octant_feats, self.ms).squeeze(-1)
        # [batch_size, out_channels, num_points, 8]
        octant_feats = octant_feats.view(batch_size, num_points, 8, self.out_channels).permute(0, 3, 1, 2)
        # [batch_size, out_channels, num_points]
        out = self.depthwise_conv(octant_feats).squeeze(-1)
        return out


class PointConv2(nn.Module):
    """Yet another impl using OctantQuery."""

    def __init__(self, in_channels, out_channels, radius, bias=True):
        super(PointConv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.octant_query = OctantQuery(radius, 1)
        self.true_conv = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=[1, 9], bias=bias)
        self.reset_parameters()

    def forward(self, x, pcs):
        # [ batch_size, num_points, 9, ]
        octant_idx = self.octant_query(pcs).squeeze(-1)
        grouped_x = GatherPoints()(x, octant_idx.view(octant_idx.size(0), -1))
        # [ batch_size, in_channels, num_points, 9 ]
        grouped_x = grouped_x.view(
            grouped_x.size(0), grouped_x.size(1), octant_idx.size(1), 9
        )
        out = self.true_conv(grouped_x).squeeze(-1)
        return out

    def reset_parameters(self):
        """Parameter initialization."""
        nn.init.xavier_uniform_(self.true_conv.weight)
        if self.true_conv.bias is not None:
            self.true_conv.bias.data.zero_()


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
