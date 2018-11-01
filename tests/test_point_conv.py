"""Test script for PointConv."""
import unittest
import torch
from descriptors import PointConv


@unittest.skipUnless(torch.cuda.is_available(), "No cuda is available")
class TestPointConv(unittest.TestCase):

    def setUp(self):
        self.in_channels = 16
        self.out_channels = 64
        self.radius = 0.5
        self.max_samples = 12
        self.batch_size = 5
        self.num_points = 100

    def test_sample_octant(self):
        conv = PointConv(self.in_channels, self.out_channels, self.radius,
                         self.max_samples).cuda()
        pcs = 2*torch.rand(self.batch_size, 3, self.max_samples)-1.
        feats = torch.randn(self.batch_size, self.in_channels, self.max_samples)
        pcs, feats = pcs.cuda(), feats.cuda()
        sampled_pcs = conv.sample_each_octant(feats, pcs)
        self.assertSequenceEqual(sampled_pcs.size(),
                                 [self.batch_size, self.in_channels, 9])

    def test_group_points(self):
        conv = PointConv(self.in_channels, self.out_channels, self.radius,
                         self.max_samples).cuda()
        pcs = 2*torch.rand(self.batch_size, 3, self.num_points)-1.
        feats = torch.randn(self.batch_size, self.in_channels, self.num_points)
        pcs, feats = pcs.cuda(), feats.cuda()
        group_feats, group_pcs = conv.group_points(feats, pcs)
        self.assertSequenceEqual(group_feats.size(),
                                 [self.batch_size, self.in_channels,
                                  self.num_points, self.max_samples])
        self.assertSequenceEqual(group_pcs.size(),
                                 [self.batch_size, 3, self.num_points,
                                  self.max_samples])

    def test_forward(self):
        conv = PointConv(self.in_channels, self.out_channels, self.radius,
                         self.max_samples).cuda()
        pcs = 2*torch.rand(self.batch_size, 3, self.num_points)-1.
        feats = torch.randn(self.batch_size, self.in_channels, self.num_points)
        pcs, feats = pcs.cuda(), feats.cuda()
        out = conv(feats, pcs)
        self.assertSequenceEqual(
            out.size(), [self.batch_size, self.out_channels, self.num_points]
        )


if __name__ == "__main__":
    unittest.main()
