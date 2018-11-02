"""Test gather points cuda module."""
import unittest
import torch
from modules.gather_points import GatherPoints

@unittest.skipUnless(torch.cuda.is_available(),
                     'GatherPoints is implemented only on GPU')
class TestGatherPoints(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.feature_size = 6
        self.num_points = 64
        self.index_size = 32
        self.device = torch.device('cuda')

    def test_gather_points_size(self):
        feats = torch.randn(self.batch_size, self.feature_size,
                            self.num_points).cuda()
        indices = torch.LongTensor(self.batch_size, self.index_size).cuda()
        indices = indices.random_() % self.num_points
        gathered = GatherPoints()(feats, indices)
        self.assertSequenceEqual(
            gathered.size(),
            [self.batch_size, self.feature_size, self.index_size]
        )

    def test_gather_points_forward(self):
        feats = torch.randn(self.batch_size, self.feature_size,
                            self.num_points, requires_grad=True).cuda()
        feats_torch = torch.zeros_like(feats, requires_grad=True)
        feats_torch.data.copy_(feats.data)
        indices = torch.LongTensor(self.batch_size, self.index_size).cuda()
        indices = indices.random_() % self.num_points
        gathered = GatherPoints()(feats, indices)
        gathered_torch = feats_torch.gather(
            2,
            indices.unsqueeze(1).repeat(1, feats.size(1), 1)
        )
        self.assertTrue(gathered.eq(gathered_torch).all().item())

    def test_gather_points_backward(self):
        # NOTE: you should not use cuda() over a variable (or a tensor with
        # requires_grad=True).  This creates an intermediate node and the
        # true leaf node lies on the cpu tensor, which is not released.
        feats = torch.randn(self.batch_size, self.feature_size,
                            self.num_points, requires_grad=True,
                            device=self.device)
        feats_torch = torch.zeros_like(feats, requires_grad=True)
        feats_torch.data.copy_(feats.data)
        indices = torch.cuda.LongTensor(self.batch_size, self.index_size)
        indices = indices.random_() % self.num_points
        gathered = GatherPoints()(feats, indices)
        gathered_torch = feats_torch.gather(
            2,
            indices.unsqueeze(1).repeat(1, feats_torch.size(1), 1)
        )
        self.assertIsNotNone(gathered_torch.grad_fn)
        self.assertIsNotNone(gathered.grad_fn)
        grad_tensor = torch.randn_like(gathered)
        gathered_torch.backward(grad_tensor)
        gathered.backward(grad_tensor)
        self.assertIsNotNone(feats_torch.grad)
        self.assertIsNotNone(feats.grad)
        self.assertLess(feats.grad.sub(feats_torch.grad).abs().max().item(),
                        1e-6)


if __name__ == "__main__":
    unittest.main()
