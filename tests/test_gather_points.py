"""Test gather points cuda module."""
import unittest
import numpy as np
import torch
from modules.gather_points import GatherPoints


@unittest.skipUnless(torch.cuda.is_available(),
                     'GatherPoints is implemented only on GPU')
class TestGatherPoints(unittest.TestCase):

    def setUp(self):
        self.batch_size = 16
        self.feature_size = 64
        self.num_points = 1024
        self.index_size = 65536
        self.device = torch.device('cuda')

    def _assert_float_tensor_close(self, tensor, ref):
        """Assert that two pytorch tensors are close."""
        self.assertIs(tensor.dtype, ref.dtype)
        nomin = tensor.sub(ref).norm().item()
        denom = ref.norm().item()
        # We don't check for float64 since float32 is enough.
        self.assertLess(nomin/denom, np.finfo(np.float32).eps)

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
        self._assert_float_tensor_close(gathered, gathered_torch)

    def test_gather_points_backward_naive(self):
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
        gathered = GatherPoints('naive')(feats, indices)
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
        self._assert_float_tensor_close(feats, feats_torch)

    def test_gather_points_backward_reduction(self):
        feats = torch.randn(self.batch_size, self.feature_size,
                            self.num_points, requires_grad=True,
                            device=self.device)
        feats_torch = torch.zeros_like(feats, requires_grad=True)
        feats_torch.data.copy_(feats.data)
        indices = torch.cuda.LongTensor(self.batch_size, self.index_size)
        indices = indices.random_() % self.num_points
        gathered = GatherPoints('reduction')(feats, indices)
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
        self._assert_float_tensor_close(feats, feats_torch)

    def test_gather_points_backward_atomicadd(self):
        feats = torch.randn(self.batch_size, self.feature_size,
                            self.num_points, requires_grad=True,
                            device=self.device)
        feats_torch = torch.zeros_like(feats, requires_grad=True)
        feats_torch.data.copy_(feats.data)
        indices = torch.cuda.LongTensor(self.batch_size, self.index_size)
        indices = indices.random_() % self.num_points
        gathered = GatherPoints('atomic')(feats, indices)
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
        self._assert_float_tensor_close(feats, feats_torch)


if __name__ == "__main__":
    unittest.main()
