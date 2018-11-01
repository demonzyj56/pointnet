"""Test script for OctantSample."""
import unittest
import torch
from modules.octant_sample import OctantSample


@unittest.skipUnless(torch.cuda.is_available(),
                     'OctantSample works only for cuda tensors')
class TestOctantSample(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10
        self.max_samples = 12
        self.ocs = OctantSample()

    def test_size(self):
        x = torch.randn(self.batch_size, 3, self.max_samples).cuda()
        octant_idx = self.ocs(x)
        self.assertSequenceEqual(octant_idx.size(),
                                 [self.batch_size, 8, self.max_samples])

    def test_exhaustion(self):
        """All indices are properly allocated to 8 octants."""
        x = torch.randn(self.batch_size, 3, self.max_samples).cuda()
        octant_idx = self.ocs(x)
        for i in range(self.batch_size):
            collected = []
            for j in range(8):
                for k in range(self.max_samples):
                    idx = octant_idx[i, j, k].item()
                    if idx != 0:
                        self.assertTrue(idx not in collected)
                        collected.append(idx)
            self.assertSequenceEqual(sorted(collected),
                                     range(1, self.max_samples))

    def test_octant_sanity(self):
        """If all points lay in the same octant."""
        def _test_octant_sanity_once():
            sign = torch.rand(1, 3).ge(0.5)
            x = torch.rand(self.batch_size, 3, self.max_samples).cuda()
            for i in range(3):
                if torch.rand(1).ge(0.5).item():
                    x[:, i, :] = -x[:, i, :]
            octant_idx = self.ocs(x).cpu()
            octant = None
            for i in range(self.batch_size):
                num_nonzeros = octant_idx[i].ne(0).sum(dim=1).view(-1)
                nonzero_idx = num_nonzeros.nonzero().view(-1)
                self.assertEqual(nonzero_idx.numel(), 1)
                if octant is None:
                    octant = nonzero_idx.item()
                else:
                    self.assertEqual(octant, nonzero_idx.item())
                self.assertEqual(num_nonzeros[octant].item(),
                                 self.max_samples-1)

        for _ in range(10):
            _test_octant_sanity_once()


if __name__ == "__main__":
    unittest.main()
