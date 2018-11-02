"""Dataset of modelnet10/40."""
import os
import logging
import numpy as np
import torch
import torch.utils.data
import modelnet.provider as mp

logger = logging.getLogger(__name__)


class ModelNetCls(torch.utils.data.Dataset):
    """ModelNet 10/40 for 3D shape classification."""

    def __init__(self, root, modelnet40=True, train=True, transform=None,
                 num_points=1024):
        """
        Parameters
        ----------
        root:
            Root path of the modelnet data.
        modelnet40: bool
            If true, then modelnet40 dataset is loaded, otherwise modelnet10
            is used.
        train: bool
            Whether should load train set or test set.
        transform: callable
            Specify transform (augmentation) for point clouds before output.
        num_points: int
            Number of data points for each point cloud. This selects the first
            num_points for each point cloud.
        """
        assert modelnet40
        data = []
        labels = []
        if train:
            filenames = [
                os.path.join(root, 'ply_data_train{:d}.h5').format(i)
                for i in range(5)
            ]
        else:
            filenames = [
                os.path.join(root, 'ply_data_test{:d}.h5').format(i)
                for i in range(2)
            ]
        for f in filenames:
            assert os.path.exists(f), \
                'Path does not exists: {:s}'.format(f)
            d, l = mp.loadDataFile(f)
            data.append(d)
            labels.append(l)
        self.data = np.concatenate(data, axis=0)
        self.num_points = min(num_points, self.data.shape[1])
        self.data = self.data[:, :self.num_points, :]
        self.labels = np.concatenate(labels)
        assert self.data.shape[0] == len(self.labels)
        if train and (transform is not None):
            self.transform = transform
        else:
            self.transform = lambda x: x

    def __len__(self):
        """Length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index):
        """Get the i-th point cloud."""
        data = self.data[index]
        data = self.transform(data[np.newaxis, :])
        data = torch.from_numpy(data)
        return data.permute(0, 2, 1), self.labels[index]


class PCAugmentation(object):
    """Simple augmentation for point cloud data."""

    def __call__(self, pc):
        pc = mp.rotate_point_cloud(pc)
        pc = mp.rotate_perturbation_point_cloud(pc)
        jittered_pc = mp.random_scale_point_cloud(pc[:, :, :3])
        jittered_pc = mp.shift_point_cloud(jittered_pc)
        jittered_pc = mp.jitter_point_cloud(jittered_pc)
        pc[:, :, :3] = jittered_pc
        for idx in range(pc.shape[0]):
            pc[idx] = pc[idx, np.random.permutation(pc.shape[1]), :]
        return pc.astype(np.float32)


def collate_fn(batch):
    """Collate function for ModelNetCls."""
    data, labels = [], []
    for b in batch:
        data.append(b[0])
        labels.append(b[1][0])
    data = torch.cat(data, dim=0)
    labels = torch.LongTensor(labels)
    return data, labels
