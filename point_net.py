import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from descriptors import PointConv
from descriptors import PointConv2
from descriptors import AttnPointConv

logger = logging.getLogger(__name__)


class PointNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointNet, self).__init__()
        self.convs = nn.ModuleList([
            PointConv2(in_channels, 64, 0.2),
            PointConv2(64, 64, 0.2),
            PointConv2(64, 64, 0.2),
            PointConv2(64, 128, 0.2),
            PointConv2(128, 1024, 0.2),
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(m.out_channels) for m in self.convs
        ])
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

        self.reset_parameters()

    def forward(self, pt_coordinates, pt_features=None):
        if pt_features is None:
            pt_features = pt_coordinates
        for idx, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            pt_features = F.relu(bn(conv(pt_features, pt_coordinates)))
        pt_features = F.adaptive_max_pool1d(pt_features, 1).squeeze(-1)
        pt_features = self.dp1(F.relu(self.bn1(self.fc1(pt_features))))
        pt_features = self.dp2(F.relu(self.bn2(self.fc2(pt_features))))
        pt_features = self.fc3(pt_features)
        return pt_features

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet(3, 40).to(device)
    print(model)
    try:
        import torchsummary
        torchsummary.summary(model, (3, 1024))
    except ImportError as e:
        pass
    except Exception as e:
        raise e
