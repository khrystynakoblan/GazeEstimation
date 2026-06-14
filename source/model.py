import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeEstimationNet(nn.Module):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.pose_fc1 = nn.Linear(3, 64)
        self.pose_bn1 = nn.BatchNorm1d(64)
        self.pose_fc2 = nn.Linear(64, 32)
        self.pose_bn2 = nn.BatchNorm1d(32)

        combined_dim = 128 * 4 * 7 + 32

        self.fc1    = nn.Linear(combined_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop1  = nn.Dropout(dropout_rate)

        self.fc2    = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop2  = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        p = F.relu(self.pose_bn1(self.pose_fc1(pose)))
        p = F.relu(self.pose_bn2(self.pose_fc2(p)))

        x = torch.cat([x, p], dim=1)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc_out(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
