import torch.nn as nn
import torch

class OctClassifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x_h, x_l):

        x = torch.cat([x_h, x_l], dim=1)
        x = self.pool(x).flatten(1)

        return self.fc(x)
