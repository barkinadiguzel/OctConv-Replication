import torch.nn as nn
from src.blocks.octave_conv import OctaveConv

class OctaveBlock(nn.Module):
    def __init__(self, in_ch, out_ch, alpha=0.5):
        super().__init__()

        self.conv = OctaveConv(in_ch, out_ch, alpha)
        self.bn_h = nn.BatchNorm2d(int((1-alpha)*out_ch))
        self.bn_l = nn.BatchNorm2d(out_ch - int((1-alpha)*out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l):
        y_h, y_l = self.conv(x_h, x_l)

        y_h = self.relu(self.bn_h(y_h))
        y_l = self.relu(self.bn_l(y_l))

        return y_h, y_l
