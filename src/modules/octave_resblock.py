import torch.nn as nn
from .octave_block import OctaveBlock

class OctaveResBlock(nn.Module):
    def __init__(self, ch, alpha=0.5):
        super().__init__()

        self.conv1 = OctaveBlock(ch, ch, alpha)
        self.conv2 = OctaveBlock(ch, ch, alpha)

    def forward(self, x_h, x_l):

        identity_h, identity_l = x_h, x_l

        x_h, x_l = self.conv1(x_h, x_l)
        x_h, x_l = self.conv2(x_h, x_l)

        x_h = x_h + identity_h
        x_l = x_l + identity_l

        return x_h, x_l
