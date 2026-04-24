import torch.nn as nn
from src.modules.octave_resblock import OctaveResBlock
from src.modules.octave_transition import split_to_octave, merge_from_octave

class OctResNet(nn.Module):
    def __init__(self, in_ch=64, alpha=0.5):
        super().__init__()

        self.alpha = alpha

        self.layer1 = OctaveResBlock(in_ch, alpha)
        self.layer2 = OctaveResBlock(in_ch, alpha)

    def forward(self, x):

        x_h, x_l = split_to_octave(x, self.alpha)

        x_h, x_l = self.layer1(x_h, x_l)
        x_h, x_l = self.layer2(x_h, x_l)

        return x_h, x_l
