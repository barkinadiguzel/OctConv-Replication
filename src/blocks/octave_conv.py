import torch
import torch.nn as nn

from .pooling import AvgPoolDownsample
from .upsample import Upsample


class OctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, alpha=0.5, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.alpha = alpha

        self.in_ch_h = int((1 - alpha) * in_ch)
        self.in_ch_l = in_ch - self.in_ch_h

        self.out_ch_h = int((1 - alpha) * out_ch)
        self.out_ch_l = out_ch - self.out_ch_h

        self.pool = AvgPoolDownsample(2)
        self.up = Upsample(2)

        self.conv_hh = nn.Conv2d(self.in_ch_h, self.out_ch_h, kernel_size, stride, padding)
        self.conv_hl = nn.Conv2d(self.in_ch_h, self.out_ch_l, kernel_size, stride, padding)

        self.conv_lh = nn.Conv2d(self.in_ch_l, self.out_ch_h, kernel_size, stride, padding)
        self.conv_ll = nn.Conv2d(self.in_ch_l, self.out_ch_l, kernel_size, stride, padding)

    def forward(self, x_h, x_l):
        y_hh = self.conv_hh(x_h)

        y_hl = self.conv_hl(self.pool(x_h))

        y_ll = self.conv_ll(x_l)

        y_lh = self.up(self.conv_lh(x_l))

        y_h = y_hh + y_lh
        y_l = y_ll + y_hl

        return y_h, y_l
