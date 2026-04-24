import torch
import torch.nn.functional as F

class AvgPoolDownsample:
    def __init__(self, kernel_size=2):
        self.k = kernel_size

    def __call__(self, x):
        return F.avg_pool2d(x, kernel_size=self.k, stride=self.k)
