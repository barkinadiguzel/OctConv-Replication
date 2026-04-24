import torch.nn.functional as F

class Upsample:
    def __init__(self, scale_factor=2):
        self.s = scale_factor

    def __call__(self, x):
        return F.interpolate(x, scale_factor=self.s, mode="nearest")
