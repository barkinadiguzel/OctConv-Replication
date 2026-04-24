def split_to_octave(x, alpha):
    c = x.shape[1]
    c_h = int((1-alpha)*c)

    x_h = x[:, :c_h, :, :]
    x_l = x[:, c_h:, :, :]

    return x_h, x_l


def merge_from_octave(x_h, x_l):
    return torch.cat([x_h, x_l], dim=1)
