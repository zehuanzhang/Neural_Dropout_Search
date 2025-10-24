import torch
import torch.nn as nn


class RandomDrop(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(RandomDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # shape = [bchw]
        assert x.dim() == 4, \
            "Expected input with 4 dimensions bchw"

        if x.is_cuda:
            mask = (torch.rand(*x.shape) > self.drop_prob).float().to('cuda:0')
        else:
            mask = (torch.rand(*x.shape) > self.drop_prob).float()

        if self.drop_prob == 0:
            return x
        else:
            return x * mask


class MaskRandomDrop(nn.Module):
    def __init__(self, batch_size, channel, fm_size, drop_prob=0.0):
        super(MaskRandomDrop, self).__init__()
        self.drop_prob = drop_prob
        if torch.cuda.is_available():
            self.mask = (torch.rand(batch_size, channel, fm_size, fm_size) > self.drop_prob).float().to('cuda:0')
        else:
            self.mask = (torch.rand(batch_size, channel, fm_size, fm_size) > self.drop_prob).float()

    def forward(self, x):
        # shape = [bchw]
        assert x.dim() == 4, \
            "Expected input with 4 dimensions bchw"

        if self.drop_prob == 0:
            return x
        else:
            return x * self.mask
