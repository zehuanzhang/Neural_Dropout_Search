import torch
import torch.nn.functional as F
from torch import nn

class SlidingWindowDropout2D(nn.Module):
    def __init__(self, drop_prob, window_size):
        super(SlidingWindowDropout2D, self).__init__()

        self.drop_prob = drop_prob
        self.window_size = window_size
        print('here')

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.drop_prob == 0.:  #if not self.training or self.drop_prob == 0.:
            return x
        else:
            # create a mask with ones
            mask = torch.ones(x.shape[0], *x.shape[2:], device=x.device)

            # iterate over the feature map with the sliding window
            for i in range(mask.shape[1] - self.window_size + 1):
                for j in range(mask.shape[2] - self.window_size + 1):
                    # in each window, randomly decide if the window should be dropped
                    if torch.rand(1).item() < self.drop_prob:
                        mask[:, i:i + self.window_size, j:j + self.window_size] = 0

            return x * mask[:, None, :, :]




class ChannelwiseSlidingWindowDropout2D(nn.Module):
    def __init__(self, drop_prob, window_size):
        super(ChannelwiseSlidingWindowDropout2D, self).__init__()

        self.drop_prob = drop_prob
        self.window_size = window_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.drop_prob == 0.:  #if not self.training or self.drop_prob == 0.:
            return x
        else:
            # create a mask with ones
            mask = torch.ones(x.shape, device=x.device)

            # iterate over the feature map with the sliding window
            for c in range(mask.shape[1]):
                for i in range(mask.shape[2] - self.window_size + 1):
                    for j in range(mask.shape[3] - self.window_size + 1):
                        # in each window, randomly decide if the window should be dropped
                        if torch.rand(1).item() < self.drop_prob:
                            mask[:, c, i:i + self.window_size, j:j + self.window_size] = 0

            return x * mask