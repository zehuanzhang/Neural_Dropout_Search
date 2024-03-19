import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.drop_prob == 0.: #if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma()

            # sample mask
            mask = (torch.rand(x.shape[1], *x.shape[2:]) < gamma).float()

            # compute block mask
            block_mask = self._compute_block_mask(mask).to(x.device)

            # apply block mask
            out = x * block_mask#[None, :, :, :] #x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[None, :, :, :],
                                  #input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self):
        return self.drop_prob / (self.block_size ** 2)

class MaskDropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, batch_size, channel, fm_size, drop_prob, block_size):
        super(MaskDropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

        # get gamma value
        self.gamma = self._compute_gamma()

        # sample mask
        # self.mask = (torch.rand(x.shape[0], *x.shape[2:]) < self.gamma).float()
        self.mask = (torch.rand(channel, fm_size, fm_size) < self.gamma).float()


        # compute block mask
        self.block_mask = self._compute_block_mask(self.mask)


    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.drop_prob == 0.: #if not self.training or self.drop_prob == 0.:
            return x
        else:
            # place mask on input device
            block_mask = self.block_mask.to(x.device)

            # apply block mask
            out = x * block_mask[None, :, :, :] #x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self):
        return self.drop_prob / (self.block_size ** 2)


# class DropBlock3D(DropBlock2D):
#     r"""Randomly zeroes 3D spatial blocks of the input tensor.
#     An extension to the concept described in the paper
#     `DropBlock: A regularization method for convolutional networks`_ ,
#     dropping whole blocks of feature map allows to remove semantic
#     information as compared to regular dropout.
#     Args:
#         drop_prob (float): probability of an element to be dropped.
#         block_size (int): size of the block to drop
#     Shape:
#         - Input: `(N, C, D, H, W)`
#         - Output: `(N, C, D, H, W)`
#     .. _DropBlock: A regularization method for convolutional networks:
#        https://arxiv.org/abs/1810.12890
#     """
#
#     def __init__(self, drop_prob, block_size):
#         super(DropBlock3D, self).__init__(drop_prob, block_size)
#
#     def forward(self, x):
#         # shape: (bsize, channels, depth, height, width)
#
#         assert x.dim() == 5, \
#             "Expected input with 5 dimensions (bsize, channels, depth, height, width)"
#
#         if self.drop_prob == 0.: #if not self.training or self.drop_prob == 0.:
#             return x
#         else:
#             # get gamma value
#             gamma = self._compute_gamma(x)
#
#             # sample mask
#             mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
#
#             # place mask on input device
#             mask = mask.to(x.device)
#
#             # compute block mask
#             block_mask = self._compute_block_mask(mask)
#
#             # apply block mask
#             out = x * block_mask[:, None, :, :, :]
#
#             # scale output
#             out = out * block_mask.numel() / block_mask.sum()
#
#             return out
#
#     def _compute_block_mask(self, mask):
#         block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
#                                   kernel_size=(self.block_size, self.block_size, self.block_size),
#                                   stride=(1, 1, 1),
#                                   padding=self.block_size // 2)
#
#         if self.block_size % 2 == 0:
#             block_mask = block_mask[:, :, :-1, :-1, :-1]
#
#         block_mask = 1 - block_mask.squeeze(1)
#
#         return block_mask
#
#     def _compute_gamma(self, x):
#         return self.drop_prob / (self.block_size ** 3)
