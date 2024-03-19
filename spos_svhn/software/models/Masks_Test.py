import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.quantized import QFunctional
import torch.nn.functional as F

import common

# from dropout import BernoulliDropout
from BernoulliDropout import BernoulliDropout, MaskBernoulliDropout
from Masksembles import Masksembles2D, Masksembles1D
from DropBlock import DropBlock2D  # , DropBlock3D
from SlidingWindow import SlidingWindowDropout2D, ChannelwiseSlidingWindowDropout2D
from RandomDrop import RandomDrop, MaskRandomDrop



def test_RandomDropout():
    # Create a random tensor
    # N, C, H, W = 10, 3, 32, 32
    input_tensor = torch.ones(3, 3, 5, 5)
    print('input_tensor:', input_tensor[0])

    # Set up a DropBlock2D instance
    drop_prob = 0.1
    dropout = RandomDrop(drop_prob=drop_prob)
    # dropout = MaskRandomDrop(batch_size=3, channel=3, fm_size=5, drop_prob=drop_prob)

    # Pass the tensor through the DropBlock2D module
    output_tensor = dropout(input_tensor)
    print('output_tensor:', output_tensor[0])
    print('mask:', output_tensor[0] != 0)

    # Check if the output shape is the same as the input shape
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape doesn't match input tensor shape."

    # Check if some elements in the output tensor are zero (indicating that some blocks have been dropped)
    assert (output_tensor == 0).any(), "No blocks were dropped in the output tensor."

    print("Dropout tests passed.")
def test_bernoulli_dropout():
    dropout = BernoulliDropout(p=0.5)  # create instance of BernoulliDropout with p=0.5
    # dropout = Masksembles1D(channels=12, n=4, scale=1.5)

    # Create a 2D tensor representing a batch of linear data
    # Dimensions: [batch size, channels]
    x = torch.ones(4, 12)
    print(x)

    # Pass the tensor through the dropout layer
    output = dropout(x)
    print(output)

    # Count the number of non-zero elements in the output tensor
    num_nonzero = torch.count_nonzero(output)

    print(f"Number of Non-Zero elements in Output: {num_nonzero.item()}")

    # To verify the behavior of the dropout layer, we can check the number of non-zero elements in the output.
    # It should be approximately half of the total elements in the input tensor, since we set p=0.5.
    # However, since dropout is a stochastic process, it won't be exactly half every time.
def test_bernoulli_dropout_images():
    # Create a 4D tensor representing a batch of RGB images
    # Dimensions: [batch size, channels, height, width]
    x = torch.ones(4, 12, 4, 4)

    # Create an instance of BernoulliDropout with p=0.5
    dropout = BernoulliDropout(p=0.5)
    # dropout = Masksembles2D(channels=12, n=4, scale=2.0)

    # Pass the tensor through the dropout layer
    output = dropout(x)

    # We expect that approximately half of the channels in the output tensor are zeroed out, on average.
    # Since dropout is channel-wise, we can check this by counting the number of channels that have been zeroed out.
    # A channel is considered zeroed out if the sum of its elements is 0.
    num_zeroed_out_channels = (output.sum(dim=[2, 3]) == 0).sum()

    print(f"Number of Zeroed Out Channel  s in Output: {num_zeroed_out_channels.item()}")
def test_DropBlock2D():
    # Create a random tensor
    # N, C, H, W = 10, 3, 32, 32
    input_tensor = torch.ones(3, 3, 5, 5)
    print('input_tensor:', input_tensor[0])

    # Set up a DropBlock2D instance
    drop_prob = 0.3
    block_size = 2
    drop_block_2d = DropBlock2D(drop_prob=drop_prob, block_size=block_size)
    # drop_block_2d.train()

    # Pass the tensor through the DropBlock2D module
    output_tensor = drop_block_2d(input_tensor)
    print('output_tensor:', output_tensor[0])
    print(output_tensor.shape, input_tensor.shape)

    # Check if the output shape is the same as the input shape
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape doesn't match input tensor shape."

    # Check if some elements in the output tensor are zero (indicating that some blocks have been dropped)
    assert (output_tensor == 0).any(), "No blocks were dropped in the output tensor."

    print("DropBlock2D tests passed.")
def test_Mask_DropBlock2D():
    # Create a random tensor
    # N, C, H, W = 10, 3, 32, 32
    input_tensor = torch.ones(3, 3, 5, 5)
    print('input_tensor:', input_tensor[0])

    # Set up a DropBlock2D instance
    drop_prob = 0.3
    block_size = 2
    drop_block_2d = DropBlock2D(batch_size=3, channel=3, fm_size=5, drop_prob=drop_prob, block_size=block_size)
    # drop_block_2d.train()

    # Pass the tensor through the DropBlock2D module
    output_tensor = drop_block_2d(input_tensor)
    print('output_tensor:', output_tensor[0])

    # Check if the output shape is the same as the input shape
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape doesn't match input tensor shape."

    # Check if some elements in the output tensor are zero (indicating that some blocks have been dropped)
    assert (output_tensor == 0).any(), "No blocks were dropped in the output tensor."

    print("DropBlock2D tests passed.")
def test_DropBlock3D():
    # Set up a DropBlock3D instance
    drop_prob = 0.1
    block_size = 3
    drop_block_3d = DropBlock3D(drop_prob=drop_prob, block_size=block_size)
    # drop_block_3d.train()

    # Create a random tensor
    # N, C, D, H, W = 10, 3, 8, 32, 32
    input_tensor = torch.ones(10, 3, 8, 32, 32)
    # print('input_tensor:', input_tensor[0])

    # Pass the tensor through the DropBlock3D module
    output_tensor = drop_block_3d(input_tensor)

    # Check if the output shape is the same as the input shape
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape doesn't match input tensor shape."

    # Check if some elements in the output tensor are zero (indicating that some blocks have been dropped)
    assert (output_tensor == 0).any(), "No blocks were dropped in the output tensor."

    print("DropBlock3D tests passed.")
def test_Masksembles2D():
    # Create a 4D tensor representing a batch of RGB images
    # Dimensions: [batch size, channels, height, width]
    x = torch.ones(4, 12, 4, 4)
    # x = torch.ones([4, 16, 28, 28])

    # Create an instance of BernoulliDropout with p=0.5
    # dropout = BernoulliDropout(p=0.5, x=x)
    dropout = Masksembles2D(channels=12, n=2, scale=1.05)

    # Pass the tensor through the dropout layer
    output = dropout(x)
    print(output[0][0])

    # We expect that approximately half of the channels in the output tensor are zeroed out, on average.
    # Since dropout is channel-wise, we can check this by counting the number of channels that have been zeroed out.
    # A channel is considered zeroed out if the sum of its elements is 0.
    num_zeroed_out_channels = (output.sum(dim=[2, 3]) == 0).sum()

    print(f"Number of Zeroed Out Channel  s in Output: {num_zeroed_out_channels.item()}")

def test_SlidingWindowDropout2D():
    # Set up a DropBlock2D instance
    drop_prob = 0.2
    window_size = 2
    slidingwindow_dropout_2d = SlidingWindowDropout2D(drop_prob=drop_prob, window_size=window_size)
    # slidingwindow_dropout_2d.train()

    # Create a sample input tensor
    # Create a random tensor
    # N, C, H, W = 10, 3, 32, 32
    input_tensor = torch.ones((3, 3, 5, 5))
    print('input_tensor:', input_tensor[0])

    # Pass the input tensor through the dropout module
    output_tensor = slidingwindow_dropout_2d(input_tensor)
    print('output_tensor:', output_tensor[0])

    # Verify the output
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape doesn't match the input tensor shape."

    print("DropBlock2D tests passed.")

#######################################################################
# test_bernoulli_dropout()
# test_bernoulli_dropout_images()
# test_DropBlock2D()
# test_DropBlock3D()
# test_SlidingWindowDropout2D()
# test_RandomDropout()
test_Masksembles2D()


# input = torch.tensor([
#     [[1,0,0,0],
# [1,0,0,0],
# [1,0,0,0],
# [1,0,0,0]]
# ]).float()
#
#
# print(input)
#
#
# block_size = 2
# output = F.max_pool2d(input = input,
#                       kernel_size = (block_size,block_size),
#                       stride=(1,1),
#                       padding = 1)
#
# print(output)
