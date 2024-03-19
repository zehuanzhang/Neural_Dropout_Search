import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.autograd.profiler as profiler

# from software.models.dropout import BernoulliDropout
from software.utils import Flatten, index_to_bool_list, i_to_mutuablelayer_mnist

import sys
sys.path.append('../')

import nni
from nni.nas.pytorch import mutables

from software.models.Masksembles import Masksembles2D, Masksembles1D
from software.models.DropBlock import DropBlock2D
from software.models.RandomDrop import RandomDrop
from software.models.BernoulliDropout import BernoulliDropout



class LeNet(nn.Module):
    def __init__(self, input_size, output_size, args):
        super(LeNet, self).__init__()
        self.args = args
        self.init_channels = input_size[1]

        self.q = args.q
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x, set_mask_dict=False, samples=1, profile=False):
        if not profile:
            return self._forward_no_profile(x, set_mask_dict=set_mask_dict)
        else:
            return self._forward_profile(x, samples)

    def _forward_no_profile(self, x, set_mask_dict=False):
        if self.q:
            x = self.quant(x)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nni.nas.pytorch.mutables.LayerChoice) and set_mask_dict != False:
                if i == 2 or i==5:
                    layerChoice_counter = i_to_mutuablelayer_mnist(i)
                    choice_index = set_mask_dict[f'choice_{layerChoice_counter}']
                    set_mask = index_to_bool_list(choice_index, len(layer))
                    assert isinstance(set_mask, list), 'set_mask must be list type'
                elif i==9:
                    layerChoice_counter = 3
                    choice_index = set_mask_dict[f'choice_{layerChoice_counter}']
                    set_mask = index_to_bool_list(choice_index, len(layer))
                    assert isinstance(set_mask, list), 'set_mask must be list type'
                else:
                    sys.exit()

                x = layer(set_mask, x)
            else:
                x = layer(x)
            #
            # x = layer(x)
        if self.q:
            x = self.dequant(x)
        return F.softmax(x, dim=-1)

    def _forward_profile(self, x, samples):
        static = -1
        cache = None
        with profiler.record_function("static_part"):
            if self.q:
                x = self.quant(x)

            for i, layer in enumerate(self.layers):
                if isinstance(layer, BernoulliDropout):
                    static = i
                    cache = x
                    break
                x = layer(x)
            if static == -1:
                if self.q:
                    x = self.dequant(x)
                return F.softmax(x, dim=-1)
        out = []
        with profiler.record_function("dynamic_part"):
            for sample in range(samples):
                x = cache
                for i in range(static, len(self.layers)):
                    x = self.layers[i](x)

                if self.q:
                    x = self.dequant(x)
                x = F.softmax(x, dim=-1)
                out.append(x.detach())
            # out = torch.stack(out, dim=1).mean(dim=1)
        return out


class LeNet_P(LeNet):
    def __init__(self, input_size, output_size, args):
        super(LeNet_P, self).__init__(input_size, output_size, args)

        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             Flatten(),
             nn.Linear(in_features=50 * 7 * 7, out_features=500, bias=False),
             nn.ReLU(),
             nn.Linear(in_features=500, out_features=output_size, bias=False)])

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['5', '6'], inplace=True)


class LeNet_LL(LeNet):
    def __init__(self, input_size, output_size, args):
        super(LeNet_LL, self).__init__(input_size, output_size, args)

        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(
                 in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             Flatten(),
             nn.Linear(in_features=50 * 7 * 7,
                       out_features=500, bias=False),
             nn.ReLU(),
             BernoulliDropout(self.args.p),
             nn.Linear(in_features=500, out_features=output_size, bias=False)])

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['5', '6'], inplace=True)


class LeNet_TWO_THIRD(LeNet):
    def __init__(self, input_size, output_size, args):
        super(LeNet_TWO_THIRD, self).__init__(input_size, output_size, args)

        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             nn.Conv2d(
                 in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             BernoulliDropout(self.args.p),
             Flatten(),
             nn.Linear(in_features=50 * 7 * 7,
                       out_features=500, bias=False),
             nn.ReLU(),
             BernoulliDropout(self.args.p),
             nn.Linear(in_features=500, out_features=output_size, bias=False)])

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['6', '7'], inplace=True)


# class LeNet_ALL(LeNet):
#     def __init__(self, input_size, output_size, args):
#         super(LeNet_ALL, self).__init__(input_size, output_size, args)
#
#         self.layers = nn.ModuleList([nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2,  bias=False),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),
#                                      # BernoulliDropout(self.args.p),
#                                      nn.Conv2d(
#                                          in_channels=20, out_channels=50, kernel_size=5, padding=2,  bias=False),
#                                      nn.MaxPool2d(kernel_size=2, stride=2),
#                                      # BernoulliDropout(self.args.p),
#                                      Flatten(),
#                                      nn.Linear(in_features=50*7*7,
#                                                out_features=500,  bias=False),
#                                      nn.ReLU(),
#                                      BernoulliDropout(self.args.p),
#                                      nn.Linear(in_features=500, out_features=output_size,  bias=False)])
#
#     def fuse_model(self):
#         torch.quantization.fuse_modules(self.layers, ['7', '8'], inplace=True)

class LeNet_ALL(LeNet):
    def __init__(self, input_size, output_size, args):
        super(LeNet_ALL, self).__init__(input_size, output_size, args)

        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             mutables.LayerChoice([
                 BernoulliDropout(self.args.p),
                 RandomDrop(self.args.p),
                 DropBlock2D(self.args.p, block_size=2),
                 Masksembles2D(channels=20, n=4, scale=1.2)
             ]),
             nn.Conv2d(
                 in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False),
             nn.MaxPool2d(kernel_size=2, stride=2),
             mutables.LayerChoice([
                 BernoulliDropout(self.args.p),
                 RandomDrop(self.args.p),
                 DropBlock2D(self.args.p, block_size=2),
                 Masksembles2D(channels=50, n=16, scale=1.2)
             ]),
             Flatten(),
             nn.Linear(in_features=50 * 7 * 7,
                       out_features=500, bias=False),
             nn.ReLU(),
             mutables.LayerChoice([
                 BernoulliDropout(self.args.p),
                 # RandomDrop(self.args.p),
                 # DropBlock2D(self.args.p, block_size=2),
                 Masksembles1D(channels=500, n=16, scale=1.2)
             ]),
             nn.Linear(in_features=500, out_features=output_size, bias=False)])

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, ['7', '8'], inplace=True)
