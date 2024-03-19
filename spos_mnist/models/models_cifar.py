import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import torch
import torch.autograd.profiler as profiler

import sys
sys.path.append('../')

import nni
from nni.nas.pytorch import mutables

from software.models.Masksembles import Masksembles2D, Masksembles1D
from software.models.DropBlock import DropBlock2D
from software.models.RandomDrop import RandomDrop
from software.models.BernoulliDropout import BernoulliDropout

# from software.models.dropout import BernoulliDropout
from software.utils import Flatten, Add, index_to_bool_list, ij_to_mutuablelayer


# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride, args, dropout=False):
#         super(BasicBlock, self).__init__()
#         self.args = args
#         self.dropout = dropout
#         self.stem = nn.ModuleList([])
#         self.stem.append(nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
#         self.stem.append(nn.BatchNorm2d(planes))
#         self.stem.append(nn.ReLU())
#         if dropout:
#             self.stem.append(BernoulliDropout(self.args.p))
#         self.stem.append(nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False))
#         self.stem.append(nn.BatchNorm2d(planes))
#         if dropout:
#             self.stem.append(BernoulliDropout(self.args.p))
#
#         self.shortcut = nn.ModuleList([])
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut.append(nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False))
#             self.shortcut.append(nn.BatchNorm2d(self.expansion*planes))
#             if dropout:
#                 self.shortcut.append(BernoulliDropout(self.args.p))
#
#         self.add = Add()
#         self.end = nn.ReLU()
#
#
#     def fuse_model(self):
#         if not self.dropout:
#             torch.quantization.fuse_modules(self.stem, [['0', '1','2'], ['3', '4']], inplace=True)
#             if len(self.shortcut)>=1:
#                 torch.quantization.fuse_modules(self.shortcut, ['0', '1'], inplace=True)
#         else:
#             torch.quantization.fuse_modules(
#                 self.stem, [['0', '1', '2'], ['4', '5']], inplace=True)
#             if len(self.shortcut)>= 1:
#                 torch.quantization.fuse_modules(
#                     self.shortcut, ['0', '1'], inplace=True)
#
#     def forward(self, x):
#         out = x
#         for layer in self.stem:
#             out = layer(out)
#         shortcut = x
#         for layer in self.shortcut:
#             shortcut = layer(shortcut)
#         out = self.add(out, shortcut)
#         out = self.end(out)
#         return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, args, dropout=False):
        super(BasicBlock, self).__init__()
        self.args = args
        self.dropout = dropout
        self.stem = nn.ModuleList([])
        self.stem.append(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.stem.append(nn.BatchNorm2d(planes))
        self.stem.append(nn.ReLU())
        if dropout:
            self.stem.append(BernoulliDropout(self.args.p))

        self.stem.append(nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        self.stem.append(nn.BatchNorm2d(planes))
        if dropout:
            self.stem.append(BernoulliDropout(self.args.p))

        self.shortcut = nn.ModuleList([])
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.append(nn.Conv2d(in_planes, self.expansion * planes,
                                           kernel_size=1, stride=stride, bias=False))
            self.shortcut.append(nn.BatchNorm2d(self.expansion * planes))
            if dropout:
                self.shortcut.append(BernoulliDropout(self.args.p))

        self.add = Add()
        self.end = nn.ReLU()

    def fuse_model(self):
        if not self.dropout:
            torch.quantization.fuse_modules(self.stem, [['0', '1', '2'], ['3', '4']], inplace=True)
            if len(self.shortcut) >= 1:
                torch.quantization.fuse_modules(self.shortcut, ['0', '1'], inplace=True)
        else:
            torch.quantization.fuse_modules(
                self.stem, [['0', '1', '2'], ['4', '5']], inplace=True)
            if len(self.shortcut) >= 1:
                torch.quantization.fuse_modules(
                    self.shortcut, ['0', '1'], inplace=True)

    def forward(self, x):
        out = x
        for layer in self.stem:
            out = layer(out)
        shortcut = x
        for layer in self.shortcut:
            shortcut = layer(shortcut)
        out = self.add(out, shortcut)
        out = self.end(out)
        return out


class BasicBlockMutables(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, args, dropout=False):
        super(BasicBlockMutables, self).__init__()
        self.args = args
        self.dropout = dropout
        self.stem = nn.ModuleList([])
        self.stem.append(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.stem.append(nn.BatchNorm2d(planes))
        self.stem.append(nn.ReLU())
        if dropout:
            # self.stem.append(BernoulliDropout(self.args.p))
            self.stem.append(
                mutables.LayerChoice([
                    BernoulliDropout(self.args.p),
                    RandomDrop(self.args.p),
                    DropBlock2D(self.args.p, block_size=2),
                    Masksembles2D(channels=planes, n=4, scale=1.1)
                ])
            )

        self.stem.append(nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        self.stem.append(nn.BatchNorm2d(planes))
        if dropout:
            self.stem.append(BernoulliDropout(self.args.p))
            # self.stem.append(
            #     mutables.LayerChoice([
            #         BernoulliDropout(self.args.p),
            #         RandomDrop(self.args.p),
            #         DropBlock2D(self.args.p, block_size=2),
            #         Masksembles2D(channels=planes, n=4, scale=1.1)
            #     ])
            # )

        self.shortcut = nn.ModuleList([])
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.append(nn.Conv2d(in_planes, self.expansion * planes,
                                           kernel_size=1, stride=stride, bias=False))
            self.shortcut.append(nn.BatchNorm2d(self.expansion * planes))
            if dropout:
                self.shortcut.append(BernoulliDropout(self.args.p))

        self.add = Add()
        self.end = nn.ReLU()

    def fuse_model(self):
        if not self.dropout:
            torch.quantization.fuse_modules(self.stem, [['0', '1', '2'], ['3', '4']], inplace=True)
            if len(self.shortcut) >= 1:
                torch.quantization.fuse_modules(self.shortcut, ['0', '1'], inplace=True)
        else:
            torch.quantization.fuse_modules(
                self.stem, [['0', '1', '2'], ['4', '5']], inplace=True)
            if len(self.shortcut) >= 1:
                torch.quantization.fuse_modules(
                    self.shortcut, ['0', '1'], inplace=True)

    def forward(self, x, layerChoice_counter,  set_mask_dict=False):
        # print('Basic_Mutable')
        out = x
        for layer in self.stem:
            # print(type(layer))
            if isinstance(layer, nni.nas.pytorch.mutables.LayerChoice) and set_mask_dict != False:
                # print(f'choice_{layerChoice_counter}')
                choice_index = set_mask_dict[f'choice_{layerChoice_counter}']
                set_mask = index_to_bool_list(choice_index, len(layer))

                assert isinstance(set_mask, list), 'set_mask must be list type'
                # print('You are using the set_mask')
                out = layer(set_mask, out)
            else:
                out = layer(out)

        shortcut = x
        for layer in self.shortcut:
            shortcut = layer(shortcut)

        out = self.add(out, shortcut)
        out = self.end(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size, output_size, args, dropout=[False, False, False, False]):
        super(ResNet, self).__init__()
        self.args = args
        self.in_planes = 24
        self.init_channels = input_size[1]

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv2d(self.init_channels, self.in_planes, kernel_size=3,
                                     stride=1, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(self.in_planes))
        self.layers.append(nn.ReLU())
        if dropout[0]:
            self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(self._make_layer(self.in_planes, 2, stride=1, dropout=dropout[0]))
        self.layers.append(self._make_layer_mutables(48, 2, stride=2, dropout=dropout[1]))
        self.layers.append(self._make_layer_mutables(96, 2, stride=2, dropout=dropout[2]))
        self.layers.append(self._make_layer(192, 2, stride=2, dropout=dropout[3]))
        self.layers.append(nn.AvgPool2d(4))
        self.layers.append(Flatten())
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, 192 * BasicBlock.expansion, bias=False))
        self.layers.append(nn.ReLU())

        self.q = args.q
        if self.q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def _make_layer(self, planes, num_blocks, stride, dropout=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, self.args, dropout))
            self.in_planes = planes * BasicBlock.expansion
        return nn.ModuleList(layers)

    def _make_layer_mutables(self, planes, num_blocks, stride, dropout=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlockMutables(self.in_planes, planes, stride, self.args, dropout))
            self.in_planes = planes * BasicBlockMutables.expansion
        return nn.ModuleList(layers)

    # def forward(self, x, samples=1, profile=False):
    def forward(self, x, set_mask_dict=False, samples=1, profile=False):
        # print('forward')
        if not profile:
            # return self._forward_no_profile(x)
            return self._forward_no_profile(x, set_mask_dict=set_mask_dict)
        else:
            return self._forward_profile(x, samples)

    # def _forward_no_profile(self, x):
    #     if self.q:
    #         x = self.quant(x)
    #
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, nn.ModuleList):
    #             for sub_layer in layer:
    #                 x = sub_layer(x)
    #         else:
    #             x = layer(x)
    #     if self.q:
    #         x = self.dequant(x)
    #     return F.softmax(x, dim=-1)

    # def _forward_no_profile(self, x):
    def _forward_no_profile(self, x, set_mask_dict=False):
        # print('_forward_no_profile')
        if self.q:
            x = self.quant(x)

        for i, layer in enumerate(self.layers):
            # print(i,':',type(layer))
            if isinstance(layer, nn.ModuleList):
                # for sub_layer in layer:
                for j, sub_layer in enumerate(layer):
                    if isinstance(sub_layer, BasicBlock):
                        # print('**********---------***********sub_layer is BasicBlock************---------******************')
                        x = sub_layer(x)
                    else:
                        assert isinstance(sub_layer, BasicBlockMutables), 'sub_layer is not BasicBlockMutables'
                        # print('*********************sub_layer is BasicBlockMutables******************************')
                        layerChoice_counter = ij_to_mutuablelayer(i,j)
                        x = sub_layer(x, layerChoice_counter, set_mask_dict=set_mask_dict)
            else:
                x = layer(x)
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
                if isinstance(layer, nn.ModuleList):
                    for sub_layer in layer:
                        if sub_layer.dropout:
                            static = i
                            cache = x
                            break
                        x = sub_layer(x)
                    if static != -1:
                        break
                else:
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
                for i, layer in enumerate(self.layers[static:]):
                    if isinstance(layer, nn.ModuleList):
                        for sub_layer in layer:
                            x = sub_layer(x)
                    else:
                        x = layer(x)

                if self.q:
                    x = self.dequant(x)
                x = F.softmax(x, dim=-1)
                out.append(x.detach())
            # out = torch.stack(out, dim=1).mean(dim=1)
            return out

    def qat_hook(self, epoch):
        if self.q:
            if epoch > 5:
                # Freeze quantizer parameters
                self.apply(torch.quantization.disable_observer)
            if epoch > 4:
                # Freeze batch norm mean and variance estimates
                self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.layers, [['0', '1', '2'], ['9', '10']], inplace=True)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.fuse_model()


class ResNet_P(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_P, self).__init__(input_size, output_size, args, [False, False, False, False])
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))


class ResNet_LL(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_LL, self).__init__(input_size,
                                        output_size, args, [False, False, False, False])
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))


class ResNet_ONE_THIRD(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_ONE_THIRD, self).__init__(input_size,
                                               output_size, args, [False, False, False, True])
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))


class ResNet_HALF(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_HALF, self).__init__(input_size,
                                          output_size, args, [False, False, True, True])
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))


class ResNet_TWO_THIRD(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_TWO_THIRD, self).__init__(input_size,
                                               output_size, args, [False, True, True, True])
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))


class ResNet_ALL(ResNet):
    def __init__(self, input_size, output_size, args):
        super(ResNet_ALL, self).__init__(input_size,
                                         output_size, args, [True, True, True, True])
        self.layers.append(BernoulliDropout(self.args.p))
        self.layers.append(
            nn.Linear(192 * BasicBlock.expansion, output_size, bias=False))

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self.layers, [['0', '1', '2'], ['10', '11']], inplace=True)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.fuse_model()
