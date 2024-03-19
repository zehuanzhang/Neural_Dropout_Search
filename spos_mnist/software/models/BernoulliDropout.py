import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.quantized import QFunctional
import copy

class BernoulliDropout(nn.Module):
    def __init__(self, p=0.0):
        super(BernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,)) * p, requires_grad=False)
        if self.p < 1:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) / (1.0 - self.p), requires_grad=False)
        else:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) * 0.0, requires_grad=False)

        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        # print('BernoulliDropout_x:', x.shape, x.shape[:2])
        if self.p <= 0.0:
            return x
        mask_ = None
        if len(x.shape) <= 2:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1. - self.p)
            else:
                mask_ = torch.FloatTensor(x.shape).bernoulli_(1. - self.p)
        else:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)
            else:
                mask_ = torch.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)
        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            mask_ = torch.quantize_per_tensor(
                mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            mask_ = mask_.view(
                mask_.shape[0], mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.mul_mask.mul(x, mask_)
        x = self.mul_scalar.mul_scalar(x, self.multiplier.item())
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )


class MaskBernoulliDropout(nn.Module):
    def __init__(self, batch_size, channel, p=0.0):
        super(MaskBernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,)) * p, requires_grad=False)
        if self.p < 1:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) / (1.0 - self.p), requires_grad=False)
        else:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) * 0.0, requires_grad=False)

        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()

        # Generate mask during initialization
        # if self.p > 0:
        if torch.cuda.is_available():
            self.mask_ = torch.cuda.FloatTensor(torch.Size([batch_size, channel])).bernoulli_(1. - self.p)
        else:
            self.mask_ = torch.FloatTensor(torch.Size([batch_size, channel])).bernoulli_(1. - self.p)

    def forward(self, x):
        mask_ = copy.deepcopy(self.mask_)
        if self.p <= 0.0:
            return x

        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            mask_ = torch.quantize_per_tensor(
                self.mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            mask_ = mask_.view(
                mask_.shape[0], mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.mul_mask.mul(x, mask_)
        x = self.mul_scalar.mul_scalar(x, self.multiplier.item())
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )
