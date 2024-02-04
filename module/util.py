import torch.nn as nn
from torch.autograd import Function
import torch
from visdom import Visdom
import numpy as np
import time

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)

class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)
            
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SuareModule(nn.Module):
    def forward(self, x):
        return torch.square(x)


class SafelogModule(nn.Module):
    def forward(self, x, eps=1e-6):
        return torch.log(torch.clamp(x, min=eps))

#%%
class EEGNet_util(nn.Module):
    def __init__(self, in_channel = 1,eeg_chans=22, dropoutRate=0.5, kerSize=64, kerSize_Tem=16, F1=8, D=2, poolDept=8, poolSeq=8, bias=False):
        super(EEGNet_util, self).__init__()
        F2 = F1*D

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=F1,
                kernel_size=(1, kerSize),
                stride=1,
                padding='same',
                bias=bias,
                groups=in_channel
            ),
            nn.BatchNorm2d(num_features=F1)
        )
        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1*D,
                kernel_size=(eeg_chans, 1),
                groups=F1,
                bias=bias,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,poolDept),
                stride=(1,poolDept)
            ),
            nn.Dropout(p=dropoutRate)
        )

        self.seqarableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1, kerSize_Tem),
                stride=1,
                padding='same',
                groups=F2,
                bias=bias
            ),
            nn.Conv2d(
                in_channels=F2,
                out_channels=F2,
                kernel_size=(1, 1),
                stride=1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,poolSeq),
                stride=(1,poolSeq)
            ),
            nn.Dropout(p=dropoutRate)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.depthwiseConv(x)
        x = self.seqarableConv(x)

        return x