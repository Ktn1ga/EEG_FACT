"""
Model Reference:
-----------------------------------------------------
Model Name:         FACT
Repository URL:     https://github.com/Ktn1ga/EEG_FACT
Original Paper:
-----------------------------------------------------
"""
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import torch.fft
import torch.nn.functional as F
from module.util import *
import numpy as np
import torch
from torch import nn
import math

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        # index = list(range(0, seq_len // 2))
        # np.random.shuffle(index)
        # index = index[:modes]
        index = np.random.choice(seq_len // 2, modes, replace=False)
    elif mode_select_method =='segmented_random':
        # Divide the frequency range into `modes` segments
        segments = np.array_split(np.arange(seq_len // 2), modes)
        # Randomly select one frequency from each segment
        index = [np.random.choice(segment) for segment in segments]
    else:
        # index = list(range(0, modes))
        index = np.arange(modes)
    index.sort()
    return index

class FA(nn.Module):
    def __init__(self,in_depth = 1,in_channel=22,out_depth=1,out_channel =22):
        super(FA, self).__init__()
        # # 频域加权
        seq_len = 1000
        modes = 64
        self.radio = 1
        mode_select_method = 'random'
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        # Define weights for each frequency mode group
        self.fweights_size = [min(modes,math.ceil(len(self.index)/self.radio)) +1,1]
        self.fweights = nn.Parameter(torch.zeros(self.fweights_size),
                                     requires_grad=True)
        self.fweights_im = nn.Parameter(torch.zeros(self.fweights_size),
                                     requires_grad=True)
        self.dropout = nn.Dropout(p=0.3)

    # Complex multiplication
    def compl_mul1d(self, input,weights,i,flag = 'freq'):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        if flag == 'freq':
            rate = 0
            weight = weights[i].unsqueeze(-1).unsqueeze(-1)
            if i == 0:
                all_weight = weight
            elif i == 1:
                all_weight = weight + rate * weights[i+1].unsqueeze(-1).unsqueeze(-1)
            elif i == self.fweights_size[0]-1:
                all_weight = weight + rate * weights[i-1].unsqueeze(-1).unsqueeze(-1)
            else:
                all_weight = weight + rate * weights[i + 1].unsqueeze(-1).unsqueeze(-1)
            return input*all_weight

    def forward(self, x):
        x = x.squeeze(dim=1)
        # B 1 Channel Time
        B, E, L = x.shape
        # (batch, c, h, w/2+1, 2)
        fft_dim = -1
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted_re = ffted.real
        ffted_im = ffted.imag
        # Perform Fourier neural operations
        out_ft_re = torch.zeros(B, E, L // 2 + 1, device=x.device)
        out_ft_im = torch.zeros(B, E, L // 2 + 1, device=x.device)
        for wi, i in enumerate(self.index):
            if wi == 0:
                out_ft_re[:, :, wi] = self.compl_mul1d(ffted_re[:, :, i],self.fweights,0,flag='freq')
                out_ft_im[:, :, wi] = self.compl_mul1d(ffted_im[:, :, i],self.fweights_im, 0, flag='freq')
            else:
                out_ft_re[:, :, wi] = self.compl_mul1d(ffted_re[:, :, i],self.fweights,int(wi/self.radio)+1,flag='freq')
                out_ft_im[:, :, wi] = self.compl_mul1d(ffted_im[:, :, i],self.fweights_im, int(wi / self.radio) + 1,
                                                    flag='freq')
        self.fweights.data = torch.renorm(
            self.fweights.data, p=2, dim=0, maxnorm=1)

        self.fweights_im.data = torch.renorm(
            self.fweights.data, p=2, dim=0, maxnorm=1)

        x_ft = torch.complex(ffted_re+out_ft_re, ffted_im+out_ft_im)
        # Return to time domain
        x = torch.fft.irfftn(x_ft, s=L,dim=2,norm = 'ortho')
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        return x

class EEGDepthAware(nn.Module):
    def __init__(self, W, C, k=7):
        super(EEGDepthAware, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)
    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)
        return y * self.C * x

class FFT_Based_Refactor(nn.Module):
    def __init__(self, k=2):
        super(FFT_Based_Refactor, self).__init__()
        self.k = k
        self.top_list = None

    def forward(self, x):
        if self.top_list is None or 1:
            # [B, T, C]
            xf = torch.fft.rfft(x, dim=1)
            # find period by amplitudes
            frequency_list = abs(xf).mean(0).mean(-1)
            # 排除直流分量
            frequency_list[0] = 0
            value, top_list = torch.topk(frequency_list, self.k)
            top_list = top_list.detach().cpu().numpy()
            # top_list = np.array([2])
            n = top_list.shape[0]
            for i in range(n-1,0,-1):
                top_list[i]=top_list[i-1]
            top_list[0]=1
            self.top_list = top_list
            # print(self.top_list)
        else:
            xf = torch.fft.rfft(x, dim=1)
        # top_list = np.array([1])
        period = x.shape[1] // self.top_list
        # self.draw(frequency_list)
        # print(period)
        return period,abs(xf).mean(-1)[:, self.top_list]

class Multi_periodicity_Inception(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Multi_periodicity_Inception, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i, groups=in_channels))
        kernels.append(nn.AvgPool2d(kernel_size=(3, 3),padding=(1, 1)))

        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TPI(nn.Module):
    def __init__(self,F2,num_kernel):
        super(TPI, self).__init__()
        self.seq_len = 15
        self.k = 3
        self.fft_get_p = FFT_Based_Refactor(self.k)
        self.d_model = F2
        self.conv = nn.Sequential(
            Multi_periodicity_Inception(self.d_model, self.d_model,
                                        num_kernels=num_kernel),
            nn.GELU(),
            Multi_periodicity_Inception(self.d_model, self.d_model,
                                        num_kernels=num_kernel)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = self.fft_get_p(x)
        # print(period_list)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class FACT(nn.Module):
    def __init__(self,nChan=22,nTime=1000,nClass=4):
        super(FACT,self).__init__()
        F0 = 1
        F1 = 8
        D = 2
        F2 = 16
        self.use_fa = True

        if self.use_fa:
            self.fa = FA(in_depth=1,in_channel=nChan,
                          out_depth=F0,out_channel=nChan)

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F0,
                out_channels=F1,
                kernel_size=(1, 64),
                stride=1,
                padding='same',
                bias=False,
                groups=1
            ),
            nn.BatchNorm2d(num_features=F1)
        )

        self.channel_conv = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1*D,
                kernel_size=(nChan, 1),
                groups=F1,
                bias=False,
                max_norm=1.
            ),
            nn.BatchNorm2d(num_features=F1*D),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.Dropout(p=0.3)
        )
        self.depth_seqarable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1*D,
                out_channels=F1*D,
                kernel_size=(1, 16),
                stride=1,
                padding='same',
                groups=F1*D,
                bias=False
            ),
            nn.Conv2d(
                in_channels=F1*D,
                out_channels=F2,
                kernel_size=(1, 1),
                groups=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1,8),
                stride=(1,8)
            ),
            nn.Dropout(p=0.3)
        )
        self.depth_aware_conv = EEGDepthAware(W=15, C=F2, k=7)
        self.layer = 1
        num_kernel = 4
        self.layer_norm = nn.LayerNorm(F2)
        self.model = nn.ModuleList([TPI(F2, num_kernel)
                                    for _ in range(self.layer)])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=F2*15,
                out_features=nClass,
                max_norm=.25
            ),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)

        if self.use_fa:
            x = self.fa(x)

        x = self.temporal_conv(x)

        x = self.channel_conv(x)
        x = self.depth_seqarable_conv(x)
        x = self.depth_aware_conv(x)

        x = torch.squeeze(x, dim=2)  # NCW
        x = x.permute(0,2,1)
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))

        x = self.classifier(x)

        return x


def main():
    channels = 22
    samples = 1000
    input=torch.randn(32, channels, samples)
    output = torch.randn(32,1)
    model = FACT(nChan=channels,nTime=samples,nClass=4)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable Parameters in the network are: ' + str(count_parameters(model)))
    # summary(model=model, input_data=((input,output)), device="cpu")
    # stat(model, (1,channels, samples))


if __name__ == "__main__":
    main()