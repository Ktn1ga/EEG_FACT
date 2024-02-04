"""
Model Reference:
-----------------------------------------------------
Model Name:         EEGNet
Repository URL:     https://github.com/vlawhern/arl-eegmodels
Original Paper:     Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ.
                    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
                    J Neural Eng. 2018 Oct;15(5):056013. doi: 10.1088/1741-2552/aace8c. Epub 2018 Jun 22. PMID: 29932424.
                    https://pubmed.ncbi.nlm.nih.gov/29932424/
-----------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout))

        self.lin = nn.Linear(self.F2 * self.feature_dim, num_classes, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            x = torch.unsqueeze(x, 1)
        # x = x.unsqueeze(dim=1)

        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)

        x = self.lin(x)
        x = nn.Softmax(dim=1)(x)
        return x


###============================ Initialization parameters ============================###
channels = 22
samples = 1000
from torchinfo import summary
from torchstat import stat
###============================ main function ============================###
# import sys  # 导入sys模块
# sys.setrecursionlimit(3000)  # 将默认的递归深度修改为300
###============================ main function ============================###
def main():
    input = torch.randn(32, channels, samples)
    model = EEGNet(chunk_size=1000,
                   num_electrodes=22,
                   dropout=0.5,
                   kernel_1=64,
                   kernel_2=16,
                   F1=8,
                   F2=16,
                   D=2,
                   num_classes=4)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable Parameters in the network are: ' + str(count_parameters(model)))
    summary(model=model, input_size=(1,1,channels,samples), device="cpu")

if __name__ == "__main__":
    main()