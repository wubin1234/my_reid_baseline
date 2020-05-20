"""
@author : wubin
@connection : 799639771@qq.com
"""
import math
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inputs, outputs, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
