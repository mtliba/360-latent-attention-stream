import torch
from torch import nn
from torch.autograd import Variable
from troch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import  ReLU

class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size, stride):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.pool(x, kernel_size= self.kernel_size, stride= self.stride)
        return x


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M_2':
            layers += [Downsample(kernel_size= 2, stride=2)]
        elif v == 'M_4':
            layers += [Downsample(kernel_size= 4, stride=4)]
        else:
            conv = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv, ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'global_attention': [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 'M_4', 512, 512, 512],
    'based_AM'        : [64, 64, 'M_2', 128, 128, 'M_2', 256, 256, 256, 'M_4', 512, 512, 512, 512, 512, 512]

      }
global_attention = make_conv_layers(cfg['global_attention'])
based_AM = make_conv_layers(cfg['based_AM'])
