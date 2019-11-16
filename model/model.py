import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math

# Model related functions

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

def no_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Super Resolution
class RCAN(nn.Module):
    def __init__(self,scale = 2,res_blocks = 10, rcab_blocks= 20, channels=64 ):
        super(RCAN, self).__init__()
        self.RBs = res_blocks
        self.RCABs = rcab_blocks
        self.scale = scale
        self.channels =channels
        # Define Network
        # ===========================================
        upsample_block_num = int(math.log(self.scale, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(3, self.channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        block2= [RG(self.channels,self.RCABs) for _ in range(self.RBs)]
        block2.append(nn.Conv2d(self.channels,self.channels,3,1,1))
        self.block2 = nn.Sequential(*block2)
        block3 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block3.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.block3 = nn.Sequential(*block3)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2+block1)
        return block3

class RG(nn.Module):
    def __init__(self,in_channels = 64,RCAB_blocks = 20):
        super(RG,self).__init__()
        RCABs = [RCAB(in_channels) for _ in range(RCAB_blocks)]
        RCABs.append(nn.Conv2d(in_channels,in_channels,3,padding=1))
        self.RCABs = nn.Sequential(*RCABs)

    def forward(self, x):
        out = self.RCABs(x)
        return out + x

class RCAB(nn.Module):

    def __init__(self,in_channels =64):
        super(RCAB,self).__init__()
        self.in_channels= in_channels
        self.filter_size = in_channels
        self.conv1 = nn.Conv2d(self.in_channels,self.filter_size,3,1,1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.filter_size,self.filter_size,3,1,1)
        self.ca1 = CA(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out =self.conv2(out)
        out = self.ca1(out )
        return x+out


class CA(nn.Module):
    def __init__(self,in_channels = 64,reduction = 8):
        super(CA,self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_bottle = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pooling(x)
        out = self.conv_bottle(x)
        return x*out.expand_as(x)

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_suffle = nn.PixelShuffle(up_scale)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_suffle(x)
        x = self.relu(x)
        return x



class EDSR(nn.Module):
    def __init__(self, filter_channels = 64, res_scale = 1, res_blocks = 16, upscale = 2):
        super(EDSR, self).__init__()
        self.filter_channel = filter_channels
        self.res_scale = res_scale
        self.res_blocks = res_blocks
        self.upscale = upscale

        # Network Blocks
        # ===========================================
        upsample_block_num = int(math.log(self.upscale, 2))
        self.block1 = nn.Conv2d(3, self.filter_channel, kernel_size=3, padding=1)

        block2 = [ResidualBlock(self.filter_channel,self.res_scale) for _ in range(self.res_blocks)]
        self.block2 = nn.Sequential(*block2)

        block3 = [UpsampleBlock(self.filter_channel, 2) for _ in range(upsample_block_num)]
        block3.append(nn.Conv2d(self.filter_channel, 3, kernel_size=3, padding=1))
        self.block3 = nn.Sequential(*block3)



    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block1+block2)
        return block3


class ResidualBlock(nn.Module):

    def __init__(self, channels, scale_factor = 1):
        super(ResidualBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual.mul(self.scale_factor)