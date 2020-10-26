import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

DEFAULT_USE_BIAS = True
DEFAULT_USE_INSTANCE_NORM = True


def conv1x1x1(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding.
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        dilation=dilation,
        stride=stride,
        padding=0,
        bias=DEFAULT_USE_BIAS)

def conv1x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 1x3x3 convolution with padding in the 2nd and 3rd dimensions.
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=[1, 3, 3],
        dilation=dilation,
        stride=stride,
        padding=[0, dilation, dilation],
        bias=DEFAULT_USE_BIAS)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding.
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=DEFAULT_USE_BIAS)


def tranpose_conv1x3x3(in_planes, out_planes, stride=2, dilation=1):
    return nn.ConvTranspose3d(in_planes,
                              out_planes,
                              kernel_size=[1, 2, 2],
                              stride=[1, stride, stride],
                              padding=0,
                              dilation=dilation,
                              bias=False)


class UNetBlock(nn.Module):
    def __init__(self, in_planes,
                 planes, stride=1,
                 dilation=1,
                 first_conv=True,
                 scale_change=0):
        """

        scale_change = 0 means no op
        scale_change = 1 means downsample
        scale_change = 2 means upsample
        """
        super(UNetBlock, self).__init__()

        if first_conv:
            self.conv1 = conv3x3x3(in_planes, planes//2, stride=stride,
                                   dilation=dilation)
            self.conv2 = conv1x3x3(planes//2, planes, dilation=dilation)
            self.conv3 = conv1x3x3(planes, planes, dilation=dilation)
        else:
            self.conv1 = conv1x3x3(in_planes, planes, stride=stride,
                                   dilation=dilation)
            self.conv2 = conv1x3x3(planes, planes, dilation=dilation)

        self.normalizer = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.last_op = None
        if scale_change == 1:
            self.last_op = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])
        elif scale_change == 2:
            self.last_op = tranpose_conv1x3x3(planes, planes//2)

        self.stride = stride
        self.dilation = dilation
        self.first_conv = first_conv
        self.scale_change = scale_change

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.first_conv:
            out = self.relu(out)
            out = self.conv3(out)
        if DEFAULT_USE_INSTANCE_NORM:
            out = self.normalizer(out)
        out = self.relu(out)

        if self.scale_change:
            block_out = out
            out = self.last_op(out)

        if self.scale_change == 1:
            return out, block_out
        else:
            return out


class UNet(nn.Module):

    def __init__(self, planes=32, n_classes=7):

        super(UNet, self).__init__()
        self.initial_conv = conv3x3x3(in_planes=1, out_planes=planes)
        self.block1 = UNetBlock(in_planes=planes,  # 32
                                planes=planes*2,  # 64
                                first_conv=True,
                                scale_change=1)

        self.block2 = UNetBlock(in_planes=planes * 2,  # 64
                                planes=planes * 4,  # 128
                                first_conv=True, scale_change=1)

        self.block3 = UNetBlock(in_planes=planes * 4,  # 128
                                planes=planes * 8,  # 256
                                first_conv=False,
                                scale_change=1)

        self.bottom_block = UNetBlock(in_planes=planes*8,  # 256
                                      planes=planes*16,  # 512
                                      first_conv=False,
                                      scale_change=0)

        self.bottom_transpose_conv = tranpose_conv1x3x3(
            in_planes=planes * 16,  # 512
            out_planes=planes * 8  # 256
        )

        self.block4 = UNetBlock(in_planes=planes * 16,  # 512
                                planes=planes * 8,  # 256
                                first_conv=False,
                                scale_change=2)

        self.block5 = UNetBlock(in_planes=planes * 8,  # 256
                                planes=planes * 4,  # 128
                                first_conv=True,
                                scale_change=2)

        self.block6 = UNetBlock(in_planes=planes * 4,  # 128
                                planes=planes * 2,  # 64
                                first_conv=True,
                                scale_change=0)

        self.logits = conv1x1x1(64, n_classes)

    def forward(self, x):
        out = self.initial_conv(x)
        out, block1_out = self.block1(out)
        out, block2_out = self.block2(out)
        out, block3_out = self.block3(out)
        out = self.bottom_block(out)
        out = self.bottom_transpose_conv(out)
        out = torch.cat([out, block3_out], dim=1)
        out = self.block4(out)
        out = torch.cat([out, block2_out], dim=1)
        out = self.block5(out)
        out = torch.cat([out, block1_out], dim=1)
        features = self.block6(out)
        logits = self.logits(features)

        return features, logits


class Hybrid2D3D(nn.Module):
    def __init__(self, planes=32, n_classes=7):
        super(Hybrid2D3D, self).__init__()
        self.planes = planes
        self.UNet = UNet(planes, n_classes)

        self.softmax = nn.Softmax(dim=1)

        self.first_branch = nn.ModuleList(
            [conv3x3x3(planes*2+n_classes, planes*2)]
            + [conv3x3x3(planes*2, planes*2)] * 2)

        self.second_branch = nn.ModuleList(
            [conv3x3x3(planes*2+n_classes, planes*4)] +
            [conv3x3x3(planes*4, planes*4)] * 6 +
            [conv1x1x1(planes*4, planes*2)])

        self.transpose_conv = tranpose_conv1x3x3(
            in_planes=planes*2,
            out_planes=planes*2
        )
        self.logits = conv1x1x1(planes*2, n_classes)

        self.relu = nn.ReLU(inplace=True)
        self.normalizer = nn.InstanceNorm3d
        self.pool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

    def forward(self, x):
        features, logits2D = self.UNet(x)
        softmaxed_logits = self.softmax(self.relu(logits2D))
        out_1 = torch.cat([features, softmaxed_logits], dim=1)
        out_2 = self.pool(out_1)

        # First Branch
        for i, op in enumerate(self.first_branch):
            out_1 = op(out_1)
            if DEFAULT_USE_INSTANCE_NORM and i == len(self.first_branch) - 1:
                out_1 = self.normalizer(self.planes * 2)(out_1)
            out_1 = self.relu(out_1)

        # Second Branch
        for j, op in enumerate(self.second_branch):
            out_2 = op(out_2)
            if DEFAULT_USE_INSTANCE_NORM:
                if j == len(self.second_branch) - 2:
                    out_2 = self.normalizer(self.planes * 4)(out_2)
                elif j == len(self.second_branch) - 1:
                    out_2 = self.normalizer(self.planes * 2)(out_2)
            out_2 = self.relu(out_2)
        out_2 = self.transpose_conv(out_2)
        out = out_1 + out_2
        logits3D = self.logits(out)

        return logits2D, logits3D











