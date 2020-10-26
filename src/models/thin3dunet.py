"""Build a thin 3D U-Net, which processes a small number of sequential z-slices
in 3D using z-padded convolutions and litle or no z spatial pooling.

"""
import torch
from torch import nn
import torch.nn.functional as F


class Thin3DUNet(nn.Module):
    def __init__(
            self,
            z_size: int = 5,
            in_channels: int = 1,
            n_classes: int = 2,
            n_convs_per_down_block: int = 2,
            n_convs_per_up_block: int = 2,
            depth: int = 5,
            n_init_filters: int = 32,
            padding: bool = False,
            instance_norm: bool = True,
            up_mode: str = 'upconv',
            separable: bool = True,
            leaky: bool = True):
        """

        Args:
            z_size (int): Initial data size along the z axis. Everywhere
                in the Thin3DUNet, convolutions are padded along the z axis
                and z pooling is disabled to retain this shape throughout
                processing. Must be an odd int.
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            n_convs_per_down_block (int): Number of convolution layers per conv
                block in the downsampling path of the u-net.
            n_convs_per_up_block (int): Number of convolution layers per conv
                block in the upsampling path of the u-net.
            depth (int): depth of the network
            n_init_filters (int): number of filters in the first layer
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            instance_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'bilinear'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'bilinear' will use bilinear upsampling.
            separable (bool): If True, use separable convs instead of regular.
            leaky (bool): If True, use LeakyReLU activation instead of ReLU.
        """
        assert z_size % 2 == 1
        assert up_mode in ('upconv', 'bilinear')
        super().__init__()

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        # Build the downsampling path
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(Thin3DUNetConvBlock(
                z_size,
                n_convs_per_down_block,
                prev_channels,
                n_init_filters * 2**i,
                padding,
                instance_norm,
                separable,
                leaky))
            prev_channels = n_init_filters * 2**i

        # Build the upsampling path
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(Thin3DUNetUpBlock(
                z_size,
                n_convs_per_up_block,
                prev_channels,
                n_init_filters * 2**i,
                up_mode,
                padding,
                instance_norm,
                separable,
                leaky))
            prev_channels = n_init_filters * 2**i

        # Predictor head
        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x = torch.unsqueeze(1-x, dim=1)
        blocks = []
        # Pass x through the downsampling path
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool3d(x, [1, 2, 2])

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class Thin3DUNetConvBlock(nn.Module):
    def __init__(
            self,
            z_size,
            n_convs,
            in_size,
            out_size,
            padding,
            instance_norm,
            separable,
            leaky):
        super().__init__()
        block = []

        conv = SeparableConv3d if separable else nn.Conv3d
        activation = nn.LeakyReLU if leaky else nn.ReLU

        if padding:
            pad = [(z_size - 1) // 2, 1, 1]
        else:
            pad = [(z_size - 1) // 2, 0, 0]

        block.append(conv(
            in_size,
            out_size,
            kernel_size=[z_size, 3, 3],
            padding=pad))
        block.append(activation())

        for n in range(n_convs - 1):
            block.append(conv(
                out_size,
                out_size,
                kernel_size=[z_size, 3, 3],
                padding=pad))
            block.append(activation())
        if instance_norm:
            block.append(nn.InstanceNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class Thin3DUNetUpBlock(nn.Module):
    def __init__(
            self,
            z_size,
            n_convs,
            in_size,
            out_size,
            up_mode,
            padding,
            instance_norm,
            separable,
            leaky):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(
                in_size,
                out_size,
                kernel_size=[1, 2, 2],
                stride=[1, 2, 2])
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=[1, 2, 2]),
                nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = Thin3DUNetConvBlock(
            z_size,
            n_convs,
            in_size,
            out_size,
            padding,
            instance_norm,
            separable,
            leaky)

    def center_crop(self, layer, target_size):
        _, _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[3:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class SeparableConv3d(nn.Module):
    def __init__(self, nin, nout, *args, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv3d(nin, nin, *args, groups=nin, **kwargs)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1)
        pass

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
