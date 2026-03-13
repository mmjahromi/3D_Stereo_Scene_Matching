"""FCGF network architecture (ResUNetBN2C).

Reproduced from https://github.com/chrischoy/FCGF with minor simplifications.
MinkowskiEngine must be installed separately.
"""

from __future__ import annotations

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
import torch.nn as nn


class BasicBlockBN(ME.MinkowskiNetwork):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, D=3):
        super().__init__(D)
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation,
            bias=False, dimension=D
        )
        self.norm1 = ME.MinkowskiBatchNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation,
            bias=False, dimension=D
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResUNetBN2C(ME.MinkowskiNetwork):
    """ResUNet with Batch Norm and 2 residual blocks per level, 4 channels base."""

    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]
    BLOCK = BasicBlockBN

    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        normalize_feature=True,
        conv1_kernel_size=5,
        D=3,
    ):
        super().__init__(D)
        self.normalize_feature = normalize_feature
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
            in_channels, CHANNELS[1], kernel_size=conv1_kernel_size,
            stride=1, dilation=1, bias=False, dimension=D
        )
        self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1])

        self.block1 = self._make_layer(CHANNELS[1], CHANNELS[1], 2, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            CHANNELS[1], CHANNELS[2], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2])
        self.block2 = self._make_layer(CHANNELS[2], CHANNELS[2], 2, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            CHANNELS[2], CHANNELS[3], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3])
        self.block3 = self._make_layer(CHANNELS[3], CHANNELS[3], 2, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            CHANNELS[3], CHANNELS[4], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm4 = ME.MinkowskiBatchNorm(CHANNELS[4])
        self.block4 = self._make_layer(CHANNELS[4], CHANNELS[4], 2, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            CHANNELS[4], TR_CHANNELS[4], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm4_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[4])
        self.block4_tr = self._make_layer(TR_CHANNELS[4] + CHANNELS[3], TR_CHANNELS[4], 2, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            TR_CHANNELS[4], TR_CHANNELS[3], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm3_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[3])
        self.block3_tr = self._make_layer(TR_CHANNELS[3] + CHANNELS[2], TR_CHANNELS[3], 2, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            TR_CHANNELS[3], TR_CHANNELS[2], kernel_size=3,
            stride=2, dilation=1, bias=False, dimension=D
        )
        self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[2])
        self.block2_tr = self._make_layer(TR_CHANNELS[2] + CHANNELS[1], TR_CHANNELS[2], 2, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            TR_CHANNELS[2], out_channels, kernel_size=1,
            stride=1, dilation=1, bias=False, dimension=D
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1, dilation=1, D=3):
        downsample = None
        if stride != 1 or inplanes != planes * self.BLOCK.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    inplanes, planes * self.BLOCK.expansion,
                    kernel_size=1, stride=stride, bias=False, dimension=D
                ),
                ME.MinkowskiBatchNorm(planes * self.BLOCK.expansion),
            )
        layers = [self.BLOCK(inplanes, planes, stride=stride, dilation=dilation,
                              downsample=downsample, D=D)]
        inplanes = planes * self.BLOCK.expansion
        for _ in range(1, num_blocks):
            layers.append(self.BLOCK(inplanes, planes, stride=1, dilation=dilation, D=D))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_s1 = self.relu(self.norm1(self.conv1(x)))
        out_s1 = self.block1(out_s1)

        out_s2 = self.relu(self.norm2(self.conv2(out_s1)))
        out_s2 = self.block2(out_s2)

        out_s4 = self.relu(self.norm3(self.conv3(out_s2)))
        out_s4 = self.block3(out_s4)

        out_s8 = self.relu(self.norm4(self.conv4(out_s4)))
        out_s8 = self.block4(out_s8)

        out = self.relu(self.norm4_tr(self.conv4_tr(out_s8)))
        out = me.cat(out, out_s4)
        out = self.block4_tr(out)

        out = self.relu(self.norm3_tr(self.conv3_tr(out)))
        out = me.cat(out, out_s2)
        out = self.block3_tr(out)

        out = self.relu(self.norm2_tr(self.conv2_tr(out)))
        out = me.cat(out, out_s1)
        out = self.block2_tr(out)

        out = self.conv1_tr(out)

        if self.normalize_feature:
            import MinkowskiEngine.MinkowskiFunctional as MF
            out = MF.normalize(out, p=2, dim=1)

        return out
