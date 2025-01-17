import torch
import torch.nn as nn


class Conv_maxpool(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        assert stride in [1, 2]
        mid_channels = out_channels // 2
        self.stride = stride

        if stride == 2:  # Down-sampling
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride == 2 else mid_channels, mid_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        assert x.size(1) % 2 == 0, "Input channel number must be divisible by 2 for channel shuffle."
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(batchsize, -1, height, width)


class ShuffleNetV2_CSP(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=1):
        super().__init__()
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.downsample_bn = nn.BatchNorm2d(out_channels)
        self.downsample_act = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            *[BasicBlock(out_channels // 2, out_channels // 2) for _ in range(n_blocks)]
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, 1, stride=1, bias=False)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        if x.shape[-1] < 2 or x.shape[-2] < 2:
            raise ValueError(f"Spatial dimensions too small: {x.shape[-2:]} (height, width)")

        x = self.downsample_act(self.downsample_bn(self.downsample_conv(x)))
        print(f"After downsample_conv shape: {x.shape}")

        x1, x2 = x.chunk(2, dim=1)
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        x2 = self.blocks(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.final_act(self.final_bn(self.final_conv(x)))
        print(f"Output shape: {x.shape}")
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x
