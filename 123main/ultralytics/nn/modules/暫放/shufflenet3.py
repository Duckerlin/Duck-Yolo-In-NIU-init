import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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


class InvertedResidualWithSE(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidualWithSE, self).__init__()
        self.block = ShuffleNetV2(inp, oup, stride)
        self.se = SEBlock(oup)

    def forward(self, x):
        x = self.block(x)
        x = self.se(x)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out


class ShuffleNetV2_SE(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super(ShuffleNetV2_SE, self).__init__()
        self.stages_repeats = stages_repeats
        self.stages_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self.stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idx_stage, num_repeat in enumerate(self.stages_repeats):
            output_channels = self.stages_out_channels[idx_stage + 1]
            for i in range(num_repeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidualWithSE(input_channels, output_channels, stride))
                input_channels = output_channels
        self.features = nn.Sequential(*self.features)

        output_channels = self.stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.fc(x)
        return x


def make_divisible(x, divisor=8):
    """
    确保 `x` 是 `divisor` 的倍数。如果不是，则向上取整到最近的倍数。
    """
    return int((x + divisor - 1) // divisor * divisor)


# 修改 parse_model 函數
def parse_model(m, args, ch, nc, width):
    """
    解析并构建模型的配置。
    :param m: 当前模型类
    :param args: 模型参数
    :param ch: 输入通道数
    :param nc: 类别数
    :param width: 宽度因子
    :return: 修改后的参数
    """
    if m == ShuffleNetV2_SE:  # 处理 ShuffleNetV2_SE
        c1, c2 = ch[args[0]], args[1]
        if c2 != nc:
            c2 = make_divisible(c2 * width, 8)  # 确保输出通道是8的倍数
        args = [c1, c2, *args[2:]]
    elif m == Conv_maxpool:  # 处理 Conv_maxpool
        c1, c2 = ch[args[0]], args[1]
        args = [c1, c2]
    else:
        # 默认处理其他模型（您可以根据需要补充其他模型类型）
        pass
    return args
