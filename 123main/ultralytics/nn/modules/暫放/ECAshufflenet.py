import torch
import torch.nn as nn

class Conv_maxpool(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.maxpool(self.conv(x))

class ECA(nn.Module):
    """ECA Attention with dynamic kernel size"""
    def __init__(self, kernel_size=3, max_kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size if kernel_size <= max_kernel_size else max_kernel_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)

class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride, eca_kernel_size=3, use_eca=True, max_eca_kernel_size=5):
        super().__init__()
        self.stride = stride
        self.use_eca = use_eca
        if self.use_eca:
            self.eca = ECA(kernel_size=eca_kernel_size, max_kernel_size=max_eca_kernel_size)

        # Ensure branch features are divided by 2 to match expected output channels
        branch_features = oup // 2  # output channels should be evenly divisible by 2
        assert (self.stride != 1) or (inp == branch_features * 2)

        # Define the first branch (branch1) with specific stride and convolution settings
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        # Modify the second branch (branch2) to ensure it matches the input and output channels
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if self.stride == 2 else branch_features, branch_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass for branch1 and branch2
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # Ensure that after concatenation, the channel number is correct
        out = torch.cat((branch1_out, branch2_out), dim=1)

        # Ensure that after concatenation the channel number is divisible by 2 for shuffle operation
        out = self.channel_shuffle(out, 2)

        if self.use_eca:
            out = self.eca(out)  # Apply ECA after channel shuffle

        return out

    def channel_shuffle(self, x, groups):
        """Channel shuffle operation"""
        N, C, H, W = x.size()
        return x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
