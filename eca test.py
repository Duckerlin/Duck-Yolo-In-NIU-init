import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義 ECA 模塊
class ECA(nn.Module):
    """ECA Attention"""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)

# 定義 ShuffleNetV2 模塊
class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride, eca_kernel_size=3, use_eca=True):
        super().__init__()
        self.stride = stride
        self.use_eca = use_eca
        if self.use_eca:
            self.eca = ECA(kernel_size=eca_kernel_size)

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features * 2)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

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
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            branch2_out = self.branch2(x2)
            out = torch.cat((x1, branch2_out), dim=1)
        else:
            branch1_out = self.branch1(x)
            branch2_out = self.branch2(x)
            out = torch.cat((branch1_out, branch2_out), dim=1)

        out = self.channel_shuffle(out, 2)
        if self.use_eca:
            out = self.eca(out)
        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        return x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

# 測試程式：檢查 ECA 是否被成功使用
def test_shuffle_eca():
    # 模擬一個 batch 的輸入圖像，大小為 (batch_size=1, channels=64, height=32, width=32)
    input_tensor = torch.randn(1, 64, 32, 32)

    # 使用 ShuffleNetV2 模塊
    model = ShuffleNetV2(64, 128, stride=2, use_eca=True)
    output = model(input_tensor)

    # 檢查是否有 ECA 的應用，通過觀察特徵圖的變化
    print("Input Tensor Shape: ", input_tensor.shape)
    print("Output Tensor Shape: ", output.shape)

    # 檢查是否 ECA 改變了特徵圖，這可以通過查看前後的平均激活值來判斷
    input_activation_mean = input_tensor.mean().item()
    output_activation_mean = output.mean().item()
    print(f"Input Mean Activation: {input_activation_mean}")
    print(f"Output Mean Activation after ECA: {output_activation_mean}")

    # 判斷是否 ECA 有效果，若改變了平均激活值，則可認為 ECA 影響了特徵圖
    if abs(input_activation_mean - output_activation_mean) > 0.01:
        print("ECA is applied successfully!")
    else:
        print("ECA is not applied or has minimal effect.")

# 呼叫測試函數
test_shuffle_eca()
