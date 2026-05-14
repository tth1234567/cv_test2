import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 1. SE + Spatial Attention (整合版 CBAM) ==========

class SEModule(nn.Module):
    """Squeeze-and-Excitation 模块：纯粹的全局通道注意力"""

    def __init__(self, channels, ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块：关注“哪里是重要的”"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv1(res)
        return self.sigmoid(res) * x


class CBAM_SE(nn.Module):
    """使用 SE 模块作为通道注意力的 CBAM 变体"""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM_SE, self).__init__()
        self.ca = SEModule(in_planes, ratio)  # 替换为 SE
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ========== 2. U-Net 基础组件 ==========

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# ========== 3. UNet + CBAM(SE) 模型 ==========

class UNet_CBAM_SE(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet_CBAM_SE, self).__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # CBAM-SE Attention Gates
        self.att1 = CBAM_SE(64)
        self.att2 = CBAM_SE(128)
        self.att3 = CBAM_SE(256)
        self.att4 = CBAM_SE(512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5)
        x4 = self.att4(x4)  # 应用 SE + Spatial 过滤
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        x3 = self.att3(x3)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        x2 = self.att2(x2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        x1 = self.att1(x1)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)

        return self.outc(u4)


if __name__ == "__main__":
    net = UNet_CBAM_SE()
    print("UNet_CBAM_SE 整合完成。")