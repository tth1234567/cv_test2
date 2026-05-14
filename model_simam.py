import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 1. SimAM 无参注意力模块 (直接内置) ==========
class SimAM(nn.Module):
    """
    SimAM: Simple, Parameter-Free Attention Module
    利用能量函数计算3D注意力权重。
    优势：不引入任何可学习参数，直接利用现有特征图的分布进行加权。
    """

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # b: batch_size, c: channels, h: height, w: width
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算特征图的空间能量分布
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# ========== 2. U-Net 基础组件 ==========
class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""

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


# ========== 3. 集成了 SimAM 的 U-Net 模型 ==========
class UNet_SimAM(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet_SimAM, self).__init__()

        # --- 编码器 (Downsampling) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- SimAM 注意力门 (放在跳跃连接处) ---
        # 针对每一层 Skip Connection 放置一个 SimAM 模块
        self.simam1 = SimAM()  # 处理 64 通道 (x1)
        self.simam2 = SimAM()  # 处理 128 通道 (x2)
        self.simam3 = SimAM()  # 处理 256 通道 (x3)
        self.simam4 = SimAM()  # 处理 512 通道 (x4)

        # --- 解码器 (Upsampling) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)