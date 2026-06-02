# 🍎 实验二：水果分割算法设计与实现

> **课程**：计算机视觉实验 ｜ **班级**：计232 ｜ **姓名**：孙朝阳 ｜ **学号**：234903  
> **完整代码仓库**：[https://github.com/tth1234567/cv_test2](https://github.com/tth1234567/cv_test2)

---

## 📋 目录

- [实验概述](#实验概述)
- [数据集说明](#数据集说明)
- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [数据预处理](#数据预处理)
- [模型架构](#模型架构)
- [损失函数设计](#损失函数设计)
- [训练方法](#训练方法)
- [消融实验结果](#消融实验结果)
- [可视化分析](#可视化分析)
- [实验结论](#实验结论)

---

## 实验概述

本实验针对复杂果园环境下的苹果检测与分割任务，在 **MinneApple** 数据集上设计并实现了基于深度学习的语义分割算法。

**核心工作包括：**

- 从零构建标准 **U-Net** 编码器-解码器架构作为 Baseline
- 设计定制化数据预处理流水线，解决实例掩码二值化陷阱
- 引入 **CBAM 注意力机制**（SE通道注意力 + 空间注意力串联），提升模型对遮挡目标的感知能力
- 设计 **混合损失函数**（Dice Loss + BCE Loss），缓解前景背景类别不平衡
- 进行完整消融实验，从 IoU、Precision、Recall、Pixel Accuracy 四个维度量化评估

**核心挑战：**

| 挑战类型 | 描述 |
|---|---|
| 枝叶遮挡 | 苹果目标被大量树叶、枝干横向遮挡 |
| 光照不均 | 果园强光/阴影交替，造成目标色彩失真 |
| 同色背景 | 青色果实与大量绿叶背景色彩相近，难以区分 |
| 小目标密集 | 图像中存在大量紧密排列的小苹果个体 |

---

## 数据集说明

本实验使用明尼苏达大学开源的 **MinneApple** 苹果检测与分割数据集。

```
数据集规模：670 张带有像素级掩码标注的果园采摘图像
标注类型：实例级掩码（不同苹果个体像素值标记为 1, 2, 3...）
划分策略：8:2 随机划分，固定随机种子 seed=42
```

| 数据子集 | 图像数量 | 说明 |
|---|---|---|
| 训练集 | 536 张 | 用于模型参数学习 |
| 验证集 | 134 张 | 用于消融实验对比评估 |

> ⚠️ **注意**：官方测试集标注未公开（用于线上打榜），本实验采用本地划分验证集的学术标准策略，固定随机种子确保所有消融实验数据分布完全一致。

---

## 环境配置

### 硬件环境

| 组件 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB 显存) |
| CPU | Intel Core i7 |
| RAM | 16GB |
| CUDA | 11.x / 12.x |

### 软件环境

```bash
# 克隆仓库
git clone https://github.com/tth1234567/cv_test2.git
cd cv_test2

# 安装依赖
pip install torch torchvision numpy pillow matplotlib tqdm
```

| 依赖库 | 版本要求 | 用途 |
|---|---|---|
| Python | 3.8+ | 编程语言 |
| PyTorch | 2.x | 深度学习框架 |
| NumPy | - | 数组与矩阵运算 |
| Pillow | - | 图像读取与预处理 |
| Matplotlib | - | 训练曲线与结果可视化 |
| tqdm | - | 训练进度显示 |

---

## 项目结构

```
cv_test2/
├── data/
│   └── MinneApple/
│       ├── images/          # 原始果园图像 (.jpg)
│       └── masks/           # 像素级实例掩码 (.png)
├── models/
│   ├── unet.py              # 标准 U-Net 架构实现
│   └── cbam_unet.py         # CBAM-UNet 改进模型
├── utils/
│   ├── dataset.py           # 自定义数据集与预处理流水线
│   ├── loss.py              # Dice+BCE 混合损失函数
│   └── metrics.py           # IoU / Precision / Recall / PixelAcc 评估
├── train.py                 # 训练主脚本（含验证集推理拦截与可视化）
├── evaluate.py              # 模型评估脚本
└── README.md
```

---

## 数据预处理

### 核心预处理流水线

本实验针对 MinneApple 数据集的特殊性，设计了三步防失真预处理流水线：

#### Step 1：非对称插值缩放（防边缘脏数据）

将所有图像统一缩放至 **256×256** 分辨率，但对图像和掩码采用不同的插值方式：

```python
# 彩色原图：双线性插值，保持视觉平滑
image = image.resize((256, 256), Image.BILINEAR)

# 二值掩码：最近邻插值，防止边界产生 0.5 类中间值
mask = mask.resize((256, 256), Image.NEAREST)
```

> ✅ **关键设计**：掩码若使用双线性插值，会在苹果与背景交界处产生 0~1 之间的过渡值，破坏二值语义。强制使用最近邻插值确保掩码始终"非黑即白"。

#### Step 2：实例掩码 → 语义掩码强制二值化

MinneApple 的原始掩码为**实例分割格式**（不同苹果标记为极小整数 1, 2, 3...），若直接除以 255 进行归一化，这些微小值会被视为黑色背景，导致掩码全黑。

```python
import numpy as np

# ❌ 错误做法：直接归一化会把苹果像素当背景
# mask_tensor = transforms.ToTensor()(mask)  # 极小值 → 近似0 → 丢失标注

# ✅ 正确做法：提取底层数组，逻辑运算强制二值化
mask_np = np.array(mask)
mask_binary = (mask_np > 0).astype(np.float32)  # 所有非零像素 → 1.0（苹果）
mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)
```

#### Step 3：张量化与格式对齐

```python
# 图像：[H, W, C] → [C, H, W]，像素值 [0,255] → [0.0, 1.0]
image_tensor = transforms.ToTensor()(image)

# 掩码：[H, W] → [1, H, W]，值域 {0.0, 1.0}
# （如 Step 2 所示）
```

---

## 模型架构

### Baseline：标准 U-Net

U-Net 采用对称的**编码器-解码器**结构，通过**跳跃连接（Skip Connection）** 融合深层语义与浅层空间细节。

```
输入 (3×256×256)
    │
    ▼
┌─────────────────────────────────────────────┐
│  编码器（Encoder）—— 下采样路径              │
│  Conv Block(64) → MaxPool → Skip①           │
│  Conv Block(128) → MaxPool → Skip②          │
│  Conv Block(256) → MaxPool → Skip③          │
│  Conv Block(512) → MaxPool → Skip④          │
│  Bottleneck Conv Block(1024)                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  解码器（Decoder）—— 上采样路径              │
│  TransConv + Concat(Skip④) + Conv Block     │
│  TransConv + Concat(Skip③) + Conv Block     │
│  TransConv + Concat(Skip②) + Conv Block     │
│  TransConv + Concat(Skip①) + Conv Block     │
└─────────────────────────────────────────────┘
    │
    ▼
输出 (1×256×256) — Sigmoid 激活
```

### 改进模型：CBAM-UNet

在标准 U-Net 的每个编码阶段和瓶颈层后插入 **CBAM 注意力模块**，对跳跃连接传递的特征进行双重重构：

#### CBAM 模块结构

**① SE 通道注意力（回答"哪些通道更重要"）**

```python
class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)          # 全局平均池化：H×W → 1×1
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False),        # 降维：c → c/r
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),        # 升维：c/r → c
            nn.Sigmoid()                             # 输出通道权重 ∈ [0,1]
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.fc(self.pool(x).view(B, C))         # 通道权重
        return x * w.view(B, C, 1, 1)               # 逐通道加权
```

**② 空间注意力（回答"哪些位置更重要"）**

```python
class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)            # 通道平均
        max_f, _ = x.max(dim=1, keepdim=True)        # 通道最大
        spatial_w = self.sigmoid(self.conv(
            torch.cat([avg, max_f], dim=1)           # 拼接 → 2×H×W
        ))
        return x * spatial_w                         # 空间位置加权
```

**③ CBAM 串联（先通道，后空间）**

```python
class CBAM(nn.Module):
    def __init__(self, c: int, r: int = 16, k: int = 7):
        super().__init__()
        self.channel = SE(c, r)
        self.spatial = SpatialAttention(k)

    def forward(self, x):
        x = self.channel(x)   # Step1：通道校准（排除噪声通道）
        x = self.spatial(x)   # Step2：空间定位（聚焦目标区域）
        return x
```

---

## 损失函数设计

针对苹果分割任务中**前景（苹果）占比远小于背景**的类别不平衡问题，设计混合损失函数：

```python
class DiceBCELoss(nn.Module):
    """
    混合损失 = Dice Loss + BCE Loss
    
    Dice Loss：直接优化 IoU 指标，对小目标和类别不平衡鲁棒
    BCE Loss：逐像素交叉熵，提供稳定的梯度信号
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE 损失（pred 为 logits，内部自动 Sigmoid）
        bce_loss = self.bce(pred, target)

        # Dice 损失
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )

        return bce_loss + dice_loss.mean()
```

---

## 训练方法

### 超参数配置

| 超参数 | 设定值 | 说明 |
|---|---|---|
| 模型架构 | U-Net / CBAM-UNet | Baseline 与改进模型 |
| 输入尺寸 | 256 × 256 | 兼顾特征提取与显存开销 |
| 优化器 | Adam | 自适应学习率，收敛快 |
| 初始学习率 | 1e-4 | 保证训练稳定性 |
| Batch Size | 8 | 适配 8GB 显存 |
| 训练轮数 | 50 Epochs | 确保充分收敛 |
| 损失函数 | DiceBCELoss | 解决类别不平衡 |
| 数据划分 | 8:2 (seed=42) | 固定随机种子 |

### 训练命令

```bash
# 训练 Baseline U-Net
python train.py --model unet --epochs 50 --batch_size 8 --lr 1e-4

# 训练 CBAM-UNet 改进模型
python train.py --model cbam_unet --epochs 50 --batch_size 8 --lr 1e-4
```

---

## 消融实验结果

两组模型在验证集上达到最优 IoU 时的完整指标对比：

| 模型架构 | 最优轮次 | IoU（最优）| Precision | Recall | Pixel Acc |
|---|---|---|---|---|---|
| **Baseline (U-Net)** | 36 | 0.6505 | 0.8250 | 0.7565 | 0.9886 |
| **CBAM-UNet（改进）** | **24** | **0.6626** | 0.7922 | **0.8040** | 0.9885 |

**核心结论：**

- **精度提升**：CBAM-UNet 最优 IoU 达到 **66.26%**，相较 Baseline（65.05%）提升 **+1.21%**
- **收敛提速**：改进模型仅需 **24 轮**即达到峰值，比 Baseline（36 轮）**提前 12 轮收敛**，证明注意力机制能引导网络更快学习关键特征
- **召回率跃升**：Recall 从 75.65% 大幅提升至 **80.40%（+4.75%）**，证明 CBAM 有效降低了遮挡场景下的漏检率

---

## 可视化分析

### Baseline 训练收敛过程

| 训练阶段 | 现象描述 |
|---|---|
| Epoch 5（初期） | 输出大面积不规则激活色块，无法贴合水果边缘 |
| Epoch 25（中期） | 具备基本目标定位能力，但掩码内部有空洞，边缘毛刺多 |
| Epoch 50（末期） | 能识别大目标，但光照不足或严重遮挡区域仍有边缘混淆 |

### CBAM-UNet 的抗干扰优势

在被枝叶横向遮挡或处于边缘背光处的苹果目标上：

- **Baseline**：缺乏针对性的特征权重分配，将遮挡目标误判为背景（**漏检**）
- **CBAM-UNet**：通道注意力提取特异性色泽特征 + 空间注意力局部强化，成功在树叶缝隙中抠出完整苹果轮廓

这与 Recall 指标大幅提升的数据结论完全吻合。

---

## 实验结论

通过本次实验得到以下核心结论：

**1. 数据预处理是模型成败的隐形决定者**

MinneApple 实例掩码的二值化陷阱说明，构建严格防脏化的数据流水线，其重要性不亚于设计复杂的网络架构。

**2. 注意力机制在复杂遮挡场景下有显著效果**

CBAM 通过通道与空间的双重注意力，赋予模型"穿透遮挡"的能力。Recall 提升近 5 个百分点，在农业采摘这种极度厌恶漏检的业务场景中，意义重大。

**3. 损失函数选择需匹配任务特性**

Dice Loss + BCE Loss 的混合设计，直接优化 IoU 指标，有效缓解了苹果前景像素远少于背景的类别不平衡问题。

---

*实验报告完整版见 `计算机视觉实验2.docx`，可视化结果图见 `runs/` 目录。*
