# 🍎 Fruit Segmentation in Orchards (MinneApple)

基于 U-Net 架构与 CBAM 注意力机制的水果语义分割项目 (实验二)

---

## 📖 项目简介 (Introduction)
本项目针对农业场景中水果采摘视觉算法的痛点，使用深度学习技术对 [MinneApple 数据集](https://github.com/nicolaihaeni/MinneApple) 进行像素级语义分割。

针对果园环境中严重的**枝叶遮挡**、**背光阴影**以及**小目标密集**等问题，本项目在标准 U-Net 编码器-解码器架构的基础上，创造性地引入了 **CBAM (Convolutional Block Attention Module)** 机制。通过通道注意力与空间注意力的双重聚焦，模型在复杂背景下的分割精度与召回率得到了显著提升。

---

## ✨ 核心特性 (Features)
- **网络架构重构**：构建了纯净版 U-Net (Baseline) 以及深度集成了 CBAM 模块的改进版架构。
- **自定义数据集处理**：设计了严谨的数据预处理流水线，解决掩码插值模糊与实例分割数值极小导致的数据失真陷阱。
- **混合损失函数**：采用自适应的 `Dice Loss + BCE Loss` 混合损失函数，有效应对医学/农业图像中常见的正负样本极端不平衡问题。
- **动态可视化**：在验证环节加入了每 N 个 Epoch 的推理拦截机制，动态保存模型认知演化过程图。

---

## 📁 目录结构 (Directory Structure)

为保持仓库整洁，巨大的数据集文件与训练生成的权重文件已被 `.gitignore` 过滤。本地项目核心树如下：

```text
cv_test2/
├── model.py                    # 🧠 U-Net 基准模型架构定义
├── model_cbam_se.py            # 🧠 结合 CBAM 注意力机制的改进模型定义
├── dataset.py                  # 🗂️ 自定义 MinneApple 数据集加载器与预处理逻辑
├── train.py                    # 🚀 核心训练脚本 (包含验证、日志记录与可视化)
├── environment_settings.pdf    # 📝 实验环境与超参数配置详情 (PDF)
└── README.md                   # 📖 项目说明文档