import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt

# 导入你定义的模型类
from model import UNet
from model_simam import UNet_SimAM
from model_cbam_se import UNet_CBAM_SE
from dataset import MinneAppleDataset
from torch.utils.data import random_split


# ========== 1. 损失函数 (Dice + BCE) ==========
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. 极其稳定的 BCE 计算方式（直接输入未经 Sigmoid 的 logits）
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

        # 2. 计算 Dice 时依然需要 Sigmoid 将数值压到 0~1 之间
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_sigmoid * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_sigmoid.sum() + targets_flat.sum() + self.smooth)

        return bce_loss + dice_loss


# ========== 2. 核心指标评估 (全家桶) ==========
def calculate_metrics(pred, target):
    """
    除了 IoU，还计算了 Precision (精确率), Recall (召回率) 和 Pixel Accuracy。
    这些指标能让你在报告中像 YOLO 一样展示多维度的性能。
    """
    pred = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target = target.view(-1)

    tp = (pred * target).sum().item()  # 真正例
    fp = (pred * (1 - target)).sum().item()  # 假正例
    fn = ((1 - pred) * target).sum().item()  # 假负例
    tn = ((1 - pred) * (1 - target)).sum().item()  # 真负例

    # 1. Pixel Accuracy (像素准确率)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    # 2. IoU (交并比)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)

    # 3. Precision (精确率: 预测的苹果中有多少是真的)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)

    # 4. Recall (召回率: 真的苹果中有多少被预测到了)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)

    return acc, iou, precision, recall


# ========== 3. 辅助功能：目录管理与可视化 ==========
def get_exp_dir(base_path="./runs/train", name="exp"):
    os.makedirs(base_path, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base_path, f"{name}{i}")): i += 1
    exp_dir = os.path.join(base_path, f"{name}{i}")
    os.makedirs(exp_dir);
    os.makedirs(os.path.join(exp_dir, "weights"));
    os.makedirs(os.path.join(exp_dir, "visuals"))
    return exp_dir


def save_visuals(model, dataset, device, save_dir, epoch):
    model.eval()
    with torch.no_grad():
        for i in range(2):  # 选 2 张图展示
            image, mask = dataset[i * 20]
            input_tensor = image.unsqueeze(0).to(device)
            pred = (torch.sigmoid(model(input_tensor)) > 0.5).float().cpu().squeeze()

            # img_show = image.permute(1, 2, 0).numpy()
            # 只需要确保数值在 0-1 之间即可，不需要做复杂的反标准化
            img_show = np.clip(image.permute(1, 2, 0).cpu().numpy(), 0, 1)
            img_show = np.clip((img_show * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

            plt.figure(figsize=(12, 4))
            plt.subplot(131);
            plt.imshow(img_show);
            plt.title("Image");
            plt.axis('off')
            plt.subplot(132);
            plt.imshow(mask.squeeze(), cmap='gray');
            plt.title("Label");
            plt.axis('off')
            plt.subplot(133);
            plt.imshow(pred, cmap='gray');
            plt.title(f"Pred (Ep {epoch})");
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_sample_{i}.jpg"))
            plt.close()


# ========== 4. 训练主函数 ==========
def run_train():
    # 配置
    MODEL_TYPE = 'cbam_se'  # 切换: 'baseline', 'simam', 'cbam_se'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EXP_DIR = get_exp_dir()

    # 初始化
    if MODEL_TYPE == 'baseline':
        model = UNet().to(DEVICE)
    elif MODEL_TYPE == 'simam':
        model = UNet_SimAM().to(DEVICE)
    else:
        model = UNet_CBAM_SE().to(DEVICE)

    # 只需要官方的 train 文件夹！因为只有它里面有真实的 mask 答案
    train_img_dir = r'D:\cv_lesson\test_2\data\detection\train\images'
    train_msk_dir = r'D:\cv_lesson\test_2\data\detection\train\masks'

    # 1. 读取全部 670 张有标签的数据
    full_dataset = MinneAppleDataset(images_dir=train_img_dir, masks_dir=train_msk_dir)

    # 2. 按 8:2 的比例切分出训练集和验证集 (80%用来训练，20%用来考试)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # 使用 PyTorch 官方工具进行随机打乱并切分
     # 固定生成器种子(generator)保证每次切分的数据一样，方便做对比实验
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
         generator=torch.Generator().manual_seed(42)
    )

    # 后面的 DataLoader 保持不变！
    # train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
     # val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)



    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    # 日志记录
    log_file = open(os.path.join(EXP_DIR, "results.csv"), 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'loss', 'acc', 'iou', 'precision', 'recall'])

    best_iou = 0
    print(f"🚀 开始训练 {MODEL_TYPE}，结果保存至 {EXP_DIR}")

    for epoch in range(50):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/50")
        for imgs, msks in pbar:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, msks)
            loss.backward();
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        m_acc, m_iou, m_pre, m_rec = [], [], [], []
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                out = model(imgs)
                acc, iou, pre, rec = calculate_metrics(out, msks)
                m_acc.append(acc);
                m_iou.append(iou);
                m_pre.append(pre);
                m_rec.append(rec)

        avg_iou = np.mean(m_iou)
        writer.writerow(
            [epoch + 1, total_loss / len(train_loader), np.mean(m_acc), avg_iou, np.mean(m_pre), np.mean(m_rec)])
        log_file.flush()

        print(f"📊 Val: IoU={avg_iou:.4f}, Acc={np.mean(m_acc):.4f}, Precision={np.mean(m_pre):.4f}")

        # 可视化与保存
        if (epoch + 1) % 5 == 0:
            save_visuals(model, val_ds, DEVICE, os.path.join(EXP_DIR, "visuals"), epoch + 1)
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), os.path.join(EXP_DIR, "weights", "best.pth"))

    log_file.close()


if __name__ == "__main__":
    run_train()