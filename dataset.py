import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np  # 记得在最上面加上这行导入

class MinneAppleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        """
        初始化数据集
        :param images_dir: 原图所在的文件夹路径
        :param masks_dir: 掩码图(Mask)所在的文件夹路径
        :param img_size: 缩放后的统一尺寸 (默认256x256，保护显存)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        # 获取所有原图的文件名列表 (例如: ['20150919_174151_image1.png', ...])
        self.image_names = os.listdir(images_dir)

    def __len__(self):
        # 告诉 PyTorch 一共有多少对图片
        return len(self.image_names)



    def __getitem__(self, idx):
        # 1. 拼接路径
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # 2. 读取图片
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 3. 图像尺寸缩放
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 4. 原图正常转换为 Tensor
        image = T.ToTensor()(image)

        # ==========================================
        # 5. Mask 终极修复逻辑（防实例分割陷阱）
        # ==========================================
        # 先把 Mask 转成 numpy 数组来看它真实的底牌
        mask_np = np.array(mask)

        # 霸道逻辑：只要像素值大于 0（不管是 1, 2 还是 255），统统变成 1（苹果）
        mask_binary = (mask_np > 0).astype(np.float32)

        # 把 numpy 数组转回 PyTorch 张量，并手动增加一个通道维度使其变成 [1, 256, 256]
        mask = torch.from_numpy(mask_binary).unsqueeze(0)

        return image, mask


# ==========================================
# 测试一下代码是否写对了 (非常重要的一步！)
# ==========================================
if __name__ == "__main__":
    # 替换成你电脑上的真实路径
    TRAIN_IMG_DIR = r"D:\cv_lesson\test_2\data\detection\train\images"
    TRAIN_MASK_DIR = r"D:\cv_lesson\test_2\data\detection\train\masks"

    # 实例化数据集
    train_dataset = MinneAppleDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, img_size=256)

    print(f"成功加载数据集，共有图片: {len(train_dataset)} 张")

    # 取出第一对图片看看
    first_image, first_mask = train_dataset[0]
    print(f"原图 Tensor 形状: {first_image.shape}")  # 预期输出: [3, 256, 256]
    print(f"掩码 Tensor 形状: {first_mask.shape}")  # 预期输出: [1, 256, 256]
    print(f"掩码里的唯一数值: {torch.unique(first_mask)}")  # 预期输出: [0., 1.] (代表只有黑底和白苹果)