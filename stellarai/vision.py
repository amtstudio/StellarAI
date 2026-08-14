"""
StellarAI 视觉处理模块
- VisionPreprocessor: 图像预处理 (纯Python/PIL，无外部依赖也可降级)
- VisionEncoder: 轻量级CNN + ViT混合编码器
"""

import math
from typing import Optional, Tuple, List

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class VisionPreprocessor:
    """
    图像预处理器
    功能：
    - 加载/缩放图像
    - 归一化
    - 转换为模型输入格式
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def load_image(self, path: str) -> np.ndarray:
        """加载图像为RGB numpy数组 (H, W, C)"""
        if not HAS_PIL:
            raise ImportError("需要安装Pillow: pip install pillow")
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _resize(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """使用PIL双线性插值缩放"""
        if not HAS_PIL:
            # 退化：简单最近邻
            h, w, c = img.shape
            ratio = target_size / max(h, w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            result = np.zeros((target_size, target_size, c), dtype=np.uint8)
            sh, sw = target_size // new_h if new_h > 0 else 1, target_size // new_w if new_w > 0 else 1
            off_h = (target_size - new_h * sh) // 2
            off_w = (target_size - new_w * sw) // 2
            result[off_h:off_h + new_h * sh:sh, off_w:off_w + new_w * sw:sw, :] = img[::max(1, h // new_h), ::max(1, w // new_w), :]
            return result

        pil_img = Image.fromarray(img)
        # Resize保持长宽比，padding补0
        h, w = pil_img.height, pil_img.width
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        new_img.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        return np.array(new_img, dtype=np.uint8)

    def preprocess(self, image, normalize: bool = True) -> np.ndarray:
        """
        预处理图像
        Args:
            image: str (路径) 或 np.ndarray (H,W,C) 或 PIL.Image
            normalize: 是否归一化
        Returns:
            np.ndarray: (C, H, W) 或 (N, C, H, W)
        """
        # 解析输入
        if isinstance(image, str):
            img = self.load_image(image)
        elif HAS_PIL and isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                # 批量处理
                batch = []
                for i in range(image.shape[0]):
                    batch.append(self.preprocess(image[i], normalize))
                return np.stack(batch, axis=0)
            img = image.copy()
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        # 缩放
        img = self._resize(img, self.image_size)

        # (H, W, C) -> (C, H, W)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))

        # 归一化到 [0, 1]
        if normalize:
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std

        return img.astype(np.float32)

    def extract_patches(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        将图像划分为patch
        Args:
            image_tensor: (C, H, W) or (N, C, H, W)
        Returns:
            (num_patches, patch_dim) or (N, num_patches, patch_dim)
            patch_dim = C * patch_size^2
        """
        p = self.patch_size
        if image_tensor.ndim == 3:
            C, H, W = image_tensor.shape
            n_h, n_w = H // p, W // p
            patches = image_tensor.reshape(C, n_h, p, n_w, p)
            patches = patches.transpose(1, 3, 0, 2, 4)
            patches = patches.reshape(n_h * n_w, -1)
            return patches
        else:
            N, C, H, W = image_tensor.shape
            n_h, n_w = H // p, W // p
            patches = image_tensor.reshape(N, C, n_h, p, n_w, p)
            patches = patches.transpose(0, 2, 4, 1, 3, 5)
            patches = patches.reshape(N, n_h * n_w, -1)
            return patches
