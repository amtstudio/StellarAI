"""
StellarAI - 轻量级可商用多模态AI大模型 (0.05B参数)
支持文本+图像多模态输入，可在低配置设备上运行
"""

__version__ = "1.0.0"
__author__ = "StellarAI Team"
__license__ = "MIT"

from .model import StellarAI
from .config import StellarConfig
from .tokenizer import SimpleTokenizer
from .vision import VisionPreprocessor

__all__ = [
    "StellarAI",
    "StellarConfig",
    "SimpleTokenizer",
    "VisionPreprocessor",
]
