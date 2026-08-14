"""
StellarAI 模型配置
设计目标：总参数量约 0.05B (50M)
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class StellarConfig:
    """
    StellarAI 0.05B 参数配置
    文本编码器: ~28M 参数
    视觉编码器: ~14M 参数
    融合层+输出头: ~8M 参数
    总计: ~50M 参数 (0.05B)
    """

    # ============== 通用配置 ==============
    d_model: int = 512              # 模型维度
    vocab_size: int = 32000         # 词表大小
    max_seq_len: int = 2048         # 最大序列长度
    dropout: float = 0.1            # Dropout概率
    layer_norm_eps: float = 1e-6    # LayerNorm epsilon

    # ============== 文本编码器配置 (Tiny Transformer) ==============
    # ~28M 参数: 6层 × (512×8×4 + 512×2048×2) ≈ 28M
    text_num_layers: int = 6        # Transformer层数
    text_num_heads: int = 8         # 注意力头数
    text_ff_dim: int = 2048         # FFN中间维度 (4×d_model)
    text_use_flash_attn: bool = False  # 是否使用FlashAttention
    text_rope_base: float = 10000.0     # RoPE base

    # ============== 视觉编码器配置 (轻量级CNN + ViT混合) ==============
    # ~14M 参数: CNN特征提取(~8M) + ViT层(~6M)
    image_size: int = 224           # 输入图像尺寸
    image_channels: int = 3         # 图像通道数
    patch_size: int = 16            # Patch大小 (224/16=196 patches)
    vision_num_patches: int = 196   # Patch数量 (14×14)
    vision_cnn_channels: tuple = (32, 64, 128, 256)  # CNN各层通道数
    vision_num_layers: int = 4      # ViT层数
    vision_num_heads: int = 8       # ViT注意力头数
    vision_ff_dim: int = 1024       # ViT FFN维度

    # ============== 多模态融合配置 ==============
    fusion_num_layers: int = 2      # 融合层数
    fusion_num_heads: int = 8       # 融合注意力头数
    fusion_ff_dim: int = 2048       # 融合FFN维度
    fusion_cross_attn: bool = True  # 是否使用交叉注意力

    # ============== 输出头配置 ==============
    lm_head_bias: bool = True       # LM头是否有bias
    use_weight_tying: bool = True   # 权重绑定(输入embedding与输出头)

    # ============== 训练配置 ==============
    learning_rate: float = 1e-4     # 学习率
    weight_decay: float = 0.01      # 权重衰减
    warmup_steps: int = 1000        # Warmup步数
    batch_size: int = 8             # 批大小
    grad_clip: float = 1.0          # 梯度裁剪

    # ============== 分词器配置 ==============
    tokenizer_pad_id: int = 0
    tokenizer_bos_id: int = 1
    tokenizer_eos_id: int = 2
    tokenizer_unk_id: int = 3

    # ============== 特殊token (与SimpleTokenizer.SPECIAL_TOKENS保持一致) ==============
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    sep_token_id: int = 6
    vision_token_id: int = 7     # 图像占位符token (单patch位置)
    boi_token_id: int = 8       # 图像块开始
    eoi_token_id: int = 9       # 图像块结束
    # 保留0-127作特殊用途
    reserved_special_end: int = 128

    def __post_init__(self):
        """初始化检查"""
        assert self.d_model % self.text_num_heads == 0, \
            f"d_model({self.d_model}) 必须能被 num_heads({self.text_num_heads}) 整除"
        assert self.image_size % self.patch_size == 0, \
            f"image_size({self.image_size}) 必须能被 patch_size({self.patch_size}) 整除"

    def total_params_estimate(self) -> dict:
        """估算各模块参数量"""
        d = self.d_model
        V = self.vocab_size

        # 词表embedding
        emb_params = V * d  # ~16.4M

        # 文本Transformer层 (每层: MHA + FFN + 2×LN)
        per_layer_mha = 4 * d * d + 4 * d  # Q,K,V,O投影 + bias
        per_layer_ffn = 2 * d * self.text_ff_dim + self.text_ff_dim + d  # 两层FFN
        per_layer_ln = 2 * 2 * d  # 2个LN, 每个有gamma和beta
        text_layers_params = self.text_num_layers * (per_layer_mha + per_layer_ffn + per_layer_ln)

        # 文本最终LN
        text_final_ln = 2 * d

        # 视觉CNN部分
        c_in = self.image_channels
        cnn_params = 0
        for c_out in self.vision_cnn_channels:
            cnn_params += 3 * 3 * c_in * c_out + c_out  # Conv3x3
            cnn_params += c_out * 2  # BatchNorm
            c_in = c_out
        # CNN到d_model投影
        cnn_params += c_in * d + d

        # 视觉ViT部分
        vit_per_layer_mha = 4 * d * d + 4 * d
        vit_per_layer_ffn = 2 * d * self.vision_ff_dim + self.vision_ff_dim + d
        vit_per_layer_ln = 2 * 2 * d
        vision_layers_params = self.vision_num_layers * (vit_per_layer_mha + vit_per_layer_ffn + vit_per_layer_ln)

        # 位置embedding (视觉)
        vision_pos_emb = self.vision_num_patches * d

        # 融合层
        fusion_per_layer = (
            4 * d * d + 4 * d +  # 自注意力
            4 * d * d + 4 * d +  # 交叉注意力
            2 * d * self.fusion_ff_dim + self.fusion_ff_dim + d +  # FFN
            3 * 2 * d  # 3个LN
        )
        fusion_params = self.fusion_num_layers * fusion_per_layer + 2 * d

        # LM头
        lm_head_params = V * d + (d if self.lm_head_bias else 0)
        if self.use_weight_tying:
            lm_head_params = (d if self.lm_head_bias else 0)  # 只算bias

        total = emb_params + text_layers_params + text_final_ln + \
                cnn_params + vision_layers_params + vision_pos_emb + \
                fusion_params + lm_head_params

        return {
            "embedding": emb_params / 1_000_000,
            "text_transformer": text_layers_params / 1_000_000,
            "vision_encoder": (cnn_params + vision_layers_params + vision_pos_emb) / 1_000_000,
            "fusion": fusion_params / 1_000_000,
            "lm_head": lm_head_params / 1_000_000,
            "total_M": total / 1_000_000,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls) -> "StellarConfig":
        """更小的配置 (约20M参数，适合极低配置设备)"""
        return cls(
            d_model=384,
            vocab_size=32000,
            max_seq_len=1024,
            text_num_layers=4,
            text_num_heads=6,
            text_ff_dim=1536,
            image_size=224,
            patch_size=16,
            vision_cnn_channels=(24, 48, 96, 192),
            vision_num_layers=2,
            vision_num_heads=6,
            vision_ff_dim=768,
            fusion_num_layers=1,
            fusion_num_heads=6,  # 与text_num_heads一致，保证RoPE维度匹配
            fusion_ff_dim=1536,
            learning_rate=2e-4,
        )

    @classmethod
    def standard(cls) -> "StellarConfig":
        """标准 0.05B 配置"""
        return cls()
