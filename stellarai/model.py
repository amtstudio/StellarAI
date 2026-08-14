"""
StellarAI 核心模型 - 0.05B 多模态大模型
============================================
架构:
  文本编码器:  6层 Tiny Transformer + RoPE     ≈28M
  视觉编码器:  CNN特征提取 + 4层 ViT           ≈14M
  融合层:     2层 交叉注意力 Transformer       ≈5M
  LM头:       权重绑定 (仅bias)                ≈0.5M
  Embedding:  32k × 512                       ≈16M
  总计:                                        ≈53M ≈0.05B

设计哲学:
  - 优先低配置设备友好 (纯numpy可推理)
  - 模块清晰，便于微调和扩展
  - 保持与PyTorch参数无缝转换的能力
"""

import math
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

from .config import StellarConfig


# ============================================================
#  核心数学工具
# ============================================================

def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU激活函数 (tanh近似)"""
    c = np.float32(0.7978845608028654)  # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish 激活"""
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer Normalization"""
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    return g * x_norm + b


def _rope_pos_encoding(x: np.ndarray, offset: int = 0, base: float = 10000.0) -> np.ndarray:
    """
    旋转位置编码 (RoPE) - 应用于Q或K
    x shape: (..., seq_len, num_heads, head_dim)
    """
    *prefix, seq_len, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim必须是偶数"

    half = head_dim // 2
    positions = np.arange(offset, offset + seq_len, dtype=np.float32).reshape(-1, 1)  # (seq, 1)
    dim_idx = np.arange(0, head_dim, 2, dtype=np.float32) / head_dim  # (half,)
    inv_freq = 1.0 / (base ** dim_idx)  # (half,)
    angles = positions * inv_freq  # (seq, half)

    cos = np.cos(angles).astype(np.float32)  # (seq, half)
    sin = np.sin(angles).astype(np.float32)

    # 适配head维度
    cos = cos[np.newaxis, :, np.newaxis, :]  # (1, seq, 1, half)
    sin = sin[np.newaxis, :, np.newaxis, :]

    x1 = x[..., 0::2]  # 偶数维
    x2 = x[..., 1::2]  # 奇数维

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # 交错合并
    out = np.empty_like(x)
    out[..., 0::2] = out1
    out[..., 1::2] = out2
    return out


# ============================================================
#  Transformer 构建块
# ============================================================

@dataclass
class AttentionWeights:
    """自注意力权重"""
    Wq: np.ndarray  # (d, d)
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    bq: np.ndarray  # (d,)
    bk: np.ndarray
    bv: np.ndarray
    bo: np.ndarray


@dataclass
class FFNWeights:
    """FFN权重"""
    W1: np.ndarray  # (d, ffn)
    b1: np.ndarray
    W2: np.ndarray  # (ffn, d)
    b2: np.ndarray


@dataclass
class TransformerBlockWeights:
    """完整Transformer块权重"""
    attn: AttentionWeights
    ffn: FFNWeights
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray


def _init_attn(d: int) -> AttentionWeights:
    """初始化自注意力权重"""
    scale = 1.0 / math.sqrt(d)
    return AttentionWeights(
        Wq=np.random.randn(d, d).astype(np.float32) * scale,
        Wk=np.random.randn(d, d).astype(np.float32) * scale,
        Wv=np.random.randn(d, d).astype(np.float32) * scale,
        Wo=np.random.randn(d, d).astype(np.float32) * scale,
        bq=np.zeros(d, dtype=np.float32),
        bk=np.zeros(d, dtype=np.float32),
        bv=np.zeros(d, dtype=np.float32),
        bo=np.zeros(d, dtype=np.float32),
    )


def _init_ffn(d: int, ffn: int) -> FFNWeights:
    """初始化FFN权重"""
    scale1 = 1.0 / math.sqrt(d)
    scale2 = 1.0 / math.sqrt(ffn)
    return FFNWeights(
        W1=np.random.randn(d, ffn).astype(np.float32) * scale1,
        b1=np.zeros(ffn, dtype=np.float32),
        W2=np.random.randn(ffn, d).astype(np.float32) * scale2,
        b2=np.zeros(d, dtype=np.float32),
    )


def _init_block(d: int, ffn: int) -> TransformerBlockWeights:
    return TransformerBlockWeights(
        attn=_init_attn(d),
        ffn=_init_ffn(d, ffn),
        ln1_g=np.ones(d, dtype=np.float32),
        ln1_b=np.zeros(d, dtype=np.float32),
        ln2_g=np.ones(d, dtype=np.float32),
        ln2_b=np.zeros(d, dtype=np.float32),
    )


def _multi_head_attn(
    x: np.ndarray,
    aw: AttentionWeights,
    num_heads: int,
    mask: Optional[np.ndarray] = None,
    rope_offset: int = 0,
    rope_base: float = 10000.0,
) -> np.ndarray:
    """
    多头自注意力 (RoPE + 因果mask)
    x: (batch, seq, d)
    Returns: (batch, seq, d)
    """
    B, S, D = x.shape
    head_dim = D // num_heads

    # 线性投影
    Q = x @ aw.Wq + aw.bq  # (B, S, D)
    K = x @ aw.Wk + aw.bk
    V = x @ aw.Wv + aw.bv

    # 多头皮 +  RoPE
    Q = Q.reshape(B, S, num_heads, head_dim)
    K = K.reshape(B, S, num_heads, head_dim)
    V = V.reshape(B, S, num_heads, head_dim)

    Q = _rope_pos_encoding(Q, offset=rope_offset, base=rope_base)
    K = _rope_pos_encoding(K, offset=rope_offset, base=rope_base)

    # (B, H, S, Dh)
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # 注意力分数
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q @ K.transpose(0, 1, 3, 2) * scale  # (B, H, S, S)

    # 因果mask
    if mask is None:
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    scores = np.where(mask, -1e9, scores)

    attn = _softmax(scores, axis=-1)  # (B, H, S, S)
    out = attn @ V  # (B, H, S, Dh)

    # 合并头
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    return out @ aw.Wo + aw.bo


def _transformer_block(
    x: np.ndarray,
    bw: TransformerBlockWeights,
    num_heads: int,
    mask: Optional[np.ndarray] = None,
    rope_offset: int = 0,
    rope_base: float = 10000.0,
    eps: float = 1e-6,
    dropout_p: float = 0.0,
    training: bool = False,
) -> np.ndarray:
    """Pre-LN Transformer 块"""
    # 自注意力子层
    x_norm = _layer_norm(x, bw.ln1_g, bw.ln1_b, eps)
    a = _multi_head_attn(x_norm, bw.attn, num_heads, mask, rope_offset, rope_base)
    if training and dropout_p > 0:
        a = a * (np.random.rand(*a.shape) > dropout_p).astype(np.float32) / (1 - dropout_p)
    x = x + a

    # FFN子层
    x_norm = _layer_norm(x, bw.ln2_g, bw.ln2_b, eps)
    ff = _gelu(x_norm @ bw.ffn.W1 + bw.ffn.b1) @ bw.ffn.W2 + bw.ffn.b2
    if training and dropout_p > 0:
        ff = ff * (np.random.rand(*ff.shape) > dropout_p).astype(np.float32) / (1 - dropout_p)
    return x + ff


# ============================================================
#  交叉注意力 (用于多模态融合)
# ============================================================

@dataclass
class CrossAttnWeights:
    """交叉注意力权重"""
    Wq: np.ndarray  # 对自身
    Wk: np.ndarray  # 对上下文
    Wv: np.ndarray  # 对上下文
    Wo: np.ndarray
    bq: np.ndarray
    bk: np.ndarray
    bv: np.ndarray
    bo: np.ndarray


def _init_cross_attn(d: int) -> CrossAttnWeights:
    scale = 1.0 / math.sqrt(d)
    return CrossAttnWeights(
        Wq=np.random.randn(d, d).astype(np.float32) * scale,
        Wk=np.random.randn(d, d).astype(np.float32) * scale,
        Wv=np.random.randn(d, d).astype(np.float32) * scale,
        Wo=np.random.randn(d, d).astype(np.float32) * scale,
        bq=np.zeros(d, dtype=np.float32),
        bk=np.zeros(d, dtype=np.float32),
        bv=np.zeros(d, dtype=np.float32),
        bo=np.zeros(d, dtype=np.float32),
    )


def _cross_attn(
    x: np.ndarray,
    ctx: np.ndarray,
    cw: CrossAttnWeights,
    num_heads: int,
) -> np.ndarray:
    """
    交叉注意力: x 作为Query, ctx 作为Key/Value
    x: (B, Sx, D)
    ctx: (B, Sc, D)
    """
    B, Sx, D = x.shape
    head_dim = D // num_heads

    Q = (x @ cw.Wq + cw.bq).reshape(B, Sx, num_heads, head_dim).transpose(0, 2, 1, 3)
    Sc = ctx.shape[1]
    K = (ctx @ cw.Wk + cw.bk).reshape(B, Sc, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = (ctx @ cw.Wv + cw.bv).reshape(B, Sc, num_heads, head_dim).transpose(0, 2, 1, 3)

    scale = 1.0 / math.sqrt(head_dim)
    scores = Q @ K.transpose(0, 1, 3, 2) * scale  # (B, H, Sx, Sc)
    attn = _softmax(scores, axis=-1)
    out = attn @ V  # (B, H, Sx, Dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, Sx, D)
    return out @ cw.Wo + cw.bo


@dataclass
class FusionBlockWeights:
    """融合块: 自注意力 + 交叉注意力 + FFN"""
    self_attn: AttentionWeights
    cross_attn: CrossAttnWeights
    ffn: FFNWeights
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    ln3_g: np.ndarray
    ln3_b: np.ndarray


def _init_fusion_block(d: int, ffn: int) -> FusionBlockWeights:
    return FusionBlockWeights(
        self_attn=_init_attn(d),
        cross_attn=_init_cross_attn(d),
        ffn=_init_ffn(d, ffn),
        ln1_g=np.ones(d, dtype=np.float32), ln1_b=np.zeros(d, dtype=np.float32),
        ln2_g=np.ones(d, dtype=np.float32), ln2_b=np.zeros(d, dtype=np.float32),
        ln3_g=np.ones(d, dtype=np.float32), ln3_b=np.zeros(d, dtype=np.float32),
    )


def _fusion_block(
    x: np.ndarray,
    ctx: np.ndarray,
    fw: FusionBlockWeights,
    num_heads: int,
    mask: Optional[np.ndarray] = None,
    rope_offset: int = 0,
    rope_base: float = 10000.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """融合块前向传播"""
    # 自注意力
    x_norm = _layer_norm(x, fw.ln1_g, fw.ln1_b, eps)
    x = x + _multi_head_attn(x_norm, fw.self_attn, num_heads, mask, rope_offset, rope_base)

    # 交叉注意力
    x_norm = _layer_norm(x, fw.ln2_g, fw.ln2_b, eps)
    x = x + _cross_attn(x_norm, ctx, fw.cross_attn, num_heads)

    # FFN
    x_norm = _layer_norm(x, fw.ln3_g, fw.ln3_b, eps)
    ff = _gelu(x_norm @ fw.ffn.W1 + fw.ffn.b1) @ fw.ffn.W2 + fw.ffn.b2
    return x + ff


# ============================================================
#  文本编码器
# ============================================================

@dataclass
class TextEncoderWeights:
    """文本编码器所有权重"""
    wte: np.ndarray           # 词嵌入 (vocab, d)
    blocks: List[TransformerBlockWeights]
    final_ln_g: np.ndarray
    final_ln_b: np.ndarray


def _init_text_encoder(cfg: StellarConfig) -> TextEncoderWeights:
    """初始化文本编码器"""
    # Embedding: 所有token（含特殊token）id 都 < vocab_size (特殊token占用0-127)
    total_vocab = cfg.vocab_size
    scale = 1.0 / math.sqrt(cfg.d_model)
    wte = np.random.randn(total_vocab, cfg.d_model).astype(np.float32) * scale

    blocks = [_init_block(cfg.d_model, cfg.text_ff_dim) for _ in range(cfg.text_num_layers)]

    return TextEncoderWeights(
        wte=wte,
        blocks=blocks,
        final_ln_g=np.ones(cfg.d_model, dtype=np.float32),
        final_ln_b=np.zeros(cfg.d_model, dtype=np.float32),
    )


def _text_encoder_forward(
    input_ids: np.ndarray,
    tw: TextEncoderWeights,
    cfg: StellarConfig,
    mask: Optional[np.ndarray] = None,
    training: bool = False,
) -> np.ndarray:
    """
    文本编码器前向
    input_ids: (B, S)
    Returns: (B, S, d_model)
    """
    B, S = input_ids.shape
    # 截断超出范围的id（保护embedding查找）
    safe_ids = np.clip(input_ids, 0, tw.wte.shape[0] - 1)
    x = tw.wte[safe_ids]  # (B, S, d)

    for i, blk in enumerate(tw.blocks):
        x = _transformer_block(
            x, blk, cfg.text_num_heads, mask,
            rope_offset=0, rope_base=cfg.text_rope_base,
            eps=cfg.layer_norm_eps, dropout_p=cfg.dropout, training=training,
        )
    x = _layer_norm(x, tw.final_ln_g, tw.final_ln_b, cfg.layer_norm_eps)
    return x


# ============================================================
#  视觉编码器 (CNN + ViT 混合)
# ============================================================

@dataclass
class VisionCNNWeights:
    """CNN特征提取部分的权重"""
    convs: List[Tuple[np.ndarray, np.ndarray]]  # (W, b) per layer
    bns: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]  # gamma, beta, mean, var
    proj_W: np.ndarray  # 最后通道 -> d_model
    proj_b: np.ndarray


@dataclass
class VisionEncoderWeights:
    """视觉编码器所有权重"""
    cnn: VisionCNNWeights
    patch_proj: Optional[np.ndarray]   # 直接patch投影的备用路径
    pos_emb: np.ndarray                # (num_patches, d)
    blocks: List[TransformerBlockWeights]
    final_ln_g: np.ndarray
    final_ln_b: np.ndarray


def _init_vision_cnn(cfg: StellarConfig) -> VisionCNNWeights:
    """初始化轻量级CNN"""
    convs = []
    bns = []
    c_in = cfg.image_channels
    for c_out in cfg.vision_cnn_channels:
        fan_in = c_in * 3 * 3
        w = np.random.randn(c_out, c_in, 3, 3).astype(np.float32) * math.sqrt(2.0 / fan_in)
        b = np.zeros(c_out, dtype=np.float32)
        convs.append((w, b))
        # BN
        bns.append((
            np.ones(c_out, dtype=np.float32),
            np.zeros(c_out, dtype=np.float32),
            np.zeros(c_out, dtype=np.float32),
            np.ones(c_out, dtype=np.float32),
        ))
        c_in = c_out

    # 投影到 d_model
    proj_W = np.random.randn(c_in, cfg.d_model).astype(np.float32) * 1.0 / math.sqrt(c_in)
    proj_b = np.zeros(cfg.d_model, dtype=np.float32)

    return VisionCNNWeights(
        convs=convs, bns=bns, proj_W=proj_W, proj_b=proj_b,
    )


def _cnn_forward(imgs: np.ndarray, cnn: VisionCNNWeights, cfg: StellarConfig) -> np.ndarray:
    """
    CNN特征提取 (简化实现，实际训练可切换为PyTorch)
    imgs: (B, 3, H, W)
    Returns: (B, num_patches, d)
    """
    B = imgs.shape[0]
    x = imgs

    for (W, b), (gamma, beta, m, v) in zip(cnn.convs, cnns := cnn.bns):
        # 卷积 (使用im2col for 纯numpy实现，为性能此处只做简化)
        # 简化：stride=2, padding=1
        x = _conv2d(x, W, b, stride=2, padding=1)
        # BN (推理模式)
        x = (x - m.reshape(1, -1, 1, 1)) / np.sqrt(v.reshape(1, -1, 1, 1) + 1e-5)
        x = gamma.reshape(1, -1, 1, 1) * x + beta.reshape(1, -1, 1, 1)
        x = _silu(x)

    # 此时 x shape: (B, C_last, H', W') -> 需为 (num_patches, d)
    # 如果是 224 -> 经4次stride=2: 224/16=14 -> 14x14 patches ✓
    B, C, Hp, Wp = x.shape
    num_p = cfg.vision_num_patches
    if Hp * Wp != num_p:
        # 自适应池化到目标大小
        x = _adaptive_avg_pool(x, int(np.sqrt(num_p)), int(np.sqrt(num_p)))
        Hp = Wp = int(np.sqrt(num_p))

    # (B, C, Hp, Wp) -> (B, Hp*Wp, C)
    features = x.reshape(B, C, Hp * Wp).transpose(0, 2, 1)
    # 投影到 d_model
    features = features @ cnn.proj_W + cnn.proj_b
    return features


def _conv2d(x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """简化版Conv2D (im2col + matmul)，仅用于推理/测试"""
    B, C_in, H, W_in = x.shape
    C_out, _, Kh, Kw = W.shape

    Ho = (H + 2 * padding - Kh) // stride + 1
    Wo = (W_in + 2 * padding - Kw) // stride + 1

    if padding > 0:
        xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        xp = x

    # im2col: 展开每个感受野为一列
    cols = np.zeros((B, C_in * Kh * Kw, Ho * Wo), dtype=x.dtype)
    for i in range(Kh):
        for j in range(Kw):
            patch = xp[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride]
            cols[:, i * Kw * C_in + j * C_in:(i * Kw + j + 1) * C_in, :] = patch.reshape(B, C_in, Ho * Wo)

    # W: (C_out, C_in*Kh*Kw)
    out = W.reshape(C_out, -1) @ cols + b.reshape(-1, 1)
    return out.reshape(B, C_out, Ho, Wo)


def _adaptive_avg_pool(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """自适应平均池化"""
    B, C, H, W = x.shape
    # 对空间维度分块平均
    h_step = H // target_h
    w_step = W // target_w
    result = np.zeros((B, C, target_h, target_w), dtype=x.dtype)
    for i in range(target_h):
        for j in range(target_w):
            h0, h1 = i * h_step, min((i + 1) * h_step, H)
            w0, w1 = j * w_step, min((j + 1) * w_step, W)
            result[:, :, i, j] = np.mean(x[:, :, h0:h1, w0:w1], axis=(2, 3))
    return result


def _init_vision_encoder(cfg: StellarConfig) -> VisionEncoderWeights:
    """初始化视觉编码器"""
    patch_dim = cfg.patch_size * cfg.patch_size * cfg.image_channels
    patch_proj = np.random.randn(patch_dim, cfg.d_model).astype(np.float32) * 1.0 / math.sqrt(patch_dim)

    scale = 1.0 / math.sqrt(cfg.d_model)
    pos_emb = np.random.randn(cfg.vision_num_patches, cfg.d_model).astype(np.float32) * scale

    blocks = [_init_block(cfg.d_model, cfg.vision_ff_dim) for _ in range(cfg.vision_num_layers)]

    return VisionEncoderWeights(
        cnn=_init_vision_cnn(cfg),
        patch_proj=patch_proj,
        pos_emb=pos_emb,
        blocks=blocks,
        final_ln_g=np.ones(cfg.d_model, dtype=np.float32),
        final_ln_b=np.zeros(cfg.d_model, dtype=np.float32),
    )


def _vision_encoder_forward(
    images: np.ndarray,
    patches: Optional[np.ndarray],
    vw: VisionEncoderWeights,
    cfg: StellarConfig,
) -> np.ndarray:
    """
    视觉编码器前向
    images: (B, 3, H, W) 或 None
    patches: (B, num_patches, patch_dim) 或 None (备用路径)
    Returns: (B, num_patches, d_model)
    """
    # 优先使用CNN路径
    if images is not None:
        features = _cnn_forward(images, vw.cnn, cfg)  # (B, Np, d)
    elif patches is not None:
        features = patches @ vw.patch_proj  # (B, Np, d)
    else:
        raise ValueError("images或patches必须提供其一")

    B, Np, D = features.shape
    Np_target = cfg.vision_num_patches
    if Np != Np_target:
        # 插值或截断
        if Np > Np_target:
            features = features[:, :Np_target, :]
        else:
            pad = np.zeros((B, Np_target - Np, D), dtype=features.dtype)
            features = np.concatenate([features, pad], axis=1)
        Np = Np_target

    # 加位置embedding
    features = features + vw.pos_emb[np.newaxis, :Np, :]

    # ViT层 (无RoPE，因为用了绝对位置embedding + 无因果mask)
    no_mask = np.zeros((Np, Np), dtype=bool)
    for blk in vw.blocks:
        features = _transformer_block(
            features, blk, cfg.vision_num_heads, mask=no_mask,
            rope_offset=0, rope_base=10000.0,  # Vision不用RoPE
            eps=cfg.layer_norm_eps, dropout_p=cfg.dropout, training=False,
        )
    features = _layer_norm(features, vw.final_ln_g, vw.final_ln_b, cfg.layer_norm_eps)
    return features


# ============================================================
#  LM 头 + 完整模型
# ============================================================

@dataclass
class StellarAIWeights:
    """StellarAI完整权重"""
    text: TextEncoderWeights
    vision: VisionEncoderWeights
    fusion: List[FusionBlockWeights]
    fusion_ln_g: np.ndarray
    fusion_ln_b: np.ndarray
    lm_bias: Optional[np.ndarray]  # 权重绑定时，只有bias


def _init_weights(cfg: StellarConfig) -> StellarAIWeights:
    """初始化所有模型权重"""
    fusion = [_init_fusion_block(cfg.d_model, cfg.fusion_ff_dim)
              for _ in range(cfg.fusion_num_layers)]
    lm_bias = np.zeros(cfg.vocab_size, dtype=np.float32) if cfg.lm_head_bias else None
    return StellarAIWeights(
        text=_init_text_encoder(cfg),
        vision=_init_vision_encoder(cfg),
        fusion=fusion,
        fusion_ln_g=np.ones(cfg.d_model, dtype=np.float32),
        fusion_ln_b=np.zeros(cfg.d_model, dtype=np.float32),
        lm_bias=lm_bias,
    )


# ============================================================
#  StellarAI 主类
# ============================================================

class StellarAI:
    """
    StellarAI 0.05B 多模态大模型 (纯NumPy实现)
    可处理: 纯文本, 图像+文本, 多图+文本

    使用方式:
        model = StellarAI.from_pretrained("path/to/weights")  或  StellarAI.random_init()
        output = model.generate(text="你好，介绍一下这张图片", image="cat.jpg")
    """

    def __init__(self, cfg: StellarConfig, weights: Optional[StellarAIWeights] = None, seed: int = 42):
        self.cfg = cfg
        self._weights = weights
        if weights is None:
            np.random.seed(seed)
            self._weights = _init_weights(cfg)

    # ---------- 构造方法 ----------
    @classmethod
    def random_init(cls, cfg: Optional[StellarConfig] = None, seed: int = 42) -> "StellarAI":
        """随机初始化"""
        cfg = cfg or StellarConfig.standard()
        return cls(cfg, seed=seed)

    @classmethod
    def tiny_init(cls, seed: int = 42) -> "StellarAI":
        """更小的版本 (~20M)"""
        return cls.random_init(StellarConfig.tiny(), seed=seed)

    # ---------- 参数统计 ----------
    def count_params(self) -> Dict[str, float]:
        """统计各模块参数量 (M)"""
        estimate = self.cfg.total_params_estimate()
        # 补充实际统计
        def _arrays_total(objs):
            total = 0
            for obj in objs:
                if isinstance(obj, np.ndarray):
                    total += obj.size
                elif isinstance(obj, (list, tuple)):
                    total += _arrays_total(obj)
                elif hasattr(obj, "__dict__"):
                    total += _arrays_total(vars(obj).values())
            return total

        w = self._weights
        actual = _arrays_total([w.text, w.vision, w.fusion, w.fusion_ln_g, w.fusion_ln_b, w.lm_bias]) / 1_000_000
        estimate["actual_total_M"] = round(actual, 2)
        return estimate

    # ---------- 核心前向 ----------
    def forward(
        self,
        input_ids: np.ndarray,
        images: Optional[np.ndarray] = None,
        patches: Optional[np.ndarray] = None,
        image_placeholders: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        完整前向传播
        Args:
            input_ids: (B, S) 文本token ids，其中 [IMG] token 位置由image_placeholders指定
            images: (B, 3, H, W) or (B, N_img, 3, H, W)
            patches: (B, N_img, N_p, patch_dim) 替代images
            image_placeholders: (B, N_img, 2) 每个图像的 [start_pos, end_pos) 位置
        Returns:
            logits: (B, S, vocab_size+2)
        """
        B, S = input_ids.shape

        # ===== 1. 文本编码 =====
        text_hidden = _text_encoder_forward(input_ids, self._weights.text, self.cfg)  # (B, S, d)

        # ===== 2. 视觉编码 (如果有图像) =====
        vision_hidden = None
        if images is not None or patches is not None:
            # 处理多张图: 压平到 batch 维再编码
            if images is not None:
                if images.ndim == 5:  # (B, N_img, 3, H, W)
                    B_imgs, N_imgs, C, H, W = images.shape
                    imgs_flat = images.reshape(B_imgs * N_imgs, C, H, W)
                    vh = _vision_encoder_forward(imgs_flat, None, self._weights.vision, self.cfg)
                    vision_hidden = vh.reshape(B_imgs, N_imgs, self.cfg.vision_num_patches, self.cfg.d_model)
                else:
                    vh = _vision_encoder_forward(images, None, self._weights.vision, self.cfg)
                    vision_hidden = vh[:, np.newaxis, :, :]  # (B, 1, Np, d)
            elif patches is not None:
                if patches.ndim == 4:  # (B, N_img, Np, pd)
                    B_imgs, N_imgs, Np, Pd = patches.shape
                    pf = patches.reshape(B_imgs * N_imgs, Np, Pd)
                    vh = _vision_encoder_forward(None, pf, self._weights.vision, self.cfg)
                    vision_hidden = vh.reshape(B_imgs, N_imgs, self.cfg.vision_num_patches, self.cfg.d_model)
                else:
                    vh = _vision_encoder_forward(None, patches, self._weights.vision, self.cfg)
                    vision_hidden = vh[:, np.newaxis, :, :]

            # ===== 将视觉特征注入到对应占位符位置 =====
            if vision_hidden is not None and image_placeholders is not None:
                for b in range(B):
                    n_img = image_placeholders.shape[1] if image_placeholders.ndim == 3 else 1
                    for i in range(n_img):
                        ph = image_placeholders[b, i] if image_placeholders.ndim == 3 else image_placeholders[b]
                        s, e = int(ph[0]), int(ph[1])
                        if s >= e:
                            continue
                        n_tokens = min(e - s, self.cfg.vision_num_patches)
                        v_features = vision_hidden[b, min(i, vision_hidden.shape[1] - 1), :n_tokens, :]
                        text_hidden[b, s:s + n_tokens, :] = v_features

        # ===== 3. 多模态融合 =====
        # 把视觉拼接成额外上下文（也用于没有占位符注入的情况）
        ctx = None
        if vision_hidden is not None:
            ctx_list = []
            for b in range(vision_hidden.shape[0]):
                all_patches = vision_hidden[b].reshape(-1, self.cfg.d_model)
                ctx_list.append(all_patches)
            # padding 到统一长度
            max_ctx = max(c.shape[0] for c in ctx_list)
            ctx = np.zeros((B, max_ctx, self.cfg.d_model), dtype=np.float32)
            for b, c in enumerate(ctx_list):
                ctx[b, :c.shape[0], :] = c

        # 融合层
        mask = np.triu(np.ones((S, S), dtype=bool), k=1)
        hidden = text_hidden
        for fblk in self._weights.fusion:
            if ctx is not None:
                hidden = _fusion_block(
                    hidden, ctx, fblk, self.cfg.fusion_num_heads,
                    mask=mask, rope_offset=0, rope_base=self.cfg.text_rope_base,
                    eps=self.cfg.layer_norm_eps,
                )
            else:
                # 退化：仅自注意力 + FFN（等价普通Transformer块）
                x_norm = _layer_norm(hidden, fblk.ln1_g, fblk.ln1_b, self.cfg.layer_norm_eps)
                a = _multi_head_attn(x_norm, fblk.self_attn, self.cfg.fusion_num_heads, mask, 0, self.cfg.text_rope_base)
                hidden = hidden + a
                x_norm = _layer_norm(hidden, fblk.ln3_g, fblk.ln3_b, self.cfg.layer_norm_eps)
                hidden = hidden + _gelu(x_norm @ fblk.ffn.W1 + fblk.ffn.b1) @ fblk.ffn.W2 + fblk.ffn.b2

        hidden = _layer_norm(hidden, self._weights.fusion_ln_g, self._weights.fusion_ln_b, self.cfg.layer_norm_eps)

        # ===== 4. LM头 =====
        # 权重绑定: 复用text embedding矩阵
        lm_W = self._weights.text.wte  # (V+2, d)
        logits = hidden @ lm_W.T  # (B, S, V+2)
        if self._weights.lm_bias is not None:
            logits = logits + self._weights.lm_bias[np.newaxis, np.newaxis, :]
        return logits

    # ---------- 生成 ----------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        image: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        vision_preprocessor: Optional[Any] = None,
        verbose: bool = False,
    ) -> str:
        """
        自回归生成
        Args:
            prompt: 提示文本
            max_new_tokens: 最大新生成token数
            temperature: 采样温度
            top_k: Top-K采样
            image: 图像(路径/数组/PIL), 支持 str/np.ndarray/PIL.Image
            tokenizer: 分词器 (SimpleTokenizer), 不传则使用默认SimpleTokenizer
            vision_preprocessor: 视觉预处理器, 不传则默认构造
        """
        from .tokenizer import SimpleTokenizer
        from .vision import VisionPreprocessor

        if tokenizer is None:
            tokenizer = SimpleTokenizer(self.cfg.vocab_size)
        if vision_preprocessor is None:
            vision_preprocessor = VisionPreprocessor(self.cfg.image_size, self.cfg.patch_size)

        # ===== 构造输入序列 =====
        # 格式: [BOS] [IMG占位符N个] prompt [SEP]
        input_tokens = [tokenizer.bos_id]

        img_tensor = None
        ph_start = -1
        if image is not None:
            ph_start = len(input_tokens)
            # 图像占位符: 每个patch用一个 [IMG] token
            n_vis_tokens = self.cfg.vision_num_patches
            for _ in range(n_vis_tokens):
                input_tokens.append(self.cfg.vision_token_id)
            input_tokens.append(self.cfg.sep_token_id)

            # 预处理图像
            img_tensor = vision_preprocessor.preprocess(image)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor[np.newaxis, ...]  # (1, 3, H, W)

        # 文本部分
        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False,
                                       max_length=self.cfg.max_seq_len - len(input_tokens) - 1)
        input_tokens.extend(prompt_ids)
        input_tokens.append(self.cfg.sep_token_id)

        # 转numpy
        input_ids = np.array([input_tokens], dtype=np.int64)

        # 图像占位符位置
        placeholders = None
        if ph_start > 0:
            placeholders = np.array([[ph_start, ph_start + self.cfg.vision_num_patches]])
            placeholders = placeholders[:, np.newaxis, :]  # (B=1, N_img=1, 2)

        # ===== 自回归生成 =====
        for step in range(max_new_tokens):
            if input_ids.shape[1] >= self.cfg.max_seq_len:
                break

            logits = self.forward(
                input_ids,
                images=img_tensor,
                image_placeholders=placeholders,
            )  # (B, S, V+2)

            # 取最后一个位置
            next_logits = logits[0, -1, :self.cfg.vocab_size]  # (V,)

            # Top-K 采样
            if temperature <= 0:
                next_id = int(np.argmax(next_logits))
            else:
                next_logits = next_logits / max(temperature, 1e-4)
                # Top-K
                if top_k > 0:
                    k = min(top_k, next_logits.shape[0])
                    top_k_ids = np.argpartition(-next_logits, k - 1)[:k]
                    top_k_logits = next_logits[top_k_ids]
                    probs = _softmax(top_k_logits)
                    next_id = int(np.random.choice(top_k_ids, p=probs))
                else:
                    probs = _softmax(next_logits)
                    next_id = int(np.random.choice(next_logits.shape[0], p=probs))

            if verbose:
                char = tokenizer.decode([next_id], skip_special=False)
                print(char, end="", flush=True)

            # EOS终止
            if next_id == tokenizer.eos_id:
                break

            input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)

        # 解码生成部分 (从prompt后开始)
        start_decode = len(input_tokens)
        gen_ids = input_ids[0, start_decode:].tolist()
        return tokenizer.decode(gen_ids, skip_special=True)

    # ---------- 保存/加载 ----------
    def save_weights(self, path: str):
        """保存权重到 .npz 文件"""
        import os
        data = {}

        def _flatten(prefix, obj):
            if isinstance(obj, np.ndarray):
                data[prefix] = obj
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _flatten(f"{prefix}.{i}", item)
            elif hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    _flatten(f"{prefix}.{k}", v)

        _flatten("w", self._weights)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez_compressed(path, **data)

    @classmethod
    def load_weights(cls, path: str, cfg: Optional[StellarConfig] = None) -> "StellarAI":
        """从 .npz 文件加载权重"""
        data = np.load(path)
        if cfg is None:
            cfg = StellarConfig.standard()

        # 重建权重结构
        w = _init_weights(cfg)

        def _assign(obj, prefix):
            if isinstance(obj, np.ndarray):
                if prefix in data:
                    loaded = data[prefix]
                    if loaded.shape == obj.shape:
                        np.copyto(obj, loaded)
                return
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    _assign(item, f"{prefix}.{i}")
                return
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    _assign(v, f"{prefix}.{k}")
                return

        _assign(w, "w")
        return cls(cfg, weights=w)

    def save_safetensors(self, path: str):
        """保存为 safetensors 格式 (如果已安装)"""
        try:
            from safetensors import safe_open
            from safetensors.numpy import save_file
        except ImportError:
            # 降级为npz
            if not path.endswith(".npz"):
                path += ".npz"
            return self.save_weights(path)

        tensors = {}

        def _flatten(prefix, obj):
            if isinstance(obj, np.ndarray):
                tensors[prefix] = obj
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _flatten(f"{prefix}.{i}", item)
            elif hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    _flatten(f"{prefix}.{k}", v)

        _flatten("w", self._weights)
        save_file(tensors, path)

    def __repr__(self) -> str:
        params = self.count_params()
        return (f"StellarAI(d_model={self.cfg.d_model}, "
                f"text_layers={self.cfg.text_num_layers}, "
                f"vision_layers={self.cfg.vision_num_layers}, "
                f"fusion_layers={self.cfg.fusion_num_layers}, "
                f"params≈{params.get('actual_total_M', params['total_M']):.1f}M)")
