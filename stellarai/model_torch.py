"""
StellarAI PyTorch版模型 - 用于实际训练
=======================================
与NumPy版架构完全一致，但利用PyTorch自动微分实现高效训练
支持CPU和GPU训练
"""

import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StellarConfig


# ============================================================
#  RoPE 旋转位置编码 (PyTorch版)
# ============================================================

class RoPE(nn.Module):
    """旋转位置编码"""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        # 预计算 cos/sin
        half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)  # (max_seq, half)
        # 扩展到完整head_dim
        cos = torch.zeros(max_seq_len, head_dim)
        sin = torch.zeros(max_seq_len, head_dim)
        cos[:, 0::2] = torch.cos(angles)
        cos[:, 1::2] = torch.cos(angles)
        sin[:, 0::2] = torch.sin(angles)
        sin[:, 1::2] = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, H, S, Dh) 或 (B, S, H, Dh)"""
        *_, S, Dh = x.shape
        cos = self.cos[offset:offset + S]  # (S, Dh)
        sin = self.sin[offset:offset + S]
        # 旋转: x1=偶数维, x2=奇数维
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # 广播: (S, Dh/2) -> (1, 1, S, Dh/2) 或 (1, S, 1, Dh/2)
        cos_half = cos[:, 0::2]
        sin_half = sin[:, 0::2]
        while cos_half.dim() < x1.dim():
            cos_half = cos_half.unsqueeze(0)
            sin_half = sin_half.unsqueeze(0)
        out1 = x1 * cos_half - x2 * sin_half
        out2 = x1 * sin_half + x2 * cos_half
        result = torch.empty_like(x)
        result[..., 0::2] = out1
        result[..., 1::2] = out2
        return result


# ============================================================
#  Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力 + RoPE + 因果mask"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, rope: Optional[RoPE] = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Dict] = None) -> torch.Tensor:
        """
        x: (B, S, D)
        mask: (S, S) 因果mask, True=屏蔽
        kv_cache: 可选的KV缓存 (用于推理加速)
        """
        B, S, D = x.shape

        Q = self.Wq(x)  # (B, S, D)
        K = self.Wk(x)
        V = self.Wv(x)

        # 分头: (B, S, H, Dh) -> (B, H, S, Dh)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        if self.rope is not None:
            offset = 0
            if kv_cache is not None and "K" in kv_cache:
                offset = kv_cache["K"].shape[2]
            Q = self.rope(Q.transpose(1, 2), offset=offset).transpose(1, 2)
            K = self.rope(K.transpose(1, 2), offset=offset).transpose(1, 2)

        # KV缓存
        if kv_cache is not None:
            if "K" in kv_cache:
                K = torch.cat([kv_cache["K"], K], dim=2)
                V = torch.cat([kv_cache["V"], V], dim=2)
            kv_cache["K"] = K
            kv_cache["V"] = V

        # 注意力分数: (B, H, Sq, Sk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 因果mask
        Sq = Q.shape[2]
        Sk = K.shape[2]
        if mask is not None:
            scores = scores.masked_fill(mask[:Sq, :Sk], float('-inf'))
        else:
            # 默认因果mask
            causal = torch.triu(torch.ones(Sq, Sk, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, S, Dh)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.Wo(out)


class CrossAttention(nn.Module):
    """交叉注意力: x作为Query, ctx作为Key/Value"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, Sx, D = x.shape
        Sc = ctx.shape[1]

        Q = self.Wq(x).view(B, Sx, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(ctx).view(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(ctx).view(B, Sc, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Sx, D)
        return self.Wo(out)


# ============================================================
#  Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """Pre-LN Transformer块: 自注意力 + FFN"""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, rope: Optional[RoPE] = None, eps: float = 1e-6):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, rope)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class FusionBlock(nn.Module):
    """融合块: 自注意力 + 交叉注意力 + FFN"""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, rope: Optional[RoPE] = None, eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, rope)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.ln3 = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.ln1(x), mask))
        if ctx is not None:
            x = x + self.dropout(self.cross_attn(self.ln2(x), ctx))
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


# ============================================================
#  视觉编码器
# ============================================================

class VisionEncoder(nn.Module):
    """轻量级CNN + ViT混合视觉编码器"""

    def __init__(self, cfg: StellarConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # CNN特征提取
        layers = []
        c_in = cfg.image_channels
        for c_out in cfg.vision_cnn_channels:
            layers.append(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.SiLU())
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        self.cnn_proj = nn.Linear(c_in, D)

        # 位置embedding
        self.pos_emb = nn.Parameter(torch.randn(cfg.vision_num_patches, D) * 0.02)

        # ViT层
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.vision_num_heads, cfg.vision_ff_dim,
                             cfg.dropout, rope=None, eps=cfg.layer_norm_eps)
            for _ in range(cfg.vision_num_layers)
        ])
        self.final_ln = nn.LayerNorm(D, eps=cfg.layer_norm_eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> (B, num_patches, D)"""
        x = self.cnn(images)  # (B, C, H', W')
        B, C, Hp, Wp = x.shape
        # 自适应池化到目标patch数
        target = int(math.sqrt(self.cfg.vision_num_patches))
        if Hp != target:
            x = F.adaptive_avg_pool2d(x, (target, target))
            Hp = Wp = target
        # (B, C, H, W) -> (B, H*W, C) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        x = self.cnn_proj(x)
        x = x + self.pos_emb.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, mask=None)  # 视觉无因果mask
        return self.final_ln(x)


# ============================================================
#  StellarAI 完整模型 (PyTorch)
# ============================================================

class StellarTorch(nn.Module):
    """
    StellarAI PyTorch版 - 可训练的多模态大模型
    架构与NumPy版完全一致
    """

    def __init__(self, cfg: StellarConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # 词嵌入 (权重绑定到LM头)
        self.wte = nn.Embedding(cfg.vocab_size, D)

        # RoPE
        self.text_rope = RoPE(D // cfg.text_num_heads, cfg.max_seq_len, cfg.text_rope_base)

        # 文本编码器
        self.text_blocks = nn.ModuleList([
            TransformerBlock(D, cfg.text_num_heads, cfg.text_ff_dim,
                             cfg.dropout, rope=self.text_rope, eps=cfg.layer_norm_eps)
            for _ in range(cfg.text_num_layers)
        ])
        self.text_final_ln = nn.LayerNorm(D, eps=cfg.layer_norm_eps)

        # 视觉编码器
        self.vision_encoder = VisionEncoder(cfg)

        # 融合层
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(D, cfg.fusion_num_heads, cfg.fusion_ff_dim,
                        cfg.dropout, rope=self.text_rope, eps=cfg.layer_norm_eps)
            for _ in range(cfg.fusion_num_layers)
        ])
        self.fusion_final_ln = nn.LayerNorm(D, eps=cfg.layer_norm_eps)

        # LM头 (权重绑定)
        self.lm_bias = nn.Parameter(torch.zeros(cfg.vocab_size)) if cfg.lm_head_bias else None

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_positions: Optional[List] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: (B, S)
            images: (B, 3, H, W) 或 None
            image_positions: [(start, end), ...] 图像token在序列中的位置
            labels: (B, S) next-token prediction标签, -100忽略
        Returns:
            dict: {logits, loss (如果提供labels)}
        """
        B, S = input_ids.shape
        D = self.cfg.d_model

        # 1. 词嵌入
        x = self.wte(input_ids)  # (B, S, D)

        # 2. 视觉编码 + 注入
        vision_features = None
        if images is not None:
            vision_features = self.vision_encoder(images)  # (B, Np, D)
            if image_positions is not None:
                for b in range(B):
                    if b < len(image_positions):
                        s, e = image_positions[b]
                        n = min(e - s, vision_features.shape[1])
                        x[b, s:s + n] = vision_features[b, :n]

        # 3. 文本Transformer
        for blk in self.text_blocks:
            x = blk(x)  # 默认因果mask

        # 4. 融合层 (如果有视觉特征，做交叉注意力)
        ctx = vision_features
        for fblk in self.fusion_blocks:
            x = fblk(x, ctx)

        x = self.fusion_final_ln(x)

        # 5. LM头 (权重绑定)
        logits = F.linear(x, self.wte.weight)  # (B, S, V)
        if self.lm_bias is not None:
            logits = logits + self.lm_bias

        # 6. 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        images: Optional[torch.Tensor] = None,
        image_positions: Optional[List] = None,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= self.cfg.max_seq_len:
                break
            out = self.forward(input_ids, images=images, image_positions=image_positions)
            logits = out["logits"][:, -1, :self.cfg.vocab_size]  # (B, V)

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return input_ids

    def count_params(self) -> Dict[str, float]:
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M": round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }
