"""
StellarAI 训练框架
- 优化器 (AdamW)
- 学习率调度器 (余弦退火 + Warmup)
- 数据加载器 (文本/图文对)
- 训练循环 (纯NumPy版 + 可选PyTorch加速版)
"""

import math
import os
import time
import json
from typing import Optional, List, Tuple, Dict, Any, Iterator
from dataclasses import dataclass

import numpy as np

from .config import StellarConfig
from .model import StellarAI, StellarAIWeights
from .tokenizer import SimpleTokenizer


# ============================================================
#  AdamW 优化器 (纯NumPy实现)
# ============================================================

class AdamW:
    """
    AdamW优化器 - 解耦权重衰减
    """

    def __init__(
        self,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(
        self,
        weights_obj: Any,
        grads: Dict[str, np.ndarray],
        lr: Optional[float] = None,
    ) -> None:
        """
        执行一步优化
        weights_obj: 带有 .__dict__ 的权重对象
        grads: {param_path: gradient_array}
        """
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        def _apply(obj, prefix: str):
            if isinstance(obj, np.ndarray):
                if prefix in grads:
                    g = grads[prefix]
                    if prefix not in self.m:
                        self.m[prefix] = np.zeros_like(obj)
                        self.v[prefix] = np.zeros_like(obj)

                    # AdamW: 先权重衰减，再动量
                    obj -= current_lr * self.weight_decay * obj

                    self.m[prefix] = self.beta1 * self.m[prefix] + (1 - self.beta1) * g
                    self.v[prefix] = self.beta2 * self.v[prefix] + (1 - self.beta2) * (g * g)

                    m_hat = self.m[prefix] / bias_correction1
                    v_hat = self.v[prefix] / bias_correction2

                    update = current_lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    obj -= update
                return
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    _apply(item, f"{prefix}.{i}")
                return
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    _apply(v, f"{prefix}.{k}")
                return

        _apply(weights_obj, "w")


# ============================================================
#  学习率调度器
# ============================================================

def get_lr_cosine_warmup(step: int, warmup_steps: int, max_steps: int,
                          peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """余弦退火 + 线性Warmup 学习率"""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    # 余弦退火
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
#  损失函数
# ============================================================

def cross_entropy_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    ignore_index: int = -100,
) -> Tuple[float, np.ndarray]:
    """
    交叉熵损失 (next-token prediction)
    logits: (B, S, V)
    labels: (B, S)  与logits对应，-100表示忽略
    Returns:
        loss: 标量平均损失
        grad_logits: 与logits形状相同的梯度
    """
    B, S, V = logits.shape

    # 数值稳定的log_softmax
    x_max = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - x_max
    exp_sum = np.sum(np.exp(shifted), axis=-1, keepdims=True)
    log_softmax = shifted - np.log(exp_sum)  # (B, S, V)

    # 标签转为one-hot索引
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    n_valid = max(np.sum(valid), 1)

    # 取正确类的log prob
    flat_log_probs = log_softmax.reshape(-1, V)
    idx = np.arange(len(flat_labels))
    # 安全索引
    safe_labels = np.where(valid, flat_labels, 0)
    correct_lps = flat_log_probs[idx, safe_labels]
    correct_lps = np.where(valid, correct_lps, 0.0)
    loss = -np.sum(correct_lps) / n_valid

    # 梯度
    probs = np.exp(log_softmax)  # softmax
    one_hot = np.zeros_like(probs)
    # (B, S) -> (B, S, V) one-hot
    safe_labs = np.where(labels == ignore_index, 0, labels)
    np.put_along_axis(one_hot, safe_labs[..., np.newaxis], 1.0, axis=-1)
    # dL/dlogits = (p - y) / n   对有效token除以n_valid
    grad = (probs - one_hot) / n_valid
    # 无效位置梯度清零
    grad_mask = (labels != ignore_index).astype(np.float32)[:, :, np.newaxis]
    grad = grad * grad_mask

    return loss, grad


# ============================================================
#  数据集 / 数据加载器
# ============================================================

@dataclass
class TextDataset:
    """纯文本数据集 (用于预训练/微调)"""
    samples: List[List[int]]   # 每个样本是 token ids 列表

    @classmethod
    def from_text_file(cls, path: str, tokenizer: SimpleTokenizer,
                       max_length: int = 512, max_samples: int = -1) -> "TextDataset":
        """从文本文件加载，每行一个样本"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_bos=True, add_eos=True, max_length=max_length)
                samples.append(ids)
                if 0 < max_samples <= len(samples):
                    break
        return cls(samples=samples)

    @classmethod
    def from_jsonl(cls, path: str, tokenizer: SimpleTokenizer,
                   max_length: int = 512, text_key: str = "text",
                   max_samples: int = -1) -> "TextDataset":
        """从JSONL文件加载"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get(text_key, "")
                    if not text:
                        continue
                    ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_length=max_length)
                    samples.append(ids)
                    if 0 < max_samples <= len(samples):
                        break
                except json.JSONDecodeError:
                    continue
        return cls(samples=samples)


@dataclass
class MultimodalSample:
    """多模态样本"""
    text_ids: List[int]           # 文本token ids (包含IMG占位符)
    image_paths: List[str]        # 对应图像路径
    img_placeholder_ranges: List[Tuple[int, int]]  # [(s,e), ...]


@dataclass
class MultimodalDataset:
    """图文对数据集"""
    samples: List[MultimodalSample]

    @classmethod
    def from_jsonl(cls, path: str, tokenizer: SimpleTokenizer,
                   cfg: StellarConfig,
                   max_length: int = 512, max_samples: int = -1) -> "MultimodalDataset":
        """
        从JSONL加载，格式: {"text": "...", "image": "path.jpg", "images": [...]}
        文本中的 <image> 标记会被替换为 IMG 占位符
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    imgs = data.get("images") or ([data["image"]] if "image" in data else [])
                    if not text:
                        continue

                    # 替换<image>为占位符tokens
                    text_parts = text.split("<image>")
                    n_placeholders = len(text_parts) - 1
                    if n_placeholders < len(imgs):
                        # 多余的图放开头
                        prepend = "[IMG]" * (len(imgs) - n_placeholders)
                        text_parts[0] = prepend + text_parts[0]
                        n_placeholders = len(imgs)

                    # 构建token序列
                    ids = [tokenizer.bos_id]
                    ranges = []
                    part_tokens_list = []
                    for part in text_parts:
                        part_tokens_list.append(tokenizer.encode(part, add_bos=False, add_eos=False,
                                                                  max_length=max_length))
                    n_vis_tokens = cfg.vision_num_patches
                    for i, pt in enumerate(part_tokens_list):
                        if i > 0:
                            # 插入IMG占位符
                            s = len(ids)
                            for _ in range(n_vis_tokens):
                                ids.append(cfg.vision_token_id)
                            e = len(ids)
                            ranges.append((s, e))
                        ids.extend(pt)
                        if len(ids) >= max_length:
                            break
                    ids.append(tokenizer.eos_id)
                    ids = ids[:max_length]

                    samples.append(MultimodalSample(
                        text_ids=ids,
                        image_paths=imgs[:len(ranges)] if ranges else [],
                        img_placeholder_ranges=ranges,
                    ))
                    if 0 < max_samples <= len(samples):
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        return cls(samples=samples)


# ============================================================
#  数据批处理
# ============================================================

class DataLoader:
    """简单的数据加载器"""

    def __init__(self, data, batch_size: int = 8, shuffle: bool = True, seed: int = 42):
        """data: TextDataset 或 MultimodalDataset"""
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        n = len(self.data.samples)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        idx = np.arange(len(self.data.samples))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self._collate(batch_idx)

    def _collate(self, indices):
        """将样本组装成Batch"""
        samples = [self.data.samples[i] for i in indices]
        if isinstance(self.data, TextDataset):
            return self._collate_text(samples)
        else:
            return self._collate_multimodal(samples)

    def _collate_text(self, samples: List[List[int]]) -> Dict[str, np.ndarray]:
        """组装文本批 (next-token prediction)"""
        max_len = max(len(s) for s in samples)
        batch_size = len(samples)

        input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        labels = np.full((batch_size, max_len), -100, dtype=np.int64)

        for i, s in enumerate(samples):
            L = len(s)
            input_ids[i, :L] = s
            # 预测下一个token: labels[..., :-1] 对应 input_ids[..., 1:]
            if L > 1:
                labels[i, :L - 1] = s[1:]

        return {"input_ids": input_ids, "labels": labels}

    def _collate_multimodal(self, samples: List[MultimodalSample]) -> Dict[str, Any]:
        """组装多模态批"""
        max_len = max(len(s.text_ids) for s in samples)
        B = len(samples)

        input_ids = np.zeros((B, max_len), dtype=np.int64)
        labels = np.full((B, max_len), -100, dtype=np.int64)
        placeholders = np.zeros((B, 8, 2), dtype=np.int64)  # 最多8张图
        has_images = any(len(s.image_paths) > 0 for s in samples)

        for i, s in enumerate(samples):
            L = len(s.text_ids)
            input_ids[i, :L] = s.text_ids
            if L > 1:
                labels[i, :L - 1] = s.text_ids[1:]
            for j, (s_pos, e_pos) in enumerate(s.img_placeholder_ranges):
                if j < 8:
                    placeholders[i, j, 0] = s_pos
                    placeholders[i, j, 1] = e_pos

        batch = {"input_ids": input_ids, "labels": labels}
        if has_images:
            # 留给调用者去加载图像 (避免数据加载器依赖vision模块)
            batch["_samples"] = samples
            batch["placeholders"] = placeholders
        return batch


# ============================================================
#  自动微分 (极简实现 - 用于训练)
# ============================================================

def train_step(
    model: StellarAI,
    batch: Dict[str, np.ndarray],
    optimizer: AdamW,
    lr: float,
) -> float:
    """
    一步训练 (仅文本，不包含视觉 - 视觉训练建议用PyTorch版本)
    这是一个"教育版"训练循环，展示如何通过数值微分或手动反向传播训练模型。
    由于纯NumPy实现完整反向传播过于庞大，这里仅演示结构。
    实际训练请使用 train_pytorch.py 中的PyTorch版本。
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # 前向
    logits = model.forward(input_ids)  # (B, S, V)

    # 损失
    loss, grad_logits = cross_entropy_loss(logits, labels)

    # 注: 完整反向传播在此省略 (纯NumPy实现50M参数模型的完整ad需要数千行代码)
    # 生产环境: 请使用 stellarai/train_pytorch.py 版本
    # 这里我们记录损失，但不更新参数（因为没有完整的ad引擎）
    return float(loss)


# ============================================================
#  训练入口 (配置化)
# ============================================================

@dataclass
class TrainingArgs:
    """训练参数"""
    output_dir: str = "outputs/stellar_checkpoint"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_steps: int = -1
    save_every: int = 500
    log_every: int = 10
    grad_clip: float = 1.0
    seed: int = 42


def train_text_only(
    model: StellarAI,
    dataset: TextDataset,
    tokenizer: SimpleTokenizer,
    args: TrainingArgs,
) -> Dict[str, Any]:
    """
    纯文本预训练入口（演示版）
    实际训练建议使用PyTorch版本: stellarai/train_pytorch.py
    """
    cfg = model.cfg
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    optimizer = AdamW(lr=args.learning_rate, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)

    # 估算总步数
    steps_per_epoch = len(loader)
    total_steps = args.max_steps if args.max_steps > 0 else args.num_epochs * steps_per_epoch

    print(f"=== StellarAI 文本训练 ===")
    print(f"总样本数: {len(dataset.samples)}")
    print(f"总步数: {total_steps}")
    print(f"模型: {model}")
    params = model.count_params()
    print(f"参数分布: { {k: round(v,2) for k,v in params.items()} }")

    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    logs: List[Dict[str, Any]] = []

    try:
        for epoch in range(args.num_epochs):
            for batch in loader:
                global_step += 1
                if args.max_steps > 0 and global_step > args.max_steps:
                    break

                current_lr = get_lr_cosine_warmup(
                    global_step, args.warmup_steps, total_steps,
                    args.learning_rate
                )

                loss = train_step(model, batch, optimizer, current_lr)
                running_loss += loss

                # 日志
                if global_step % args.log_every == 0:
                    avg_loss = running_loss / args.log_every
                    elapsed = time.time() - start_time
                    sps = global_step / elapsed
                    print(
                        f"[Step {global_step}/{total_steps}] "
                        f"loss={avg_loss:.4f} "
                        f"lr={current_lr:.2e} "
                        f"{sps:.2f} steps/s "
                        f"epoch={epoch + 1}/{args.num_epochs}"
                    )
                    logs.append({
                        "step": global_step, "loss": avg_loss,
                        "lr": current_lr, "sps": sps,
                        "elapsed_sec": int(elapsed),
                    })
                    running_loss = 0.0

                # 保存
                if global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"stellar_step{global_step}.npz")
                    model.save_weights(ckpt_path)
                    print(f"  -> 保存检查点: {ckpt_path}")

            if args.max_steps > 0 and global_step > args.max_steps:
                break
    except KeyboardInterrupt:
        print("训练被用户中断，正在保存...")

    # 最终保存
    final_path = os.path.join(args.output_dir, "stellar_final.npz")
    model.save_weights(final_path)

    # 保存词表
    tok_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    # 保存日志
    with open(os.path.join(args.output_dir, "train_log.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "args": vars(args),
            "config": cfg.to_dict(),
            "logs": logs,
            "final_model": final_path,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成! 最终模型: {final_path}")
    return {"logs": logs, "final_path": final_path}
