"""
StellarAI PyTorch训练脚本
==========================
用法: python train.py

功能:
  1. 训练BPE分词器
  2. 创建PyTorch模型
  3. 文本预训练 (next-token prediction)
  4. 定期生成样本查看效果
  5. 保存检查点
"""

import os
import sys
import time
import math
import json
import argparse

# 启用无缓冲输出, 确保日志实时写入文件
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stellarai.config import StellarConfig
from stellarai.model_torch import StellarTorch
from stellarai.tokenizer import SimpleTokenizer


# ============================================================
#  数据集
# ============================================================

class TextTrainDataset(Dataset):
    """文本训练数据集 - next-token prediction"""

    def __init__(self, file_path: str, tokenizer: SimpleTokenizer,
                 seq_len: int = 128, min_len: int = 16):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) < min_len:
                    continue
                ids = tokenizer.encode(line, add_bos=True, add_eos=True, max_length=seq_len + 1)
                if len(ids) >= min_len:
                    self.samples.append(ids)

        print(f"  加载训练样本: {len(self.samples)} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # 截断或填充到 seq_len
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
        elif len(ids) < self.seq_len:
            ids = ids + [0] * (self.seq_len - len(ids))  # pad

        input_ids = torch.tensor(ids, dtype=torch.long)
        # next-token: labels = input_ids 右移一位, 第一个位置忽略
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # 最后一个位置没有next token
        # padding位置忽略
        labels[input_ids == 0] = -100
        # BOS位置的label设为下一个token (保留)

        return {"input_ids": input_ids, "labels": labels}


# ============================================================
#  学习率调度
# ============================================================

def get_lr(step, warmup, max_steps, peak_lr, min_lr_ratio=0.1):
    """余弦退火 + Warmup"""
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    progress = min(progress, 1.0)
    min_lr = peak_lr * min_lr_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
#  生成样本
# ============================================================

@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device, max_new_tokens=40, temperature=0.7):
    """生成一段文本样本"""
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False, max_length=64)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= model.cfg.max_seq_len:
            break
        out = model.forward(input_ids)
        logits = out["logits"][0, -1, :model.cfg.vocab_size]

        if temperature <= 0:
            next_id = logits.argmax().unsqueeze(0)
        else:
            logits = logits / temperature
            # top-k=20
            v, _ = torch.topk(logits, min(20, logits.shape[-1]))
            logits[logits < v[-1]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.eos_id:
            break

    result_ids = input_ids[0].tolist()
    text = tokenizer.decode(result_ids, skip_special=True)
    model.train()
    return text


# ============================================================
#  主训练函数
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== StellarAI PyTorch 训练 (v2 极致版) ===")
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    # 1. 配置 (CPU用Tiny，GPU用Standard)
    if args.tiny or device == torch.device("cpu"):
        cfg = StellarConfig.tiny()
        print(f"配置: Tiny (适合CPU训练)")
    else:
        cfg = StellarConfig.standard()
        print(f"配置: Standard 0.05B")
    print(f"模型参数: d_model={cfg.d_model}, layers={cfg.text_num_layers}+{cfg.vision_num_layers}+{cfg.fusion_num_layers}")

    # 2. 分词器
    os.makedirs(args.output, exist_ok=True)
    tokenizer_path = os.path.join(args.output, "tokenizer.json")

    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"\n--- 加载已有分词器: {args.tokenizer} ---")
        tokenizer = SimpleTokenizer(cfg.vocab_size)
        tokenizer.load(args.tokenizer)
        print(f"词表大小: {tokenizer.vocab_size_effective()}")
    else:
        print(f"\n--- 分词器训练 ---")
        tokenizer = SimpleTokenizer(cfg.vocab_size)

        # 读取语料训练BPE (大语料取子集加快训练)
        with open(args.data, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        # 注入插件训练数据
        try:
            from stellarai.plugins import create_default_manager
            plugin_mgr = create_default_manager()
            plugin_data = plugin_mgr.get_training_data()
            corpus.extend(plugin_data)
            print(f"语料: {len(corpus)} 行 (含 {len(plugin_data)} 条插件数据)")
        except Exception as e:
            print(f"语料: {len(corpus)} 行 (插件数据加载失败: {e})")

        # BPE训练在大语料上很慢，取子集
        train_corpus = corpus[:5000] if len(corpus) > 5000 else corpus
        if len(corpus) > 5000:
            print(f"BPE训练使用子集: {len(train_corpus)} 行 (总 {len(corpus)} 行)")
        tokenizer.train(train_corpus, num_merges=args.bpe_merges, verbose=False)
        print(f"词表大小: {tokenizer.vocab_size_effective()}")
        tokenizer.save(tokenizer_path)

    # 3. 数据集
    print(f"\n--- 数据加载 ---")
    dataset = TextTrainDataset(args.data, tokenizer, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"数据批次/epoch: {len(loader)}")

    # 4. 模型
    print(f"\n--- 模型创建 ---")
    model = StellarTorch(cfg).to(device)
    params = model.count_params()
    print(f"参数量: {params['total_M']}M (可训练: {params['trainable_M']}M)")

    # 5. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # 更适合语言模型的betas
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 6. 断点续训: 加载已有检查点继续训练
    start_step = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"\n--- 加载检查点: {args.resume} ---")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        best_loss = ckpt.get("loss", float('inf'))
        print(f"  已恢复到 step={start_step}, loss={best_loss:.4f}")
        # 重置优化器状态以适应新语料
        print(f"  注意: 优化器状态已重置以适应新语料")

    # 7. 训练循环
    total_steps = args.steps  # 总训练步数 (从0开始计数到total_steps)
    print(f"\n--- 开始训练 ---")
    print(f"总步数: {total_steps}, batch_size: {args.batch_size}, seq_len: {args.seq_len}")
    print(f"学习率: {args.lr} (warmup={args.warmup})")
    print(f"目标: 从 step {start_step} 训练到 step {total_steps}")
    print()

    model.train()
    global_step = start_step
    running_loss = 0.0
    start_time = time.time()
    log_counter = 0

    # 数据循环
    data_iter = iter(loader)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        global_step += 1

        # 学习率 - 基于当前步数动态调整
        lr = get_lr(global_step, args.warmup, total_steps, args.lr, min_lr_ratio=0.05)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 前向
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, labels=labels)
        loss = out["loss"]

        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        log_counter += 1

        # 日志
        if global_step % args.log_every == 0:
            avg_loss = running_loss / log_counter
            elapsed = time.time() - start_time
            steps_done = global_step - start_step
            sps = steps_done / max(elapsed, 0.1)
            eta = (total_steps - global_step) / max(sps, 0.1)
            perplexity = math.exp(min(avg_loss, 20))
            print(
                f"  [Step {global_step:4d}/{total_steps}] "
                f"loss={avg_loss:.4f}  ppl={perplexity:.1f}  "
                f"lr={lr:.2e}  {sps:.2f}step/s  "
                f"ETA={eta:.0f}s"
            )
            running_loss = 0.0
            log_counter = 0

        # 生成样本
        if global_step % args.sample_every == 0 or global_step == 1:
            print(f"\n  --- 生成样本 (step {global_step}) ---")
            for prompt in ["人工智能", "StellarAI", "深度学习", "机器学习是", "Python是", "问：什么是Transformer"]:
                text = generate_sample(model, tokenizer, prompt, device,
                                       max_new_tokens=30, temperature=0.6)
                print(f"  [{prompt}] -> {text}")
            print()
            model.train()

        # 保存检查点
        if global_step % args.save_every == 0:
            ckpt_path = os.path.join(args.output, f"stellar_step{global_step}.pt")
            current_loss = running_loss / max(log_counter, 1) if log_counter > 0 else avg_loss
            torch.save({
                "model": model.state_dict(),
                "config": cfg.to_dict(),
                "step": global_step,
                "loss": current_loss,
            }, ckpt_path)
            print(f"  >> 检查点已保存: {ckpt_path}")

            # 同时保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(args.output, "stellar_best.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "step": global_step,
                    "loss": best_loss,
                }, best_path)
                print(f"  >> 新最佳模型! loss={best_loss:.4f}: {best_path}")

    # 最终保存
    final_path = os.path.join(args.output, "stellar_final.pt")
    final_loss = running_loss / max(log_counter, 1) if log_counter > 0 else avg_loss
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "step": global_step,
        "loss": final_loss,
    }, final_path)
    print(f"\n=== 训练完成 ===")
    print(f"总步数: {global_step} (本轮新增: {global_step - start_step})")
    print(f"耗时: {time.time() - start_time:.1f}s")
    print(f"最终loss: {final_loss:.4f}")
    print(f"最佳loss: {best_loss:.4f}")
    print(f"最终模型: {final_path}")
    print(f"分词器: {os.path.join(args.output, 'tokenizer.json')}")

    # 最终生成 - 多样化测试
    print(f"\n--- 最终生成效果 ---")
    prompts = [
        "人工智能是",
        "StellarAI",
        "机器学习的核心",
        "深度学习",
        "什么是Transformer",
        "神经网络如何工作",
        "多模态AI",
        "Python是",
        "问：什么是人工智能",
        "问：如何用Python排序",
        "计算 [TOOL:calc] 3*7",
        "翻译 [TOOL:translate] deep learning",
    ]
    for p in prompts:
        text = generate_sample(model, tokenizer, p, device,
                               max_new_tokens=50, temperature=0.7)
        print(f"  [{p}] -> {text}")

    return model, tokenizer


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StellarAI 训练 v2")
    parser.add_argument("--data", type=str, default="data/train_corpus.txt",
                        help="训练语料路径")
    parser.add_argument("--output", type=str, default="outputs/stellar_pt",
                        help="输出目录")
    parser.add_argument("--steps", type=int, default=3000,
                        help="总训练步数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批大小")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="序列长度")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="峰值学习率")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup步数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--bpe_merges", type=int, default=3000,
                        help="BPE合并次数")
    parser.add_argument("--tiny", action="store_true", default=True,
                        help="使用Tiny配置 (CPU推荐)")
    parser.add_argument("--tokenizer", type=str, default="",
                        help="已有的分词器路径 (跳过BPE训练)")
    parser.add_argument("--resume", type=str, default="",
                        help="断点续训检查点路径 (空则从头训练)")
    parser.add_argument("--log_every", type=int, default=20,
                        help="日志间隔步数")
    parser.add_argument("--sample_every", type=int, default=100,
                        help="生成样本间隔步数")
    parser.add_argument("--save_every", type=int, default=500,
                        help="保存检查点间隔步数")

    args = parser.parse_args()
    train(args)
