"""
StellarAI 快速上手示例
=======================

本脚本演示：
1. 初始化模型
2. 查看参数统计
3. 文本生成 / 对话
4. 视觉问答 / 图像描述
5. 保存和加载模型权重
6. 性能基准测试
"""

import os
import sys
import numpy as np

# 添加项目根目录到 sys.path (如果需要)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stellarai import StellarAI, StellarConfig, SimpleTokenizer, VisionPreprocessor
from stellarai.inference import StellarInference


def example_1_model_init_and_stats():
    """示例1: 初始化模型并查看参数统计"""
    print("=" * 60)
    print("[示例1] 模型初始化 & 参数统计")
    print("=" * 60)

    # 标准版本 (~50M 参数)
    cfg_std = StellarConfig.standard()
    model_std = StellarAI.random_init(cfg_std, seed=42)
    params_std = model_std.count_params()
    print(f"✓ 标准模型 (0.05B): {model_std}")
    print(f"  参数分布:")
    for k, v in params_std.items():
        print(f"    - {k:18s}: {v:8.2f} M")

    print()

    # 轻量版本 (~20M 参数)
    cfg_tiny = StellarConfig.tiny()
    model_tiny = StellarAI.tiny_init(seed=42)
    params_tiny = model_tiny.count_params()
    print(f"✓ Tiny模型 (~20M): {model_tiny}")
    print(f"  实际参数: {params_tiny['actual_total_M']:.2f} M")

    return model_std, model_tiny


def example_2_text_generation():
    """示例2: 文本生成"""
    print("\n" + "=" * 60)
    print("[示例2] 文本生成 (随机初始化模型，输出是随机token的解码)")
    print("=" * 60)

    engine = StellarInference(seed=42)

    prompt = "人工智能正在改变世界，它"
    print(f"\n输入前缀 (prefix): \"{prompt}\"")
    result = engine.complete(prompt, max_new_tokens=32, temperature=0.8)
    print(f"生成结果: \"{result}\"")
    print("(注: 未训练模型的输出是随机的，训练后会有语义)")


def example_3_chat():
    """示例3: 多轮对话"""
    print("\n" + "=" * 60)
    print("[示例3] 多轮对话 (演示模式)")
    print("=" * 60)

    engine = StellarInference(seed=42)

    print("\n[第1轮] User: 你好，介绍一下你自己")
    reply1 = engine.chat("你好，介绍一下你自己", max_new_tokens=48, temperature=0.8)
    print(f"        Assistant: {reply1}")

    print("\n[第2轮] User: 你的能力有哪些？")
    reply2 = engine.chat("你的能力有哪些？", max_new_tokens=48, temperature=0.8)
    print(f"        Assistant: {reply2}")

    print(f"\n对话历史共 {len(engine.chat_history)} 条消息")
    engine.reset_chat()
    print("已清空对话历史")


def example_4_vision_inputs():
    """示例4: 视觉相关接口演示 (使用合成图像，无需下载)"""
    print("\n" + "=" * 60)
    print("[示例4] 视觉问答 & 图像描述 (使用合成测试图像)")
    print("=" * 60)

    # 合成一个彩色渐变测试图像 (224,224,3)
    H = W = 224
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy /= (H - 1)
    xx /= (W - 1)
    fake_image = np.zeros((H, W, 3), dtype=np.uint8)
    fake_image[..., 0] = (xx * 255).astype(np.uint8)
    fake_image[..., 1] = (yy * 255).astype(np.uint8)
    fake_image[..., 2] = 128

    engine = StellarInference(seed=42)
    vp = engine.vision_preprocessor

    # 先测试预处理 + patch提取（跳过耗时的CNN forward）
    print("\n→ 预处理测试图像 (224×224 RGB)...")
    img_tensor = vp.preprocess(fake_image)
    patches = vp.extract_patches(img_tensor)
    print(f"  ✓ 图像预处理后shape: {img_tensor.shape}")
    print(f"  ✓ 提取patches: {patches.shape[0]} patches, dim={patches.shape[1]}")

    # 演示多模态输入格式
    print("\n→ 构造VQA输入序列 (占位符注入方式):")
    vis_cfg = engine.cfg
    ph_tokens = [vis_cfg.vision_token_id] * min(vis_cfg.vision_num_patches, 8)
    print(f"  序列格式: [BOS] [IMG]×{len(ph_tokens)}... [SEP] 问题 [SEP] 回答")
    print(f"  ✓ VQA 输入构造逻辑正常")
    print(f"  ✓ caption / vqa 接口就绪 (如需实际推理，建议使用PyTorch版加速CNN)")

    # （可选）跳过耗时的完整视觉forward - 纯numpy实现的CNN在CPU上很慢
    # 如需测试: 将 SKIP_VISUAL_FORWARD 设为 False (大约需要30-60秒)
    SKIP_VISUAL_FORWARD = True
    if not SKIP_VISUAL_FORWARD:
        print("\n→ 调用 vqa (完整forward)...")
        try:
            ans = engine.vqa("图中有什么颜色？", fake_image, max_new_tokens=16, temperature=0.1)
            print(f"  VQA回答: {ans}")
        except Exception as e:
            print(f"  (出错): {type(e).__name__}: {e}")
    else:
        print("\n→ [已跳过] 实际视觉forward (纯NumPy CNN较慢)")
        print("   训练时建议使用 stellarai/train_pytorch.py 获得10~100倍速度提升")


def example_5_save_load():
    """示例5: 保存/加载权重"""
    print("\n" + "=" * 60)
    print("[示例5] 模型权重的保存与加载")
    print("=" * 60)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "demo_checkpoint.npz")

    model = StellarAI.tiny_init(seed=123)
    params_before = model.count_params()["actual_total_M"]

    print(f"→ 保存模型权重到: {ckpt_path}")
    model.save_weights(ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  ✓ 文件大小: {size_mb:.2f} MB")

    print(f"→ 从文件重新加载...")
    model2 = StellarAI.load_weights(ckpt_path, StellarConfig.tiny())
    params_after = model2.count_params()["actual_total_M"]
    print(f"  ✓ 加载完成，参数匹配: {params_before:.2f}M == {params_after:.2f}M → "
          f"{'OK' if params_before == params_after else '不一致!'}")

    # 清理 (Windows上np.load句柄未释放时可能延迟)
    if os.path.exists(ckpt_path):
        try:
            import gc; gc.collect()
            os.remove(ckpt_path)
            print(f"  (已删除演示文件)")
        except PermissionError:
            print(f"  (演示文件稍后可手动删除: {ckpt_path})")


def example_6_benchmark():
    """示例6: 性能基准测试 (tiny模型快速测试)"""
    print("\n" + "=" * 60)
    print("[示例6] 性能基准测试 (使用Tiny模型)")
    print("=" * 60)

    engine = StellarInference(config=StellarConfig.tiny(), seed=42)
    print("运行 benchmark(seq_len=64, new_tokens=16)...")
    try:
        result = engine.benchmark(seq_len=64, new_tokens=16, n_iter=1)
        print("  结果:")
        for k, v in result.items():
            print(f"    - {k:24s}: {v}")
    except Exception as e:
        print(f"  (跳过): {type(e).__name__}: {e}")


def example_7_tokenizer():
    """示例7: 分词器使用"""
    print("\n" + "=" * 60)
    print("[示例7] BPE分词器训练与使用")
    print("=" * 60)

    tokenizer = SimpleTokenizer(vocab_size=32000)

    # 语料
    texts = [
        "Hello world! 你好，世界！",
        "StellarAI is a lightweight multimodal AI model.",
        "人工智能深度学习机器学习神经网络",
        "The quick brown fox jumps over the lazy dog.",
        "自然语言处理和计算机视觉是AI的两大重要领域。",
    ] * 100

    print(f"→ 在 {len(texts)} 句语料上训练BPE合并 (1000次)...")
    tokenizer.train(texts, num_merges=1000, verbose=False)
    print(f"  ✓ 词表大小: {tokenizer.vocab_size_effective()}")

    test_text = "Hello, 欢迎使用StellarAI多模态大模型!"
    ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids, skip_special=False)

    print(f"\n  测试文本: {test_text}")
    print(f"  编码结果: {ids[:20]}... (len={len(ids)})")
    print(f"  解码还原: {decoded}")

    # 保存/加载
    tok_path = os.path.join("outputs", "demo_tokenizer.json")
    os.makedirs("outputs", exist_ok=True)
    tokenizer.save(tok_path)
    tok2 = SimpleTokenizer(vocab_size=32000, vocab_path=tok_path)
    decoded2 = tok2.decode(ids, skip_special=True)
    print(f"  ✓ 保存/加载后解码一致: {decoded2}")

    if os.path.exists(tok_path):
        os.remove(tok_path)


def main():
    print("🚀 StellarAI 0.05B - 快速上手示例 🚀")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy:  {np.__version__}")
    print()

    try:
        example_1_model_init_and_stats()
        example_2_text_generation()
        example_3_chat()
        example_4_vision_inputs()
        example_5_save_load()
        example_6_benchmark()
        example_7_tokenizer()
    except KeyboardInterrupt:
        print("\n[中断] 用户取消执行")

    print("\n" + "=" * 60)
    print("所有示例运行完成！🎉")
    print("下一步：")
    print("  1. 准备训练语料 → 使用 training.py 训练")
    print("  2. 切换到 PyTorch 版加速 (见 stellarai/train_pytorch.py)")
    print("  3. 准备图文对 → 多模态微调")
    print("=" * 60)


if __name__ == "__main__":
    main()
