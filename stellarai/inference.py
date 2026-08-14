"""
StellarAI 高层推理接口
简化模型使用，提供：
- Chat 接口 (对话)
- VQA 接口 (视觉问答)
- 批量推理
- 流式生成
"""

import time
from typing import Optional, List, Dict, Any, Tuple, Iterator

import numpy as np

from .model import StellarAI
from .config import StellarConfig
from .tokenizer import SimpleTokenizer
from .vision import VisionPreprocessor


class StellarInference:
    """高层推理封装，管理分词器/预处理器/模型生命周期"""

    def __init__(
        self,
        model: Optional[StellarAI] = None,
        config: Optional[StellarConfig] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        vision_preprocessor: Optional[VisionPreprocessor] = None,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",  # cpu / cuda (cuda需切换PyTorch实现)
        seed: int = 42,
    ):
        self.cfg = config or StellarConfig.standard()

        # 加载模型
        if model is not None:
            self.model = model
        elif checkpoint_path and checkpoint_path.endswith((".npz",)):
            self.model = StellarAI.load_weights(checkpoint_path, self.cfg)
        else:
            print(f"[StellarInference] 使用随机初始化模型 (用于测试，请训练后再使用)")
            self.model = StellarAI.random_init(self.cfg, seed=seed)

        # 分词器
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_path:
            self.tokenizer = SimpleTokenizer(self.cfg.vocab_size, vocab_path=tokenizer_path)
        else:
            self.tokenizer = SimpleTokenizer(self.cfg.vocab_size)

        # 视觉预处理器
        self.vision_preprocessor = vision_preprocessor or VisionPreprocessor(
            self.cfg.image_size, self.cfg.patch_size
        )

        # 对话历史
        self.chat_history: List[Dict[str, str]] = []

    # --------------------------------------------------------
    #  工具
    # --------------------------------------------------------
    def reset_chat(self):
        """清空对话历史"""
        self.chat_history = []

    def _format_chat_prompt(self, user_text: str, system_prompt: Optional[str] = None) -> str:
        """格式化多轮对话为单提示"""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for msg in self.chat_history:
            parts.append(f"{msg['role']}: {msg['content']}")
        parts.append(f"User: {user_text}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # --------------------------------------------------------
    #  核心生成
    # --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.0,
        verbose: bool = False,
    ) -> str:
        """单次生成"""
        return self.model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            image=image,
            tokenizer=self.tokenizer,
            vision_preprocessor=self.vision_preprocessor,
            verbose=verbose,
        )

    def generate_stream(
        self,
        prompt: str,
        image: Optional[Any] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Iterator[str]:
        """流式生成，每次yield新生成的token解码文本"""
        # 复用generate的逻辑，逐token产出
        cfg = self.cfg
        tokenizer = self.tokenizer
        vp = self.vision_preprocessor

        input_tokens = [tokenizer.bos_id]

        img_tensor = None
        ph_start = -1
        if image is not None:
            ph_start = len(input_tokens)
            for _ in range(cfg.vision_num_patches):
                input_tokens.append(cfg.vision_token_id)
            input_tokens.append(cfg.sep_token_id)
            img_tensor = vp.preprocess(image)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor[np.newaxis, ...]

        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False,
                                       max_length=cfg.max_seq_len - len(input_tokens) - 1)
        input_tokens.extend(prompt_ids)
        input_tokens.append(cfg.sep_token_id)

        input_ids = np.array([input_tokens], dtype=np.int64)

        placeholders = None
        if ph_start > 0:
            placeholders = np.array([[ph_start, ph_start + cfg.vision_num_patches]])
            placeholders = placeholders[:, np.newaxis, :]

        start_decode = len(input_tokens)

        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= cfg.max_seq_len:
                break

            logits = self.model.forward(input_ids, images=img_tensor, image_placeholders=placeholders)
            next_logits = logits[0, -1, :cfg.vocab_size]

            if temperature <= 0:
                next_id = int(np.argmax(next_logits))
            else:
                next_logits = next_logits / max(temperature, 1e-4)
                if top_k > 0:
                    k = min(top_k, next_logits.shape[0])
                    top_k_ids = np.argpartition(-next_logits, k - 1)[:k]
                    top_k_logits = next_logits[top_k_ids]
                    # softmax
                    xmax = top_k_logits.max()
                    e = np.exp(top_k_logits - xmax)
                    probs = e / e.sum()
                    next_id = int(np.random.choice(top_k_ids, p=probs))
                else:
                    xmax = next_logits.max()
                    e = np.exp(next_logits - xmax)
                    probs = e / e.sum()
                    next_id = int(np.random.choice(next_logits.shape[0], p=probs))

            if next_id == tokenizer.eos_id:
                break

            # yield 当前解码字符
            char = tokenizer.decode([next_id], skip_special=True)
            if char:
                yield char

            input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)

    # --------------------------------------------------------
    #  对话接口
    # --------------------------------------------------------
    def chat(
        self,
        user_text: str,
        system_prompt: Optional[str] = "你是StellarAI，一个友好、专业的多模态AI助手。回答准确、简洁。",
        image: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        多轮对话
        用法:
            engine = StellarInference()
            reply = engine.chat("你好")
            print(reply)
        """
        prompt = self._format_chat_prompt(user_text, system_prompt)

        if stream:
            # 流模式: 需要在stream中记录assistant回复
            full = []

            def stream_wrapper():
                for tok in self.generate_stream(
                    prompt, image=image, max_new_tokens=max_new_tokens,
                    temperature=temperature,
                ):
                    full.append(tok)
                    yield tok

            # 返回stream，但需要在结束时更新历史
            # 由于generator是惰性的，这里我们用包装方案：
            # 用一个helper generator记录所有输出
            def with_history():
                text = ""
                for tok in stream_wrapper():
                    text += tok
                    yield tok
                self.chat_history.append({"role": "User", "content": user_text})
                self.chat_history.append({"role": "Assistant", "content": text})
            return with_history()

        reply = self.generate(
            prompt, image=image, max_new_tokens=max_new_tokens, temperature=temperature,
        )
        self.chat_history.append({"role": "User", "content": user_text})
        self.chat_history.append({"role": "Assistant", "content": reply})
        return reply

    # --------------------------------------------------------
    #  VQA 视觉问答
    # --------------------------------------------------------
    def vqa(
        self,
        question: str,
        image: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """视觉问答 (VQA) - 针对image问question"""
        prompt = f"请仔细观察这张图片并回答问题。\n问题：{question}\n回答："
        return self.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=1,  # greedy for VQA
        )

    def caption(self, image: Any, max_new_tokens: int = 128, temperature: float = 0.8) -> str:
        """图像描述 (Image Captioning)"""
        prompt = "请用简洁的语言描述这张图片中的内容："
        return self.generate(prompt, image=image, max_new_tokens=max_new_tokens, temperature=temperature)

    # --------------------------------------------------------
    #  批量推理
    # --------------------------------------------------------
    def batch_generate(
        self,
        prompts: List[str],
        images: Optional[List[Any]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> List[str]:
        """批量生成 (注意：纯NumPy实现下是串行，实际用PyTorch版才会并行)"""
        results = []
        N = len(prompts)
        for i in range(N):
            img = images[i] if images and i < len(images) else None
            results.append(self.generate(
                prompts[i], image=img, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k,
            ))
        return results

    # --------------------------------------------------------
    #  上下文相关：文本补全
    # --------------------------------------------------------
    def complete(self, prefix: str, max_new_tokens: int = 64, temperature: float = 0.7) -> str:
        """文本补全"""
        return self.generate(prefix, max_new_tokens=max_new_tokens, temperature=temperature)

    # --------------------------------------------------------
    #  benchmark
    # --------------------------------------------------------
    def benchmark(self, seq_len: int = 128, new_tokens: int = 64, n_iter: int = 3) -> Dict[str, float]:
        """性能基准测试"""
        import time
        cfg = self.cfg
        tokenizer = self.tokenizer

        # 构造一个seq_len长度的输入
        input_ids = np.random.randint(4, cfg.vocab_size - 4, size=(1, seq_len), dtype=np.int64)

        # warmup
        self.model.forward(input_ids)

        # forward time
        t0 = time.time()
        for _ in range(n_iter):
            self.model.forward(input_ids)
        t_forward = (time.time() - t0) / n_iter

        # generation time (token/s)
        t0 = time.time()
        n_gen = 0
        ids = input_ids.copy()
        for _ in range(new_tokens):
            logits = self.model.forward(ids)
            nxt = int(np.argmax(logits[0, -1, :cfg.vocab_size]))
            ids = np.concatenate([ids, np.array([[nxt]], dtype=np.int64)], axis=1)
            n_gen += 1
            if ids.shape[1] >= cfg.max_seq_len:
                break
        t_gen = time.time() - t0

        return {
            "forward_ms": round(t_forward * 1000, 2),
            "tokens_per_second": round(n_gen / t_gen, 2),
            "seq_len": seq_len,
            "generated_tokens": n_gen,
            "d_model": cfg.d_model,
            "params_M": self.model.count_params().get("actual_total_M", 0),
        }
