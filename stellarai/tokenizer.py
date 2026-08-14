"""
StellarAI 简易分词器
支持BPE分词，包含中英文特殊token处理
可扩展接入sentencepiece等更专业分词器
"""

import re
import json
import os
from typing import List, Optional, Dict, Tuple
from collections import Counter


class SimpleTokenizer:
    """
    简易BPE分词器
    - 基于字符级BPE
    - 支持中英文混合
    - 包含特殊token: [PAD], [BOS], [EOS], [UNK], [IMG], [SEP]
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
        "[CLS]": 5,
        "[SEP]": 6,
        "[IMG]": 7,   # 图像占位符 - 序列中表示一个视觉patch的位置
        "[BOI]": 8,   # 图像开始
        "[EOI]": 9,   # 图像结束
    }
    # 保留0-127给特殊token使用，128开始是常规字符/BPE合并结果
    RESERVED_SPECIAL_RANGE = 128

    def __init__(self, vocab_size: int = 32000, vocab_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.bpe_merges: Dict[Tuple[str, str], int] = {}
        # 基础字符/BPE从保留区后开始
        self._next_id = self.RESERVED_SPECIAL_RANGE

        # 预注册基础字符集
        self._init_base_chars()

        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def _init_base_chars(self):
        """初始化基础字符"""
        base_chars = [' ']
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            base_chars.append(c)
        for c in '0123456789':
            base_chars.append(c)
        base_chars.extend(list('.,!?;:()[]{}<>"\'\\/-_@#$%^&*+=|~`，。！？；：""''（）【】《》、……—·￥'))
        # 注册常用CJK统一表意文字 (一级常用字范围，覆盖约99%中文文本使用频率)
        # 范围: 0x4e00-0x7080 共约8832字，足够一般场景使用
        for code in range(0x4e00, 0x7081):
            base_chars.append(chr(code))
        # 额外常用字符
        extras = '的一是不了人我在有他这中大来上个国到说们为子和你地出会也时要就可以对生能而那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯月话合回特代内信表化老给世位次度任常先海川门士原通业教儿走答气入少马次真图安家东七口女变西士等风少南'
        for ch in extras:
            base_chars.append(ch)

        for ch in base_chars:
            if ch not in self.token_to_id and self._next_id < 32000:
                self.token_to_id[ch] = self._next_id
                self.id_to_token[self._next_id] = ch
                self._next_id += 1

    def _split_text(self, text: str) -> List[str]:
        """按词分割文本"""
        # 英文按空格和标点分割，中文按字分割
        tokens = []
        current = []
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':  # CJK
                if current:
                    tokens.append(''.join(current))
                    current = []
                tokens.append(ch)
            elif ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append(''.join(current))
                    current = []
                if ch.strip():
                    tokens.append(ch)
        if current:
            tokens.append(''.join(current))
        return tokens

    def _word_to_symbols(self, word: str) -> List[str]:
        """将词转为符号序列，末尾加 </w>"""
        if len(word) == 0:
            return []
        if '\u4e00' <= word <= '\u9fff' or len(word) == 1:
            return [word]
        return list(word[:-1]) + [word[-1] + '</w>']

    def _get_pairs(self, symbols: List[str]) -> Counter:
        """获取相邻符号对"""
        pairs = Counter()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def train(self, texts: List[str], num_merges: int = 10000, verbose: bool = True):
        """在语料上训练BPE合并"""
        # 构建词频
        word_freq = Counter()
        for text in texts:
            for word in self._split_text(text.lower()):
                if word:
                    word_freq[word] += 1

        # 词 -> 符号序列
        vocab = {}
        for word, freq in word_freq.items():
            vocab[word] = (self._word_to_symbols(word), freq)

        # BPE合并
        for i in range(num_merges):
            if self._next_id >= self.vocab_size:
                if verbose:
                    print(f"达到词表上限 {self.vocab_size}，停止训练")
                break

            pair_freq = Counter()
            for symbols, freq in vocab.values():
                for pair, count in self._get_pairs(symbols).items():
                    pair_freq[pair] += count * freq

            if not pair_freq:
                break

            best_pair = pair_freq.most_common(1)[0][0]
            if pair_freq[best_pair] < 2:
                if verbose:
                    print(f"无更多高频合并对，停止训练（第{i}次合并）")
                break

            # 注册合并
            merged = best_pair[0] + best_pair[1]
            if merged not in self.token_to_id and self._next_id < 32000:
                self.token_to_id[merged] = self._next_id
                self.id_to_token[self._next_id] = merged
                self.bpe_merges[best_pair] = self._next_id
                self._next_id += 1

            # 应用合并
            new_vocab = {}
            for word, (symbols, freq) in vocab.items():
                new_symbols = self._apply_merge(symbols, best_pair)
                new_vocab[word] = (new_symbols, freq)
            vocab = new_vocab

            if verbose and (i + 1) % 1000 == 0:
                print(f"BPE 训练进度: {i + 1}/{num_merges} 合并, 词表大小: {self._next_id}")

        if verbose:
            print(f"BPE训练完成: 共 {len(self.bpe_merges)} 次合并, 词表大小: {self._next_id}")

    def _apply_merge(self, symbols: List[str], pair: Tuple[str, str]) -> List[str]:
        """对符号序列应用一次合并"""
        if len(symbols) < 2:
            return symbols

        result = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                result.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        return result

    def _apply_bpe(self, word: str) -> List[str]:
        """对单个词应用BPE编码"""
        if '\u4e00' <= word <= '\u9fff' or len(word) == 0:
            return [word] if word else []

        symbols = self._word_to_symbols(word)
        if len(symbols) < 2:
            return symbols

        # 按合并顺序应用
        changed = True
        while changed and len(symbols) > 1:
            changed = False
            pairs = self._get_pairs(symbols)
            # 找排名最高的可合并对
            best_pair = None
            best_rank = float('inf')
            for pair in pairs:
                if pair in self.bpe_merges and self.bpe_merges[pair] < best_rank:
                    best_pair = pair
                    best_rank = self.bpe_merges[pair]
            if best_pair is not None:
                symbols = self._apply_merge(symbols, best_pair)
                changed = True

        return symbols

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               max_length: Optional[int] = None, dynamic_register: bool = True) -> List[int]:
        """编码文本为token id序列
        Args:
            dynamic_register: 词表未满时，自动注册未知的单字符（对中文非常有用）
        """
        tokens = []
        if add_bos:
            tokens.append(self.SPECIAL_TOKENS["[BOS]"])

        def _resolve_char(ch):
            """处理单个字符：查找 -> 动态注册 -> UNK"""
            if ch in self.token_to_id:
                return self.token_to_id[ch]
            # 动态注册单字符（中文/其他未知字符）
            if dynamic_register and self._next_id < self.vocab_size:
                self.token_to_id[ch] = self._next_id
                self.id_to_token[self._next_id] = ch
                self._next_id += 1
                return self.token_to_id[ch]
            return self.SPECIAL_TOKENS["[UNK]"]

        for word in self._split_text(text):
            bpe_tokens = self._apply_bpe(word)
            for tok in bpe_tokens:
                if tok in self.token_to_id:
                    tokens.append(self.token_to_id[tok])
                else:
                    for ch in tok:
                        tokens.append(_resolve_char(ch))

        if add_eos:
            tokens.append(self.SPECIAL_TOKENS["[EOS]"])

        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """解码token id序列为文本"""
        result = []
        for tid in ids:
            if skip_special and tid in self.SPECIAL_TOKENS.values():
                continue
            token = self.id_to_token.get(tid, "[UNK]")
            if token.endswith('</w>'):
                result.append(token[:-4] + ' ')
            elif token == '[UNK]':
                result.append('\ufffd')  # � 未知字符占位
            else:
                result.append(token)
        return ''.join(result).strip()

    def save(self, path: str):
        """保存词表和合并规则"""
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "bpe_merges": {f"{a}|{b}": v for (a, b), v in self.bpe_merges.items()},
            "next_id": self._next_id,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """加载词表和合并规则"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.bpe_merges = {}
        for k, v in data["bpe_merges"].items():
            a, b = k.split('|')
            self.bpe_merges[(a, b)] = v
        self._next_id = data.get("next_id", max(self.id_to_token.keys()) + 1)
        self.vocab_size = data.get("vocab_size", self.vocab_size)

    def vocab_size_effective(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS["[PAD]"]

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["[BOS]"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["[EOS]"]

    @property
    def img_id(self) -> int:
        return self.SPECIAL_TOKENS["[IMG]"]
