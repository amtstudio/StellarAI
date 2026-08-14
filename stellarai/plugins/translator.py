"""
翻译插件
========
支持中英互译，基于内置词典和规则
"""

from .base import Plugin


class TranslatorPlugin(Plugin):
    """中英互译: 基于词典和规则的轻量翻译"""

    name = "translate"
    description = "中英互译工具，支持中文到英文和英文到中文的翻译"

    # 内置核心词典
    _dict_zh2en = {
        "你好": "hello", "再见": "goodbye", "谢谢": "thank you",
        "人工智能": "artificial intelligence", "机器学习": "machine learning",
        "深度学习": "deep learning", "计算机": "computer",
        "科学": "science", "技术": "technology", "数据": "data",
        "模型": "model", "训练": "training", "网络": "network",
        "语言": "language", "图像": "image", "理解": "understand",
        "生成": "generate", "学习": "learn", "知识": "knowledge",
        "问题": "question", "答案": "answer", "方法": "method",
        "算法": "algorithm", "程序": "program", "代码": "code",
        "系统": "system", "智能": "intelligence", "自然": "natural",
        "处理": "process", "分析": "analyze", "预测": "predict",
        "优化": "optimize", "性能": "performance", "参数": "parameter",
        "神经": "neural", "注意力": "attention", "变换器": "transformer",
        "编码": "encode", "解码": "decode", "分类": "classify",
        "检测": "detect", "识别": "recognize", "分割": "segment",
        "世界": "world", "中国": "China", "美国": "America",
        "时间": "time", "今天": "today", "明天": "tomorrow",
        "是": "is", "的": "of", "和": "and", "在": "in",
        "一个": "a", "这": "this", "那": "that",
    }

    _dict_en2zh = {v: k for k, v in _dict_zh2en.items()}

    def __init__(self):
        super().__init__()
        self.examples = [
            "问：把'人工智能'翻译成英文。答：[TOOL:translate] 人工智能 [/TOOL] artificial intelligence",
            "问：'machine learning'的中文是什么？答：[TOOL:translate] machine learning [/TOOL] 机器学习",
            "问：翻译'deep learning'。答：[TOOL:translate] deep learning [/TOOL] 深度学习",
            "问：'computer'中文意思是什么？答：[TOOL:translate] computer [/TOOL] 计算机",
            "问：翻译'neural network'。答：[TOOL:translate] neural network [/TOOL] 神经网络",
        ]

    def _is_chinese(self, text: str) -> bool:
        """判断文本是否为中文"""
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def _translate_zh2en(self, text: str) -> str:
        """中文翻译为英文"""
        result = []
        i = 0
        while i < len(text):
            matched = False
            # 贪心匹配最长的词
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i + length]
                if word in self._dict_zh2en:
                    result.append(self._dict_zh2en[word])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch in self._dict_zh2en:
                    result.append(self._dict_zh2en[ch])
                elif ch.strip():
                    result.append(ch)
                i += 1
        return " ".join(result)

    def _translate_en2zh(self, text: str) -> str:
        """英文翻译为中文"""
        words = text.lower().strip().split()
        result = []
        i = 0
        while i < len(words):
            matched = False
            # 尝试匹配2-3词短语
            for length in range(min(3, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + length])
                if phrase in self._dict_en2zh:
                    result.append(self._dict_en2zh[phrase])
                    i += length
                    matched = True
                    break
            if not matched:
                word = words[i]
                if word in self._dict_en2zh:
                    result.append(self._dict_en2zh[word])
                else:
                    result.append(word)
                i += 1
        return "".join(result)

    def call(self, args: str) -> str:
        """执行翻译"""
        text = args.strip()
        if not text:
            return "请提供要翻译的文本"
        if self._is_chinese(text):
            return self._translate_zh2en(text)
        else:
            return self._translate_en2zh(text)
