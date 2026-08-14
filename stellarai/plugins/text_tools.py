"""
文本工具插件
============
文本统计、格式化、摘要等文本处理功能
"""

from .base import Plugin


class TextToolsPlugin(Plugin):
    """文本处理工具: 统计、摘要、格式化"""

    name = "text"
    description = "文本处理工具，支持字数统计、摘要提取、大小写转换等"

    def __init__(self):
        super().__init__()
        self.examples = [
            "问：'人工智能是未来'有多少个字？答：[TOOL:text] count 人工智能是未来 [/TOOL] 共7个字符，其中中文7个字。",
            "问：统计'Hello World'的单词数。答：[TOOL:text] words Hello World [/TOOL] 共2个单词。",
            "问：把'hello'转成大写。答：[TOOL:text] upper hello [/TOOL] HELLO",
            "问：把'WORLD'转成小写。答：[TOOL:text] lower WORLD [/TOOL] world",
            "问：'Python is great'反转。答：[TOOL:text] reverse Python is great [/TOOL] taerg si nohtyP",
        ]

    def call(self, args: str) -> str:
        """执行文本工具"""
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            return "用法: [command] [text]，命令: count/words/upper/lower/reverse"

        command = parts[0].lower()
        text = parts[1].strip()

        if command == "count":
            total = len(text)
            chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            english = sum(1 for c in text if c.isascii() and c.isalpha())
            digits = sum(1 for c in text if c.isdigit())
            return f"共{total}个字符，其中中文{chinese}个字，英文{english}个字母，数字{digits}个。"

        elif command == "words":
            words = text.split()
            return f"共{len(words)}个单词: {', '.join(words)}"

        elif command == "upper":
            return text.upper()

        elif command == "lower":
            return text.lower()

        elif command == "reverse":
            return text[::-1]

        elif command == "summary":
            # 简单摘要：取前50字
            if len(text) <= 50:
                return text
            return text[:50] + "..."

        else:
            return f"未知命令: {command}。可用命令: count, words, upper, lower, reverse, summary"
