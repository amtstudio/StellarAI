"""
插件基类与管理器
================
定义插件接口规范和调用管理逻辑
"""

import re
import json
from typing import Optional, List, Dict, Any, Callable


class Plugin:
    """插件基类: 所有插件继承此类"""

    name: str = "base"
    description: str = ""
    # 调用格式: [TOOL:plugin_name] args [/TOOL]
    CALL_PATTERN = r"\[TOOL:(\w+)\]\s*(.*?)\s*\[/TOOL\]"

    def __init__(self):
        self.examples: List[str] = []

    def call(self, args: str) -> str:
        """执行插件功能, 返回结果字符串"""
        raise NotImplementedError

    def get_description(self) -> str:
        """返回插件描述（用于训练和提示）"""
        return self.description

    def get_examples(self) -> List[str]:
        """返回调用示例（用于训练）"""
        return self.examples


class PluginManager:
    """
    插件管理器
    - 注册/注销插件
    - 从文本中解析工具调用并执行
    - 将结果注入回生成流程
    """

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin):
        """注册插件"""
        self.plugins[plugin.name] = plugin

    def unregister(self, name: str):
        """注销插件"""
        self.plugins.pop(name, None)

    def list_plugins(self) -> List[str]:
        """列出已注册的插件名"""
        return list(self.plugins.keys())

    def get_system_prompt(self) -> str:
        """生成系统提示，告诉模型可用插件"""
        lines = ["你可以使用以下工具来增强能力:"]
        for name, plugin in self.plugins.items():
            lines.append(f"- [TOOL:{name}] ... [/TOOL]: {plugin.get_description()}")
        lines.append("调用格式: [TOOL:工具名] 参数 [/TOOL]")
        lines.append("模型会在生成中插入工具调用，系统执行后返回结果。")
        return "\n".join(lines)

    def extract_calls(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取所有工具调用"""
        calls = []
        for match in re.finditer(Plugin.CALL_PATTERN, text, re.DOTALL):
            name = match.group(1)
            args = match.group(2).strip()
            calls.append({"name": name, "args": args, "full_match": match.group(0)})
        return calls

    def execute(self, text: str) -> str:
        """
        执行文本中的所有工具调用，返回替换后的文本
        [TOOL:calc] 2+3 [/TOOL] -> [TOOL:calc] 2+3 [/TOOL] -> 5
        """
        calls = self.extract_calls(text)
        result = text
        for call in calls:
            plugin = self.plugins.get(call["name"])
            if plugin:
                try:
                    output = plugin.call(call["args"])
                except Exception as e:
                    output = f"[错误: {e}]"
                # 将工具调用替换为结果
                result = result.replace(
                    call["full_match"],
                    f"[TOOL_RESULT]{output}[/TOOL_RESULT]"
                )
            else:
                result = result.replace(
                    call["full_match"],
                    f"[TOOL_RESULT][错误: 未知工具 {call['name']}][/TOOL_RESULT]"
                )
        return result

    def has_pending_calls(self, text: str) -> bool:
        """检查文本中是否有未执行的工具调用"""
        return bool(self.extract_calls(text))

    def get_training_data(self) -> List[str]:
        """获取所有插件的训练示例"""
        data = []
        for plugin in self.plugins.values():
            data.extend(plugin.get_examples())
        return data


def create_default_manager() -> PluginManager:
    """创建带默认插件的管理器"""
    from .calculator import CalculatorPlugin
    from .knowledge import KnowledgePlugin
    from .translator import TranslatorPlugin
    from .text_tools import TextToolsPlugin
    from .time_tool import TimePlugin

    mgr = PluginManager()
    mgr.register(CalculatorPlugin())
    mgr.register(KnowledgePlugin())
    mgr.register(TranslatorPlugin())
    mgr.register(TextToolsPlugin())
    mgr.register(TimePlugin())
    return mgr
