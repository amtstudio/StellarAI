"""
StellarAI 插件系统
==================
让AI模型具备工具调用能力，类似Function Calling
支持: 计算器、知识库、翻译、代码执行、天气查询等
"""

from .base import Plugin, PluginManager, create_default_manager
from .calculator import CalculatorPlugin
from .knowledge import KnowledgePlugin
from .translator import TranslatorPlugin
from .text_tools import TextToolsPlugin
from .time_tool import TimePlugin

__all__ = [
    "Plugin",
    "PluginManager",
    "create_default_manager",
    "CalculatorPlugin",
    "KnowledgePlugin",
    "TranslatorPlugin",
    "TextToolsPlugin",
    "TimePlugin",
]
