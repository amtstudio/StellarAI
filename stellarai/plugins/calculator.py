"""
计算器插件
==========
支持四则运算、幂运算、括号、数学函数
"""

import ast
import math
import operator
from typing import Dict, Any
from .base import Plugin


class CalculatorPlugin(Plugin):
    """数学计算器: 支持基本运算和数学函数"""

    name = "calc"
    description = "数学计算器，支持加减乘除、幂运算、三角函数、对数等"

    # 安全的运算符映射
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    # 安全的函数映射
    _functions: Dict[str, Any] = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'ceil': math.ceil,
        'floor': math.floor,
        'pi': math.pi,
        'e': math.e,
        'pow': pow,
        'factorial': math.factorial,
        'gcd': math.gcd,
    }

    def __init__(self):
        super().__init__()
        self.examples = [
            "问：计算2加3等于多少？答：[TOOL:calc] 2+3 [/TOOL] 结果是5。",
            "问：100除以7是多少？答：[TOOL:calc] 100/7 [/TOOL] 结果约等于14.29。",
            "问：2的10次方是多少？答：[TOOL:calc] 2**10 [/TOOL] 结果是1024。",
            "问：sin(π/2)的值是多少？答：[TOOL:calc] sin(pi/2) [/TOOL] 结果是1.0。",
            "问：根号144是多少？答：[TOOL:calc] sqrt(144) [/TOOL] 结果是12.0。",
            "问：3的阶乘是多少？答：[TOOL:calc] factorial(3) [/TOOL] 结果是6。",
            "问：log100的值是多少？答：[TOOL:calc] log10(100) [/TOOL] 结果是2.0。",
            "问：(15+25)*2等于多少？答：[TOOL:calc] (15+25)*2 [/TOOL] 结果是80。",
        ]

    def _safe_eval(self, node):
        """安全地递归求值AST节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Name):
            if node.id in self._functions:
                return self._functions[node.id]
            raise ValueError(f"未知变量: {node.id}")
        elif isinstance(node, ast.Call):
            func = self._safe_eval(node.func)
            args = [self._safe_eval(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    def call(self, args: str) -> str:
        """执行数学计算"""
        expr = args.strip()
        if not expr:
            return "错误: 空表达式"
        try:
            tree = ast.parse(expr, mode='eval')
            result = self._safe_eval(tree)
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                return f"{result:.6g}"
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"
