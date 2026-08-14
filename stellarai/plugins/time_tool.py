"""
时间工具插件
============
获取当前时间、日期、时间计算
"""

from datetime import datetime, timedelta
from .base import Plugin


class TimePlugin(Plugin):
    """时间工具: 获取当前时间、日期、时间差计算"""

    name = "time"
    description = "时间工具，获取当前日期时间、计算日期差"

    def __init__(self):
        super().__init__()
        self.examples = [
            "问：现在几点？答：[TOOL:time] now [/TOOL] 当前时间是2024年1月1日 12:00:00。",
            "问：今天星期几？答：[TOOL:time] weekday [/TOOL] 今天是星期一。",
            "问：今天日期？答：[TOOL:time] date [/TOOL] 今天是2024年1月1日。",
            "问：3天后是几号？答：[TOOL:time] +3 days [/TOOL] 3天后是2024年1月4日。",
        ]

    def call(self, args: str) -> str:
        """执行时间查询"""
        cmd = args.strip().lower()

        now = datetime.now()

        if cmd == "now":
            return f"当前时间是{now.strftime('%Y年%m月%d日 %H:%M:%S')}。"

        elif cmd == "date":
            return f"今天是{now.strftime('%Y年%m月%d日')}。"

        elif cmd == "weekday":
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            return f"今天是{weekdays[now.weekday()]}。"

        elif cmd == "time":
            return f"现在是{now.strftime('%H:%M:%S')}。"

        elif cmd.startswith("+") or cmd.startswith("-"):
            # 日期偏移: +3 days, -1 days
            try:
                parts = cmd.split()
                offset = int(parts[0])
                unit = parts[1] if len(parts) > 1 else "days"
                if unit in ("day", "days", "天"):
                    future = now + timedelta(days=offset)
                    return f"{abs(offset)}天{'后' if offset > 0 else '前'}是{future.strftime('%Y年%m月%d日')}。"
                elif unit in ("hour", "hours", "小时"):
                    future = now + timedelta(hours=offset)
                    return f"{abs(offset)}小时{'后' if offset > 0 else '前'}是{future.strftime('%Y年%m月%d日 %H:%M')}。"
                else:
                    return f"不支持的时间单位: {unit}"
            except (ValueError, IndexError):
                return "用法: +N days 或 -N days"

        else:
            return f"当前时间是{now.strftime('%Y年%m月%d日 %H:%M:%S')}。可用命令: now, date, time, weekday, +N days"
