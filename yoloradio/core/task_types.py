"""统一的任务类型定义模块

此模块定义了YOLO系统中使用的所有任务类型，
确保数据集、模型和训练之间的任务类型一致性。
"""

from __future__ import annotations

# 统一的任务类型映射
TASK_MAP = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}

# 反向映射：从代码到显示名称
TASK_CODE_TO_DISPLAY = {v: k for k, v in TASK_MAP.items()}

# 默认任务类型
DEFAULT_TASK = "detect"
DEFAULT_TASK_DISPLAY = "目标检测"


def get_task_code(display_name: str) -> str:
    """根据显示名称获取任务代码"""
    return TASK_MAP.get(display_name, DEFAULT_TASK)


def get_task_display(task_code: str) -> str:
    """根据任务代码获取显示名称"""
    return TASK_CODE_TO_DISPLAY.get(task_code, DEFAULT_TASK_DISPLAY)


def get_all_task_displays() -> list[str]:
    """获取所有任务显示名称"""
    return list(TASK_MAP.keys())


def get_all_task_codes() -> list[str]:
    """获取所有任务代码"""
    return list(TASK_MAP.values())


def is_valid_task_code(task_code: str) -> bool:
    """检查任务代码是否有效"""
    return task_code in TASK_CODE_TO_DISPLAY


def is_valid_task_display(display_name: str) -> bool:
    """检查任务显示名称是否有效"""
    return display_name in TASK_MAP


__all__ = [
    "TASK_MAP",
    "TASK_CODE_TO_DISPLAY",
    "DEFAULT_TASK",
    "DEFAULT_TASK_DISPLAY",
    "get_task_code",
    "get_task_display",
    "get_all_task_displays",
    "get_all_task_codes",
    "is_valid_task_code",
    "is_valid_task_display",
]
