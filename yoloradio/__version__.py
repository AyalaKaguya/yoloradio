"""YoloRadio 版本信息"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AyalaKaguya"
__email__ = ""
__description__ = "YOLO 可视化训练平台 - 基于 Gradio 和 Ultralytics"
__url__ = "https://github.com/AyalaKaguya/yoloradio"

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "url": __url__,
}


def get_version() -> str:
    """获取版本号"""
    return __version__


def get_version_info() -> dict[str, str]:
    """获取完整版本信息"""
    return VERSION_INFO.copy()
