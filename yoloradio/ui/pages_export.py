"""导出页面模块"""

from __future__ import annotations

import gradio as gr


def create_export_tab() -> None:
    """创建导出标签页"""
    gr.Markdown("## 模型导出\n在这里将训练好的模型导出为不同格式。")

    # TODO: 实现导出功能
    gr.Markdown("导出功能正在开发中...")


# 保持向后兼容性
def render() -> None:
    """向后兼容的渲染函数"""
    create_export_tab()
