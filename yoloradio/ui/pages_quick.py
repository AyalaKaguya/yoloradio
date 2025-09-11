"""快速推理页面模块"""

from __future__ import annotations

import gradio as gr


def create_quick_tab() -> None:
    """创建快速推理标签页"""
    gr.Markdown("## 快速推理\n在这里使用训练好的模型进行快速推理测试。")

    # TODO: 实现快速推理功能
    gr.Markdown("快速推理功能正在开发中...")
