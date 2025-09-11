"""验证页面模块"""

from __future__ import annotations

import gradio as gr


def create_val_tab() -> None:
    """创建验证标签页"""
    gr.Markdown("## 模型验证\n在这里验证训练好的模型性能。")

    # TODO: 实现验证功能
    gr.Markdown("验证功能正在开发中...")


# 保持向后兼容性
def render() -> None:
    """向后兼容的渲染函数"""
    create_val_tab()
