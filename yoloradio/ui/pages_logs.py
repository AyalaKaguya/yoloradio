"""日志页面 - 空页面模板"""

from __future__ import annotations

import gradio as gr


def create_logs_tab() -> None:
    """创建日志标签页"""
    gr.Markdown("## 日志\n这里是日志页面，功能待开发。")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🚧 功能开发中")
            gr.Markdown("此页面正在开发中，敬请期待...")

            # 占位内容
            gr.Textbox(
                label="占位文本框",
                placeholder="这里将来会有日志相关功能...",
                interactive=False,
            )
