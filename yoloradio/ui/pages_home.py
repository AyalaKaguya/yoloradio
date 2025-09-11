"""首页模块 - 显示项目概览和目录状态"""

from __future__ import annotations

import gradio as gr

from ..core import get_model_details, list_dir
from ..core.paths import (
    DATASETS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    MODELS_PRETRAINED_DIR,
    MODELS_TRAINED_DIR,
    PROJECT_DIR,
)

try:
    import ultralytics as _ultra

    ULTRA_VERSION = getattr(_ultra, "__version__", "unknown")
except Exception:
    ULTRA_VERSION = "not installed"


def create_home_tab() -> None:
    """创建首页标签页"""
    gr.Markdown(
        f"""
        # yoloradio
        - 工作目录: `{PROJECT_DIR}`  
        - 数据集目录: `{DATASETS_DIR}`  
        - 模型目录: `{MODELS_DIR}`  
        - Ultralytics 版本: `{ULTRA_VERSION}`  
        - 日志目录(ultralytics): `{LOGS_DIR}`

        如需使用，请将数据集放入 `Datasets/`，模型文件放入 `Models/`。
        """
    )

    refresh_btn = gr.Button("刷新目录概览", variant="secondary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 数据集概览")
            datasets_md = gr.Markdown(value="加载中…")
        with gr.Column():
            gr.Markdown("### 模型概览")
            models_md = gr.Markdown(value="加载中…")
    gr.Markdown("### 日志概览（ultralytics runs/*）")
    logs_md = gr.Markdown(value="加载中…")

    def refresh_overview():
        """刷新概览信息"""
        ds_list = list_dir(DATASETS_DIR)
        # 获取详细模型信息
        pre_details = get_model_details(MODELS_PRETRAINED_DIR)
        tr_details = get_model_details(MODELS_TRAINED_DIR)
        logs_list = list_dir(LOGS_DIR)

        # 格式化模型信息
        pre_lines = []
        for name, desc, created in pre_details:
            pre_lines.append(f"- **{name}** - {desc} _(创建: {created})_")
        if not pre_lines:
            pre_lines = ["- (无预训练模型)"]

        tr_lines = []
        for name, desc, created in tr_details:
            tr_lines.append(f"- **{name}** - {desc} _(创建: {created})_")
        if not tr_lines:
            tr_lines = ["- (无训练模型)"]

        models_md = (
            "预训练模型:\n"
            + "\n".join(pre_lines)
            + "\n\n训练模型:\n"
            + "\n".join(tr_lines)
        )
        return (
            "\n".join([f"- {line}" for line in ds_list]),
            models_md,
            "\n".join([f"- {line}" for line in logs_list]),
        )

    # 初始加载
    ds0, m0, lg0 = refresh_overview()
    datasets_md.value = ds0
    models_md.value = m0
    logs_md.value = lg0

    refresh_btn.click(fn=refresh_overview, outputs=[datasets_md, models_md, logs_md])


# 保持向后兼容性
def render() -> None:
    """向后兼容的渲染函数"""
    create_home_tab()
