from __future__ import annotations

import gradio as gr
from .paths import PROJECT_DIR, DATASETS_DIR, MODELS_DIR, LOGS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR

try:
    import ultralytics as _ultra
    ULTRA_VERSION = getattr(_ultra, "__version__", "unknown")
except Exception:
    ULTRA_VERSION = "not installed"

from .utils import list_dir


def render() -> None:
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
        ds_list = list_dir(DATASETS_DIR)
        pre_list = list_dir(MODELS_PRETRAINED_DIR, exts=(".pt", ".onnx", ".engine", ".xml", ".bin"))
        tr_list = list_dir(MODELS_TRAINED_DIR, exts=(".pt", ".onnx", ".engine", ".xml", ".bin"))
        logs_list = list_dir(LOGS_DIR)
        models_md = (
            "预训练模型:\n" + "\n".join([f"- {line}" for line in pre_list]) +
            "\n\n训练模型:\n" + "\n".join([f"- {line}" for line in tr_list])
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
