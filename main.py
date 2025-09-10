"""YoloRadio - YOLO 可视化训练平台主入口"""

from __future__ import annotations

import gradio as gr

from yoloradio.pages_datasets import render as datasets_render
from yoloradio.pages_export import render as export_render
from yoloradio.pages_home import render as home_render
from yoloradio.pages_models import render as models_render
from yoloradio.pages_quick import render as quick_render
from yoloradio.pages_train import render as train_render
from yoloradio.pages_val import render as val_render


def create_app() -> gr.Blocks:
    """创建 Gradio 应用实例"""
    app = gr.Blocks(
        title="YoloRadio - YOLO 可视化训练平台",
    )

    # Render homepage at root path
    with app:
        home_render()

    with app.route(name="数据集", path="datasets"):
        datasets_render()

    with app.route(name="模型", path="models"):
        models_render()

    with app.route(name="训练", path="train"):
        train_render()

    with app.route(name="验证", path="val"):
        val_render()

    with app.route(name="导出", path="export"):
        export_render()

    with app.route(name="快速应用", path="quick"):
        quick_render()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
