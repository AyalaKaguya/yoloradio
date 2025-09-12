"""YoloRadio - YOLO 可视化训练平台主入口"""

from __future__ import annotations

import gradio as gr

from yoloradio.ui import (
    create_datasets_tab,
    create_export_tab,
    create_home_tab,
    create_logs_tab,
    create_models_tab,
    create_quick_tab,
    create_test_tab,
    create_train_tab,
    create_val_tab,
)


def create_app() -> gr.Blocks:
    """创建 Gradio 应用实例"""
    app = gr.Blocks(
        title="YoloRadio - YOLO 可视化训练平台",
    )

    # Render homepage at root path
    with app:
        create_home_tab()

    with app.route(name="数据集", path="datasets"):
        create_datasets_tab()

    with app.route(name="模型", path="models"):
        create_models_tab()

    with app.route(name="训练", path="train"):
        create_train_tab()

    with app.route(name="验证", path="val"):
        create_val_tab()

    with app.route(name="日志", path="logs"):
        create_logs_tab()

    with app.route(name="导出", path="export"):
        create_export_tab()

    with app.route(name="测试", path="test"):
        create_test_tab()

    with app.route(name="快速应用", path="quick"):
        create_quick_tab()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
