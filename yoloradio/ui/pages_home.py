"""首页模块 - 显示项目概览和目录状态"""

from __future__ import annotations

import gradio as gr

from ..core import dataset_manager, list_dir, model_manager
from ..core.paths import DATASETS_DIR, LOGS_DIR, MODELS_DIR, PROJECT_DIR

try:
    import ultralytics as _ultra

    ULTRA_VERSION = getattr(_ultra, "__version__", "unknown")
except Exception:
    ULTRA_VERSION = "not installed"


def create_home_tab() -> None:
    """创建首页标签页"""
    gr.Markdown(
        f"""
        # yoloradio 主页
        
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
        with gr.Column():
            gr.Markdown("### 日志概览（ultralytics runs/*）")
            logs_md = gr.Markdown(value="加载中…")

    def refresh_overview():
        """刷新概览信息"""
        # 使用数据集管理器获取数据集信息
        datasets = dataset_manager.list_datasets()
        ds_lines = []
        if datasets:
            for ds in datasets:
                stats = ds.get_statistics()
                total_images = stats.get("total_images", 0)
                desc = (
                    ds.description[:30] + "..."
                    if len(ds.description) > 30
                    else ds.description
                )
                desc = desc or "无描述"
                ds_lines.append(
                    f"- **{ds.name}** ({ds.dataset_type_display}) - {total_images}张图片 - {desc}"
                )
        else:
            ds_lines = ["- (无数据集)"]

        # 使用模型管理器获取模型信息
        pretrained_models = model_manager.list_models(is_pretrained=True)
        trained_models = model_manager.list_models(is_pretrained=False)

        # 格式化预训练模型信息
        pre_lines = []
        for model in pretrained_models:
            desc = (
                model.description[:30] + "..."
                if len(model.description) > 30
                else model.description
            )
            desc = desc or "无描述"
            created = (
                model.created_time.strftime("%m-%d %H:%M")
                if model.created_time
                else "未知"
            )
            task_display = model.task_display or "未知任务"
            pre_lines.append(
                f"- **{model.filename}** ({task_display}) - {desc} _(创建: {created})_"
            )
        if not pre_lines:
            pre_lines = ["- (无预训练模型)"]

        # 格式化训练模型信息
        tr_lines = []
        for model in trained_models:
            desc = (
                model.description[:30] + "..."
                if len(model.description) > 30
                else model.description
            )
            desc = desc or "无描述"
            created = (
                model.created_time.strftime("%m-%d %H:%M")
                if model.created_time
                else "未知"
            )
            task_display = model.task_display or "未知任务"
            tr_lines.append(
                f"- **{model.filename}** ({task_display}) - {desc} _(创建: {created})_"
            )
        if not tr_lines:
            tr_lines = ["- (无训练模型)"]

        # 获取日志信息
        logs_list = list_dir(LOGS_DIR)

        models_overview = (
            "预训练模型:\n"
            + "\n".join(pre_lines)
            + "\n\n训练模型:\n"
            + "\n".join(tr_lines)
        )

        return (
            "\n".join(ds_lines),
            models_overview,
            "\n".join([f"- {line}" for line in logs_list]),
        )

    # 初始加载
    ds0, m0, lg0 = refresh_overview()
    datasets_md.value = ds0
    models_md.value = m0
    logs_md.value = lg0

    refresh_btn.click(fn=refresh_overview, outputs=[datasets_md, models_md, logs_md])
