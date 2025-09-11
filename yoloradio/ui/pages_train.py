"""训练页面模块"""

from __future__ import annotations

import gradio as gr

from ..core import (
    clear_training_logs,
    get_device_info,
    get_training_logs,
    get_training_status,
    list_datasets_for_task,
    list_models_for_task,
    pause_training,
    resume_training,
    start_training,
    stop_training,
    validate_training_environment,
)

# 数据集类型映射
DATASET_TYPE_MAP = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}


def create_train_tab() -> None:
    """创建训练标签页"""
    gr.Markdown("## 模型训练\n在这里配置训练参数，监控训练状态，查看历史训练日志。")

    # 初始可选项（默认按目标检测）
    default_task_display = "目标检测"
    default_task_code = DATASET_TYPE_MAP.get(default_task_display, "detect")
    init_ds = list_datasets_for_task(default_task_code)
    init_models = list_models_for_task(default_task_code)  # (label, path)
    init_model_labels = [m[0] for m in init_models]

    with gr.Tabs():
        with gr.Tab("配置"):
            with gr.Row():
                # 左：任务/数据集/模型
                with gr.Column(scale=1, min_width=320):
                    task_dd = gr.Dropdown(
                        choices=list(DATASET_TYPE_MAP.keys()),
                        value=default_task_display,
                        label="任务类型",
                    )
                    refresh_btn = gr.Button("刷新可选项", variant="secondary")
                    ds_dd = gr.Dropdown(
                        choices=init_ds,
                        value=(init_ds[0] if init_ds else None),
                        label="数据集",
                    )
                    mdl_dd = gr.Dropdown(
                        choices=init_model_labels,
                        value=(init_model_labels[0] if init_model_labels else None),
                        label="模型",
                    )

                # 右：训练参数
                with gr.Column(scale=2):
                    with gr.Row():
                        epochs_num = gr.Number(
                            label="训练轮数", value=10, minimum=1, maximum=1000
                        )
                        lr0_num = gr.Number(
                            label="学习率", value=0.01, minimum=0.0001, maximum=1.0
                        )
                    with gr.Row():
                        imgsz_num = gr.Number(
                            label="图像尺寸", value=640, minimum=320, maximum=1280
                        )
                        batch_num = gr.Number(
                            label="批大小", value=16, minimum=1, maximum=128
                        )
                    device_dd = gr.Dropdown(
                        choices=["auto", "cpu", "0", "1", "2", "3"],
                        value="auto",
                        label="设备选择",
                    )

            # 训练控制
            with gr.Row():
                start_btn = gr.Button("开始训练", variant="primary", size="lg")
                pause_btn = gr.Button("暂停", variant="secondary", size="lg")
                resume_btn = gr.Button("恢复", variant="secondary", size="lg")
                stop_btn = gr.Button("停止", variant="stop", size="lg")

            # 状态显示
            status_md = gr.Markdown("训练未开始")

        with gr.Tab("监控"):
            with gr.Column():
                monitor_refresh_btn = gr.Button("刷新状态", variant="secondary")

                # 训练状态概览
                with gr.Row():
                    with gr.Column():
                        progress_md = gr.Markdown("进度: 等待中...")
                        device_info_md = gr.Markdown("设备信息: 检测中...")
                    with gr.Column():
                        epoch_md = gr.Markdown("轮次: 0/0")
                        run_id_md = gr.Markdown("运行ID: 无")

                # 实时日志
                gr.Markdown("### 训练日志")
                with gr.Row():
                    logs_refresh_btn = gr.Button("刷新日志", variant="secondary")
                    clear_logs_btn = gr.Button("清空日志", variant="secondary")

                logs_textbox = gr.Textbox(
                    label="实时日志",
                    lines=20,
                    max_lines=30,
                    value="等待训练开始...",
                    interactive=False,
                    show_copy_button=True,
                )

        with gr.Tab("环境检查"):
            env_check_btn = gr.Button("检查训练环境", variant="primary")
            env_result = gr.Markdown("点击按钮检查训练环境")

    # 事件处理函数
    def _refresh_choices(task_display: str):
        """刷新数据集和模型选择"""
        task_code = DATASET_TYPE_MAP.get(task_display, "detect")
        datasets = list_datasets_for_task(task_code)
        models = list_models_for_task(task_code)
        model_labels = [m[0] for m in models]

        return (
            gr.update(choices=datasets, value=datasets[0] if datasets else None),
            gr.update(
                choices=model_labels, value=model_labels[0] if model_labels else None
            ),
        )

    def _start_training(
        task: str,
        dataset: str,
        model_label: str,
        epochs: int,
        lr0: float,
        imgsz: int,
        batch: int,
        device: str,
    ):
        """开始训练"""
        if not dataset or not model_label:
            return "请选择数据集和模型"

        # 从模型标签中找到对应的路径
        task_code = DATASET_TYPE_MAP.get(task, "detect")
        models = list_models_for_task(task_code)
        model_path = None
        for label, path in models:
            if label == model_label:
                model_path = path
                break

        if not model_path:
            return "找不到选中的模型文件"

        ok, msg = start_training(
            task_code=task_code,
            dataset_name=dataset,
            model_path=model_path,
            epochs=int(epochs),
            lr0=float(lr0),
            imgsz=int(imgsz),
            batch=int(batch),
            device=device,
        )

        return msg

    def _get_training_status():
        """获取训练状态"""
        status = get_training_status()

        # 格式化状态信息
        if status["is_running"]:
            progress_text = f"进度: {status['progress']:.1f}% ({status['current_epoch']}/{status['total_epochs']})"
            status_text = "🟢 训练进行中" + (
                f" (已暂停)" if status["is_paused"] else ""
            )
        else:
            progress_text = "进度: 等待中..."
            status_text = "⚪ 训练未开始"

        epoch_text = f"轮次: {status['current_epoch']}/{status['total_epochs']}"
        run_id_text = f"运行ID: {status['run_id'] or '无'}"

        return progress_text, status_text, epoch_text, run_id_text

    def _get_logs():
        """获取训练日志"""
        logs = get_training_logs()
        if not logs:
            return "暂无日志"
        return "\n".join(logs[-100:])  # 只显示最新100行

    def _check_environment():
        """检查训练环境"""
        ok, msg = validate_training_environment()
        device_info = get_device_info()
        return f"环境检查结果:\n{msg}\n\n设备信息:\n{device_info}"

    # 绑定事件
    task_dd.change(fn=_refresh_choices, inputs=[task_dd], outputs=[ds_dd, mdl_dd])
    refresh_btn.click(fn=_refresh_choices, inputs=[task_dd], outputs=[ds_dd, mdl_dd])

    start_btn.click(
        fn=_start_training,
        inputs=[
            task_dd,
            ds_dd,
            mdl_dd,
            epochs_num,
            lr0_num,
            imgsz_num,
            batch_num,
            device_dd,
        ],
        outputs=[status_md],
    )

    pause_btn.click(fn=lambda: pause_training()[1], outputs=[status_md])
    resume_btn.click(fn=lambda: resume_training()[1], outputs=[status_md])
    stop_btn.click(fn=lambda: stop_training()[1], outputs=[status_md])

    monitor_refresh_btn.click(
        fn=_get_training_status,
        outputs=[progress_md, status_md, epoch_md, run_id_md],
    )

    logs_refresh_btn.click(fn=_get_logs, outputs=[logs_textbox])
    clear_logs_btn.click(
        fn=lambda: (clear_training_logs(), "日志已清空")[1], outputs=[logs_textbox]
    )

    env_check_btn.click(fn=_check_environment, outputs=[env_result])

    # 初始化设备信息
    device_info_md.value = get_device_info()


# 保持向后兼容性
def render() -> None:
    """向后兼容的渲染函数"""
    create_train_tab()
