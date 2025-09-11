"""验证页面模块"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gradio as gr

from ..core import dataset_manager, model_manager
from ..core.task_manager import TaskPriority
from ..core.task_types import get_all_task_displays, get_task_code
from ..core.train_manager import TrainManager


def create_val_tab() -> None:
    """创建验证标签页"""
    gr.Markdown("## 模型验证\n验证训练好的模型在测试数据集上的性能。")

    # 初始化训练管理器
    train_manager = TrainManager()

    # 获取所有可用的任务类型
    all_task_displays = get_all_task_displays()
    default_task_display = "目标检测"

    with gr.Row():
        # 左侧：验证配置
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### ⚙️ 验证配置")

            with gr.Group():
                gr.Markdown("#### 基础设置")

                # 任务类型选择
                task_dropdown = gr.Dropdown(
                    choices=all_task_displays,
                    value=default_task_display,
                    label="验证任务类型",
                    info="选择验证的任务类型，这将决定可用的模型和数据集",
                )

                # 模型选择 - 初始为空，将根据任务类型动态更新
                model_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="模型",
                    info="选择要验证的模型",
                )

                # 数据集选择 - 初始为空，将根据任务类型动态更新
                dataset_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="验证数据集",
                    info="选择用于验证的数据集",
                )

                # 任务优先级
                priority_dropdown = gr.Dropdown(
                    choices=["低", "普通", "高", "紧急"],
                    value="普通",
                    label="任务优先级",
                    info="设置验证任务的优先级",
                )

            with gr.Group():
                gr.Markdown("#### 验证参数")

                # 置信度阈值
                conf_slider = gr.Slider(
                    0.01,
                    1.0,
                    value=0.25,
                    step=0.01,
                    label="置信度阈值 (conf)",
                    info="检测结果的最低置信度",
                )

                # IoU阈值
                iou_slider = gr.Slider(
                    0.01,
                    1.0,
                    value=0.5,
                    step=0.01,
                    label="IoU阈值 (iou)",
                    info="NMS的IoU阈值",
                )

                # 图像尺寸
                imgsz_slider = gr.Slider(
                    320,
                    2048,
                    value=640,
                    step=32,
                    label="图像尺寸 (imgsz)",
                    info="输入图像尺寸",
                )

                # 批大小
                batch_slider = gr.Slider(
                    1, 128, value=32, step=1, label="批大小 (batch)", info="验证批大小"
                )

            with gr.Group():
                gr.Markdown("#### 高级设置")

                # 设备选择
                device_dropdown = gr.Dropdown(
                    choices=["auto", "cpu", "cuda:0"],
                    value="auto",
                    label="设备",
                    info="选择运行设备",
                )

                # 工作进程数
                workers_slider = gr.Slider(
                    1,
                    16,
                    value=8,
                    step=1,
                    label="工作进程数 (workers)",
                    info="数据加载的工作进程数",
                )

            # 控制按钮
            with gr.Row():
                start_btn = gr.Button("🚀 开始验证", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ 停止验证", variant="secondary")

        # 右侧：状态和结果
        with gr.Column(scale=2, min_width=600):
            gr.Markdown("### 📊 验证状态与结果")

            # 状态显示
            status_display = gr.Textbox(
                value="状态: 就绪", label="验证状态", interactive=False, lines=1
            )

            # 验证信息
            info_display = gr.Markdown("**验证信息**: 请配置验证参数并点击开始验证")

            # 日志显示
            logs_display = gr.Textbox(
                value="",
                label="验证日志",
                lines=20,
                interactive=False,
                max_lines=30,
                show_copy_button=True,
            )

    # 动态更新函数
    def update_models_and_datasets(task_display):
        """根据任务类型更新可用的模型和数据集"""
        try:
            task_code = get_task_code(task_display)

            # 获取该任务类型的模型
            available_models = model_manager.list_models_for_task_display(task_code)
            model_choices = [m[0] for m in available_models] if available_models else []
            model_value = model_choices[0] if model_choices else None

            # 获取该任务类型的数据集
            available_datasets = dataset_manager.list_datasets_with_type(task_code)
            dataset_choices = (
                [d.name for d in available_datasets] if available_datasets else []
            )
            dataset_value = dataset_choices[0] if dataset_choices else None

            return (
                gr.Dropdown(choices=model_choices, value=model_value),
                gr.Dropdown(choices=dataset_choices, value=dataset_value),
                f"已切换到任务类型: {task_display} ({task_code})\n可用模型: {len(model_choices)} 个\n可用数据集: {len(dataset_choices)} 个",
            )
        except Exception as e:
            return (
                gr.Dropdown(choices=[], value=None),
                gr.Dropdown(choices=[], value=None),
                f"❌ 更新失败: {str(e)}",
            )

    # 绑定任务类型变化事件
    task_dropdown.change(
        fn=update_models_and_datasets,
        inputs=[task_dropdown],
        outputs=[model_dropdown, dataset_dropdown, logs_display],
    )

    # 页面加载时初始化模型和数据集选择
    def initialize_defaults():
        """初始化默认的模型和数据集选择"""
        try:
            task_code = get_task_code(default_task_display)

            # 获取该任务类型的模型
            available_models = model_manager.list_models_for_task_display(task_code)
            model_choices = [m[0] for m in available_models] if available_models else []

            # 获取该任务类型的数据集
            available_datasets = dataset_manager.list_datasets_with_type(task_code)
            dataset_choices = (
                [d.name for d in available_datasets] if available_datasets else []
            )

            # 更新dropdown选择
            model_dropdown.choices = model_choices
            model_dropdown.value = model_choices[0] if model_choices else None

            dataset_dropdown.choices = dataset_choices
            dataset_dropdown.value = dataset_choices[0] if dataset_choices else None

            return f"已初始化: 找到 {len(model_choices)} 个模型, {len(dataset_choices)} 个数据集"

        except Exception as e:
            return f"❌ 初始化失败: {str(e)}"

    # 在页面加载后初始化（不直接赋值给组件）
    initial_log = initialize_defaults()

    # 状态更新函数
    def update_status_display():
        """更新状态显示"""
        try:
            # 获取当前所有任务
            all_tasks = train_manager.task_manager.get_all_tasks()
            validation_tasks = [
                t
                for t in all_tasks
                if hasattr(t, "task_type") and t.task_type.value == "validation"
            ]

            if validation_tasks:
                # 获取最近的验证任务
                current_task = validation_tasks[-1]
                progress = current_task.progress

                # 格式化状态信息
                status_text = f"状态: {current_task.status.value} - {progress.progress_percent:.1f}%"

                info_text = f"""**验证信息**:
- **任务ID**: {current_task.id}
- **任务名称**: {current_task.name}
- **状态**: {current_task.status.value}
- **进度**: {progress.progress_percent:.1f}%
- **准确率**: {progress.accuracy:.4f}
- **创建时间**: {current_task.created_at.strftime('%H:%M:%S')}"""

                if current_task.started_at:
                    info_text += f"\n- **开始时间**: {current_task.started_at.strftime('%H:%M:%S')}"

                if current_task.completed_at:
                    info_text += f"\n- **完成时间**: {current_task.completed_at.strftime('%H:%M:%S')}"

                # 显示最近的日志（最多50行）
                log_text = (
                    "\n".join(current_task.logs[-50:])
                    if current_task.logs
                    else "暂无日志"
                )

                return status_text, info_text, log_text
            else:
                return (
                    "状态: 就绪",
                    "**验证信息**: 请配置验证参数并点击开始验证",
                    initial_log,
                )

        except Exception as e:
            return (
                f"状态: 错误 - {str(e)}",
                "**验证信息**: 状态更新失败",
                f"❌ 更新错误: {str(e)}",
            )

    # 验证启动函数
    def start_validation_task(
        task_display,
        dataset,
        model_label,
        priority_str,
        conf,
        iou,
        imgsz,
        batch,
        device,
        workers,
    ):
        """启动验证任务"""
        try:
            # 验证必需参数
            if not dataset:
                return (
                    "❌ 请选择验证数据集",
                    "状态: 错误",
                    "错误: 必须选择一个验证数据集",
                )

            if not model_label:
                return "❌ 请选择验证模型", "状态: 错误", "错误: 必须选择一个验证模型"

            if not task_display:
                return "❌ 请选择任务类型", "状态: 错误", "错误: 必须选择一个任务类型"

            # 获取任务代码
            task_code = get_task_code(task_display)

            # 构建当前的模型映射
            available_models = model_manager.list_models_for_task_display(task_code)
            current_model_map = {m[0]: m[1] for m in available_models}

            # 验证模型是否存在
            if model_label not in current_model_map:
                return (
                    "❌ 选择的模型不存在",
                    "状态: 错误",
                    f"错误: 模型 '{model_label}' 不在可用模型列表中",
                )

            # 转换优先级
            priority_map = {
                "低": TaskPriority.LOW,
                "普通": TaskPriority.NORMAL,
                "高": TaskPriority.HIGH,
                "紧急": TaskPriority.URGENT,
            }
            priority = priority_map.get(priority_str, TaskPriority.NORMAL)

            # 启动验证
            message, status = train_manager.start_validation(
                dataset=dataset,
                model_label=model_label,
                conf=conf,
                iou=iou,
                imgsz=int(imgsz),
                batch=int(batch),
                device=device,
                workers=int(workers),
                model_map=current_model_map,
                priority=priority,
            )

            return (
                message,
                status,
                f"验证任务已启动\n任务类型: {task_display} ({task_code})\n参数配置:\n- 数据集: {dataset}\n- 模型: {model_label}\n- 置信度: {conf}\n- IoU: {iou}",
            )

        except Exception as e:
            error_msg = f"❌ 启动验证失败: {str(e)}"
            return error_msg, "状态: 错误", error_msg

    # 停止验证函数
    def stop_validation_task():
        """停止验证任务"""
        try:
            current = train_manager.get_current_task()
            if current and current.task_type.value == "validation":
                message, status = train_manager.stop_training(current.id)
                return message, status, f"已请求停止验证任务: {current.id}"
            else:
                return (
                    "❌ 没有正在运行的验证任务",
                    "状态: 就绪",
                    "当前没有验证任务在运行",
                )
        except Exception as e:
            error_msg = f"❌ 停止验证失败: {str(e)}"
            return error_msg, "状态: 错误", error_msg

    # 绑定事件
    start_btn.click(
        fn=start_validation_task,
        inputs=[
            task_dropdown,
            dataset_dropdown,
            model_dropdown,
            priority_dropdown,
            conf_slider,
            iou_slider,
            imgsz_slider,
            batch_slider,
            device_dropdown,
            workers_slider,
        ],
        outputs=[info_display, status_display, logs_display],
    )

    stop_btn.click(
        fn=stop_validation_task, outputs=[info_display, status_display, logs_display]
    )

    # 添加状态监控回调
    def create_status_callback():
        """创建状态更新回调"""

        def callback(status_info):
            # 在Gradio中实时更新需要使用事件系统
            pass

        return callback

    # 注册回调（用于实时更新）
    train_manager.add_status_callback(create_status_callback())

    # 添加定期刷新 - 使用Gradio的定时器
    def refresh_status():
        """定期刷新状态"""
        return update_status_display()

    # 创建控制按钮
    with gr.Row():
        refresh_btn = gr.Button("🔄 刷新状态", variant="secondary", size="sm")
        update_btn = gr.Button("🔄 更新选择", variant="secondary", size="sm")
        auto_refresh_btn = gr.Button("🔄 自动刷新", variant="secondary", size="sm")

    # 绑定刷新事件
    refresh_btn.click(
        fn=refresh_status, outputs=[status_display, info_display, logs_display]
    )

    # 手动更新模型和数据集选择
    update_btn.click(
        fn=update_models_and_datasets,
        inputs=[task_dropdown],
        outputs=[model_dropdown, dataset_dropdown, logs_display],
    )

    # 自动刷新状态（每3秒）
    auto_refresh_timer = gr.Timer(3.0)
    auto_refresh_timer.tick(
        fn=refresh_status, outputs=[status_display, info_display, logs_display]
    )

    # 启动初始刷新
    auto_refresh_btn.click(
        fn=refresh_status, outputs=[status_display, info_display, logs_display]
    )
