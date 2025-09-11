"""增强的训练页面 - 支持任务队列和动态状态面板"""

from __future__ import annotations

import gradio as gr

from ..core import (
    TASK_MAP,
    TaskPriority,
    dataset_manager,
    model_manager,
    task_manager,
    train_manager,
)


def create_train_tab() -> None:
    """创建增强的训练标签页"""
    gr.Markdown("## 模型训练\n在这里配置训练参数，管理训练任务队列，监控训练状态。")

    # 初始可选项（默认按目标检测）
    default_task_display = "目标检测"
    default_task_code = TASK_MAP.get(default_task_display, "detect")
    init_ds = list(
        map(
            lambda d: d.name, dataset_manager.list_datasets_with_type(default_task_code)
        )
    )
    init_models = model_manager.list_models_for_task_display(default_task_code)
    init_model_labels = [m[0] for m in init_models]

    with gr.Tabs():
        with gr.Tab("任务配置"):
            with gr.Row():
                # 左：任务/数据集/模型
                with gr.Column(scale=1, min_width=320):
                    # 添加任务按钮
                    add_task_btn = gr.Button(
                        "添加到训练队列", variant="primary", size="lg"
                    )

                    task_dd = gr.Dropdown(
                        choices=list(TASK_MAP.keys()),
                        value=default_task_display,
                        label="任务类型",
                    )
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

                    # 任务优先级
                    priority_dd = gr.Dropdown(
                        choices=["普通", "高", "紧急"],
                        value="普通",
                        label="任务优先级",
                    )

                # 中：核心超参 + 高级折叠
                with gr.Column(scale=1, min_width=320):
                    epochs_in = gr.Slider(
                        1, 1000, value=100, step=1, label="训练轮次 epochs"
                    )
                    lr_in = gr.Number(value=0.01, label="初始学习率 lr0")
                    imgsz_in = gr.Slider(
                        256, 2048, value=640, step=32, label="图像尺寸 imgsz"
                    )
                    batch_in = gr.Slider(1, 256, value=16, step=1, label="批大小 batch")

                    with gr.Accordion("数据增强参数", open=False):
                        degrees_in = gr.Slider(
                            0, 45, value=0.0, step=0.5, label="旋转 degrees"
                        )
                        translate_in = gr.Slider(
                            0, 0.5, value=0.1, step=0.01, label="平移 translate"
                        )
                        scale_in = gr.Slider(
                            0.0, 2.0, value=0.5, step=0.05, label="缩放 scale"
                        )
                        shear_in = gr.Slider(
                            0, 10, value=0.0, step=0.5, label="剪切 shear"
                        )
                        fliplr_in = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05, label="左右翻转概率 fliplr"
                        )
                        flipud_in = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05, label="上下翻转概率 flipud"
                        )
                        mosaic_in = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.05, label="mosaic"
                        )
                        mixup_in = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05, label="mixup"
                        )

                    with gr.Accordion("训练器参数", open=False):
                        optimizer_in = gr.Radio(
                            choices=["auto", "SGD", "Adam", "AdamW"],
                            value="auto",
                            label="优化器 optimizer",
                        )
                        momentum_in = gr.Slider(
                            0.0, 1.0, value=0.937, step=0.001, label="动量 momentum"
                        )
                        weight_decay_in = gr.Number(
                            value=0.0005, label="权重衰减 weight_decay"
                        )
                        device_in = gr.Textbox(value="auto", label="设备 device")
                        workers_in = gr.Slider(
                            0, 16, value=8, step=1, label="DataLoader 线程 workers"
                        )

                # 右：实时 TOML（使用 yaml 语法高亮）
                with gr.Column(scale=1, min_width=360):
                    toml_preview = gr.Code(
                        language="yaml", value="", label="TOML 预览", interactive=False
                    )

            # 内部状态：任务代码 + 模型标签->路径 映射
            st_task_code = gr.State(default_task_code)
            st_model_map = gr.State({lbl: path for lbl, path in init_models})

        with gr.Tab("任务队列"):
            with gr.Row():
                # 左侧：队列控制
                with gr.Column(scale=1):
                    gr.Markdown("### 队列控制")

                    # 队列状态概览
                    queue_status = gr.Markdown("**队列状态**: 加载中...")

                    with gr.Row():
                        refresh_queue_btn = gr.Button("刷新队列", variant="secondary")
                        clear_completed_btn = gr.Button(
                            "清理已完成", variant="secondary"
                        )

                    # 任务操作
                    gr.Markdown("### 任务操作")
                    selected_task_id = gr.Textbox(
                        label="任务ID", placeholder="输入要操作的任务ID"
                    )

                    with gr.Row():
                        promote_btn = gr.Button("提升优先级", variant="primary")
                        cancel_btn = gr.Button("取消任务", variant="stop")

                # 右侧：任务列表
                with gr.Column(scale=2):
                    gr.Markdown("### 任务列表")
                    task_list = gr.Dataframe(
                        headers=["ID", "名称", "状态", "优先级", "进度", "创建时间"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        label="训练任务",
                        interactive=False,
                        wrap=True,
                    )

        with gr.Tab("实时监控"):
            with gr.Row():
                # 左侧：当前任务状态
                with gr.Column(scale=1):
                    gr.Markdown("### 当前任务")

                    # 自动环境检查
                    env_status_md = gr.Markdown("🔍 正在检查训练环境...")

                    current_task_status = gr.Markdown("状态: 就绪")
                    current_task_info = gr.Markdown("等待任务开始...")

                    # 任务控制
                    with gr.Row():
                        stop_current_btn = gr.Button("停止当前任务", variant="stop")
                        pause_queue_btn = gr.Button("暂停队列", variant="secondary")

                    # 设备信息
                    device_info_md = gr.Markdown(
                        f"**设备信息**: {train_manager.get_device_info()}"
                    )

                    # 实时指标面板
                    gr.Markdown("### 实时指标")
                    with gr.Row():
                        loss_display = gr.Number(label="当前Loss", interactive=False)
                        acc_display = gr.Number(label="准确率", interactive=False)
                        progress_display = gr.Number(label="进度%", interactive=False)

                        clear_logs_btn = gr.Button("清空日志", variant="secondary")

                # 右侧：实时日志
                with gr.Column(scale=2):
                    gr.Markdown("### 实时日志")
                    log_output = gr.Textbox(
                        label="训练输出",
                        lines=25,
                        max_lines=30,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                    )

    # 辅助函数
    def _to_task_code(display: str) -> str:
        return TASK_MAP.get(display, "detect")

    def _priority_to_enum(priority_str: str) -> TaskPriority:
        mapping = {
            "普通": TaskPriority.NORMAL,
            "高": TaskPriority.HIGH,
            "紧急": TaskPriority.URGENT,
        }
        return mapping.get(priority_str, TaskPriority.NORMAL)

    def _refresh_options(task_display: str):
        code = _to_task_code(task_display)
        ds = list(map(lambda d: d.name, dataset_manager.list_datasets_with_type(code)))
        models = model_manager.list_models_for_task_display(code)
        labels = [m[0] for m in models]
        ds_val = ds[0] if ds else None
        mdl_val = labels[0] if labels else None
        return (
            gr.update(choices=ds, value=ds_val),
            gr.update(choices=labels, value=mdl_val),
            code,
            {lbl: path for lbl, path in models},
        )

    def _make_toml_wrapper(*args):
        return train_manager.create_training_config(*args)

    def _add_task(
        task_display: str,
        dataset: str,
        model_label: str,
        epochs: int,
        lr0: float,
        imgsz: int,
        batch: int,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        fliplr: float,
        flipud: float,
        mosaic: float,
        mixup: float,
        optimizer: str,
        momentum: float,
        weight_decay: float,
        device: str,
        workers: int,
        priority_str: str,
        task_code: str,
        model_map: dict,
    ):
        priority = _priority_to_enum(priority_str)

        return train_manager.start_training(
            task_display=task_display,
            dataset=dataset,
            model_label=model_label,
            epochs=epochs,
            lr0=lr0,
            imgsz=imgsz,
            batch=batch,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fliplr=fliplr,
            flipud=flipud,
            mosaic=mosaic,
            mixup=mixup,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            workers=workers,
            model_map=model_map,
            priority=priority,
        )

    def _refresh_task_list():
        """刷新任务列表"""
        tasks = train_manager.get_all_tasks()
        data = []
        for task in tasks:
            data.append(
                [
                    task.id,
                    task.name,
                    task.status.value,
                    task.priority.name,
                    f"{task.progress.progress_percent:.1f}%",
                    task.created_at.strftime("%H:%M:%S"),
                ]
            )
        return data

    def _refresh_queue_status():
        """刷新队列状态"""
        status = train_manager.get_task_queue_status()
        status_text = f"""**队列状态**:
🔄 排队: {status['queued']} | ▶️ 运行: {status['running']} | ✅ 完成: {status['completed']} | ❌ 失败: {status['failed']}
📊 总任务: {status['total_tasks']} | 🎯 当前: {status['current_task'] or '无'}"""
        return status_text

    def _update_current_task_display():
        """更新当前任务显示"""
        return train_manager.get_current_status()

    def _promote_task(task_id: str):
        """提升任务优先级"""
        if not task_id.strip():
            return "❌ 请输入任务ID"

        success = train_manager.promote_task(task_id)
        if success:
            return f"✅ 任务 {task_id} 已提升优先级"
        else:
            return f"❌ 无法提升任务 {task_id} 的优先级"

    def _cancel_task(task_id: str):
        """取消任务"""
        if not task_id.strip():
            return "❌ 请输入任务ID"

        info, status = train_manager.stop_training(task_id)
        return info

    # 事件绑定
    task_dd.change(
        fn=_refresh_options,
        inputs=[task_dd],
        outputs=[ds_dd, mdl_dd, st_task_code, st_model_map],
    )

    # TOML预览更新
    inputs_for_toml = [
        task_dd,
        ds_dd,
        mdl_dd,
        epochs_in,
        lr_in,
        imgsz_in,
        batch_in,
        degrees_in,
        translate_in,
        scale_in,
        shear_in,
        fliplr_in,
        flipud_in,
        mosaic_in,
        mixup_in,
        optimizer_in,
        momentum_in,
        weight_decay_in,
        device_in,
        workers_in,
    ]

    for comp in inputs_for_toml:
        comp.change(
            fn=_make_toml_wrapper,
            inputs=inputs_for_toml,
            outputs=[toml_preview],
        )

    # 初次渲染时生成一次 TOML 预览
    toml_preview.value = train_manager.create_training_config(
        default_task_display,
        (init_ds[0] if init_ds else ""),
        (init_model_labels[0] if init_model_labels else ""),
        100,
        0.01,
        640,
        16,
        0.0,
        0.1,
        0.5,
        0.0,
        0.5,
        0.0,
        1.0,
        0.0,
        "auto",
        0.937,
        0.0005,
        "auto",
        8,
    )

    # 添加任务事件
    add_task_inputs = [
        task_dd,
        ds_dd,
        mdl_dd,
        epochs_in,
        lr_in,
        imgsz_in,
        batch_in,
        degrees_in,
        translate_in,
        scale_in,
        shear_in,
        fliplr_in,
        flipud_in,
        mosaic_in,
        mixup_in,
        optimizer_in,
        momentum_in,
        weight_decay_in,
        device_in,
        workers_in,
        priority_dd,
        st_task_code,
        st_model_map,
    ]

    add_task_btn.click(
        fn=_add_task,
        inputs=add_task_inputs,
        outputs=[current_task_info, current_task_status],
    )

    # 队列管理事件
    refresh_queue_btn.click(
        fn=_refresh_task_list,
        outputs=[task_list],
    )

    refresh_queue_btn.click(
        fn=_refresh_queue_status,
        outputs=[queue_status],
    )

    promote_btn.click(
        fn=_promote_task,
        inputs=[selected_task_id],
        outputs=[current_task_info],
    )

    cancel_btn.click(
        fn=_cancel_task,
        inputs=[selected_task_id],
        outputs=[current_task_info],
    )

    # 监控事件
    stop_current_btn.click(
        fn=lambda: train_manager.stop_training(),
        outputs=[current_task_info, current_task_status],
    )

    clear_logs_btn.click(
        fn=lambda: train_manager.clear_logs(),
        outputs=[log_output, current_task_info],
    )

    # 自动环境检查
    env_status_md.value = train_manager.get_environment_status()

    # 定期更新（每2秒）
    def _periodic_update():
        status_text, info_text, log_text = _update_current_task_display()
        queue_status_text = _refresh_queue_status()
        task_list_data = _refresh_task_list()

        # 提取当前任务的指标
        current_task = train_manager.get_current_task()
        if current_task and current_task.status.value == "running":
            loss = current_task.progress.loss
            acc = current_task.progress.accuracy
            progress = current_task.progress.progress_percent
        else:
            loss = 0.0
            acc = 0.0
            progress = 0.0

        return (
            status_text,
            info_text,
            log_text,
            queue_status_text,
            task_list_data,
            loss,
            acc,
            progress,
        )

    # 使用定时器更新状态
    timer = gr.Timer(value=2.0, active=True)
    timer.tick(
        fn=_periodic_update,
        outputs=[
            current_task_status,
            current_task_info,
            log_output,
            queue_status,
            task_list,
            loss_display,
            acc_display,
            progress_display,
        ],
    )
