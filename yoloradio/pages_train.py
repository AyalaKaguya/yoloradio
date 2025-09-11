from __future__ import annotations

import gradio as gr

from .utils import (
    DATASET_TYPE_MAP,
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


def render() -> None:
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
                    with gr.Accordion("数据增强/优化参数", open=False):
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

                    with gr.Accordion("训练器与系统参数", open=False):
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

            # 内部状态：任务代码 + 模型标签->路径 映射（供后续运行使用）
            st_task_code = gr.State(default_task_code)
            st_model_map = gr.State({lbl: path for lbl, path in init_models})

        with gr.Tab("训练"):
            with gr.Row():
                # 左侧：控制按钮
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### 训练控制")

                    # 环境检查
                    env_status_md = gr.Markdown("🔍 检查训练环境...")
                    check_env_btn = gr.Button("检查环境", variant="secondary")

                    train_status_md = gr.Markdown("状态: 就绪")
                    train_progress = gr.Progress()

                    with gr.Row():
                        start_btn = gr.Button("开始训练", variant="primary")
                        pause_btn = gr.Button("暂停", variant="secondary")
                        resume_btn = gr.Button("恢复", variant="secondary")
                        stop_btn = gr.Button("停止", variant="stop")

                    clear_logs_btn = gr.Button("清空日志", variant="secondary")

                    # 设备信息
                    device_info_md = gr.Markdown(f"**设备信息**: {get_device_info()}")

                    gr.Markdown("### 训练信息")
                    train_info_md = gr.Markdown("等待开始训练...")

                # 右侧：实时日志
                with gr.Column(scale=2, min_width=640):
                    gr.Markdown("### 训练日志")
                    log_output = gr.Textbox(
                        label="实时输出",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                    )

        with gr.Tab("日志"):
            gr.Markdown("### 训练日志浏览")
            gr.Markdown("此处可以浏览完整的训练日志和生成的图表。")

    def _to_task_code(display: str) -> str:
        return DATASET_TYPE_MAP.get(display, "detect")

    def _refresh_options(task_display: str):
        code = _to_task_code(task_display)
        ds = list_datasets_for_task(code)
        models = list_models_for_task(code)
        labels = [m[0] for m in models]
        ds_val = ds[0] if ds else None
        mdl_val = labels[0] if labels else None
        return (
            gr.update(choices=ds, value=ds_val),
            gr.update(choices=labels, value=mdl_val),
            code,
            {lbl: path for lbl, path in models},
        )

    def _make_toml(
        task_display: str,
        ds: str,
        mdl: str,
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
    ):
        task_code = _to_task_code(task_display)
        # 简单 TOML 拼装
        lines = []
        lines.append(f'task = "{task_code}"')
        if ds:
            lines.append(f'dataset = "{ds}"')
        if mdl:
            lines.append(f'model = "{mdl}"')
        lines.append("")
        lines.append("[train]")
        lines.append(f"epochs = {int(epochs)}")
        lines.append(f"lr0 = {float(lr0)}")
        lines.append(f"imgsz = {int(imgsz)}")
        lines.append(f"batch = {int(batch)}")
        lines.append("")
        lines.append("[augment]")
        lines.append(f"degrees = {float(degrees)}")
        lines.append(f"translate = {float(translate)}")
        lines.append(f"scale = {float(scale)}")
        lines.append(f"shear = {float(shear)}")
        lines.append(f"fliplr = {float(fliplr)}")
        lines.append(f"flipud = {float(flipud)}")
        lines.append(f"mosaic = {float(mosaic)}")
        lines.append(f"mixup = {float(mixup)}")
        lines.append("")
        lines.append("[trainer]")
        lines.append(f'optimizer = "{optimizer}"')
        lines.append(f"momentum = {float(momentum)}")
        lines.append(f"weight_decay = {float(weight_decay)}")
        lines.append(f'device = "{device}"')
        lines.append(f"workers = {int(workers)}")
        return "\n".join(lines) + "\n"

    # 事件：改变任务 -> 刷新 数据集/模型 选择 + 状态
    task_dd.change(
        fn=_refresh_options,
        inputs=[task_dd],
        outputs=[ds_dd, mdl_dd, st_task_code, st_model_map],
    )
    refresh_btn.click(
        fn=_refresh_options,
        inputs=[task_dd],
        outputs=[ds_dd, mdl_dd, st_task_code, st_model_map],
    )

    # 事件：任一参数变化 -> 重建 TOML 预览
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
            fn=_make_toml,
            inputs=inputs_for_toml,
            outputs=[toml_preview],
        )

    # 初次渲染时生成一次 TOML 预览
    toml_preview.value = _make_toml(
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

    # 训练相关的函数
    def _start_training(
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
        task_code: str,
        model_map: dict,
    ):
        if not dataset or not model_label:
            return "❌ 请选择数据集和模型", "状态: 就绪"

        if model_label not in model_map:
            return "❌ 模型路径无效", "状态: 就绪"

        model_path = model_map[model_label]

        # 准备训练参数
        train_kwargs = {
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "fliplr": fliplr,
            "flipud": flipud,
            "mosaic": mosaic,
            "mixup": mixup,
            "optimizer": optimizer,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "workers": workers,
        }

        success, message = start_training(
            task_code=task_code,
            dataset_name=dataset,
            model_path=model_path,
            epochs=epochs,
            lr0=lr0,
            imgsz=imgsz,
            batch=batch,
            device=device,
            **train_kwargs,
        )

        if success:
            return f"✅ {message}", "状态: 训练中..."
        else:
            return f"❌ {message}", "状态: 就绪"

    def _pause_training():
        success, message = pause_training()
        if success:
            return f"⏸️ {message}", "状态: 已暂停"
        else:
            return f"❌ {message}", "状态: 错误"

    def _resume_training():
        success, message = resume_training()
        if success:
            return f"▶️ {message}", "状态: 训练中..."
        else:
            return f"❌ {message}", "状态: 错误"

    def _stop_training():
        success, message = stop_training()
        if success:
            return f"⏹️ {message}", "状态: 已停止"
        else:
            return f"❌ {message}", "状态: 错误"

    def _clear_logs():
        clear_training_logs()
        return "", "日志已清空"

    def _update_training_display():
        """更新训练状态显示"""
        status = get_training_status()
        logs = get_training_logs()

        # 状态信息
        if status["is_running"]:
            if status["is_paused"]:
                status_text = f"状态: 已暂停 - Epoch {status['current_epoch']}/{status['total_epochs']} ({status['progress']:.1f}%)"
            else:
                status_text = f"状态: 训练中 - Epoch {status['current_epoch']}/{status['total_epochs']} ({status['progress']:.1f}%)"
        else:
            status_text = "状态: 就绪"

        # 训练信息
        if status["run_id"]:
            info_text = f"""**运行ID**: {status['run_id']}
**进度**: {status['current_epoch']}/{status['total_epochs']} ({status['progress']:.1f}%)
**日志行数**: {status['log_count']}"""
        else:
            info_text = "等待开始训练..."

        # 日志文本
        log_text = "\n".join(logs[-100:])  # 只显示最近100行

        return status_text, info_text, log_text

    def _refresh_full_logs():
        """刷新完整日志"""
        logs = get_training_logs()
        return "\n".join(logs)

    # 训练控制事件
    start_btn.click(
        fn=_start_training,
        inputs=[
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
            st_task_code,
            st_model_map,
        ],
        outputs=[train_info_md, train_status_md],
    )

    pause_btn.click(fn=_pause_training, outputs=[train_info_md, train_status_md])
    resume_btn.click(fn=_resume_training, outputs=[train_info_md, train_status_md])
    stop_btn.click(fn=_stop_training, outputs=[train_info_md, train_status_md])
    clear_logs_btn.click(fn=_clear_logs, outputs=[log_output, train_info_md])

    # 环境检查事件
    def _check_environment():
        """检查训练环境并返回结果字符串"""
        success, message = validate_training_environment()
        return message

    check_env_btn.click(fn=_check_environment, outputs=[env_status_md])

    # 定期更新训练状态（每2秒）
    def _periodic_update():
        return _update_training_display()

    # 使用定时器更新状态
    timer = gr.Timer(value=2.0, active=True)
    timer.tick(
        fn=_periodic_update, outputs=[train_status_md, train_info_md, log_output]
    )
