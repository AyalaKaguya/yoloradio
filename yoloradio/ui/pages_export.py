"""导出页面模块"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr

from ..core.model_manager import Model, model_manager
from ..core.paths import PROJECT_DIR


def create_export_tab() -> None:
    """创建导出标签页"""
    gr.Markdown("## 模型导出\n选择一个模型并导出为所需格式。")

    # 准备可导出格式
    # 参考 ultralytics YOLO export 支持的常见格式
    export_formats = [
        ("PyTorch Script", "torchscript"),
        ("ONNX", "onnx"),
        ("OpenVINO", "openvino"),
        ("TensorRT", "engine"),
        ("CoreML", "coreml"),
        ("TF SavedModel", "saved_model"),
        ("TF GraphDef", "pb"),
        ("TFLite", "tflite"),
        ("TF.js", "tfjs"),
    ]

    def _get_all_models() -> Tuple[List[str], dict]:
        models = model_manager.list_models()
        labels: List[str] = []
        mapping: dict = {}
        for m in models:
            # 标注来源与任务类型
            prefix = "预训练" if m.is_pretrained else "训练"
            label = f"[{prefix}] {m.name} ({m.task_display})"
            labels.append(label)
            mapping[label] = str(m.path) if m.path else ""
        return labels, mapping

    def _refresh_models():
        labels, _map = _get_all_models()
        return gr.update(choices=labels, value=(labels[0] if labels else None))

    def _export(
        model_label: str,
        fmt_display: str,
        opset: int,
        device: str,
        dynamic: bool,
        simplify: bool,
    ):
        if not model_label:
            return "❌ 请选择模型", "", []
        labels, mapping = _get_all_models()
        if model_label not in mapping:
            return f"❌ 模型不可用: {model_label}", "", []
        model_path = mapping[model_label]
        if not model_path:
            return f"❌ 模型路径无效", "", []

        # 找到实际导出格式关键字
        fmt_map = {d: k for d, k in export_formats}
        fmt = fmt_map.get(fmt_display, "onnx")

        try:
            from ultralytics import YOLO

            yolo = YOLO(model_path)
            out_dir = PROJECT_DIR / "Models" / "exported"
            out_dir.mkdir(parents=True, exist_ok=True)

            export_name = f"{Path(model_path).stem}_{fmt}"

            args = {
                "format": fmt,
                "device": device if device and device != "auto" else None,
                "opset": int(opset) if fmt == "onnx" and opset else None,
                "dynamic": bool(dynamic) if fmt == "onnx" else None,
                "simplify": bool(simplify) if fmt == "onnx" else None,
                "project": str(out_dir),
                "name": export_name,
                "exist_ok": True,
            }
            # 清理 None
            args = {k: v for k, v in args.items() if v is not None}

            res: Any = yolo.export(**args)

            export_run_dir = out_dir / export_name
            out_path = str(export_run_dir if export_run_dir.exists() else out_dir)

            # 收集导出文件
            export_files: List[str] = []

            def _collect_from_dir(d: Path) -> List[str]:
                if not d.exists():
                    return []
                return [str(p) for p in d.glob("*") if p.is_file()]

            # 从返回值尽量解析
            try:
                if isinstance(res, (str, Path)):
                    p = Path(res)
                    if p.is_file():
                        export_files = [str(p)]
                        out_path = str(p.parent)
                    elif p.is_dir():
                        export_files = _collect_from_dir(p)
                        out_path = str(p)
                elif isinstance(res, (list, tuple)):
                    for item in list(res):
                        pi = Path(item)
                        if pi.exists():
                            export_files.append(str(pi))
                    if export_files:
                        out_path = str(Path(export_files[0]).parent)
                else:
                    if hasattr(res, "save_dir"):
                        d = Path(getattr(res, "save_dir"))
                        export_files = _collect_from_dir(d)
                        out_path = str(d)
                    elif hasattr(res, "path"):
                        p = Path(getattr(res, "path"))
                        if p.is_file():
                            export_files = [str(p)]
                            out_path = str(p.parent)
                        elif p.exists():
                            export_files = _collect_from_dir(p)
                            out_path = str(p)
            except Exception:
                pass

            # 回退到预期目录
            if not export_files:
                export_files = _collect_from_dir(Path(out_path))

            msg = f"✅ 导出成功: {fmt_display}\n输出目录: {out_path}"
            return msg, out_path, export_files
        except ImportError:
            return "❌ 未安装 ultralytics，请先安装", "", []
        except Exception as e:
            return f"❌ 导出失败: {e}", "", []

    # 预先获取初始模型列表，避免首次渲染下拉异常
    initial_labels, _initial_map = _get_all_models()

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            refresh_btn = gr.Button("刷新模型列表", variant="secondary")
            model_dd = gr.Dropdown(
                label="选择模型",
                choices=initial_labels,
                value=(initial_labels[0] if initial_labels else None),
            )
            fmt_dd = gr.Dropdown(
                label="导出格式", choices=[d for d, _ in export_formats], value="ONNX"
            )
            opset_in = gr.Slider(7, 20, value=12, step=1, label="ONNX opset")
            with gr.Row():
                device_dd = gr.Dropdown(
                    label="设备", choices=["auto", "cpu", "cuda:0"], value="auto"
                )
                dynamic_ck = gr.Checkbox(label="动态维度(ONNX)", value=True)
                simplify_ck = gr.Checkbox(label="简化(ONNX)", value=True)
            export_btn = gr.Button("导出", variant="primary")

        with gr.Column(scale=2, min_width=520):
            status_md = gr.Markdown("等待导出...")
            outdir_tb = gr.Textbox(label="输出目录", interactive=False)
            files_out = gr.Files(label="导出文件", interactive=False)

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])
    export_btn.click(
        fn=_export,
        inputs=[model_dd, fmt_dd, opset_in, device_dd, dynamic_ck, simplify_ck],
        outputs=[status_md, outdir_tb, files_out],
    )
