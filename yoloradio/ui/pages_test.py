"""测试页面：选择图像与模型进行推理，输出图片与表格"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr
import numpy as np

from ..core.model_manager import model_manager


def create_test_tab() -> None:
    gr.Markdown("## 模型测试\n选择一张图片与模型，进行快速推理并查看结果。")

    def _get_all_models() -> Tuple[List[str], dict]:
        models = model_manager.list_models()
        labels: List[str] = []
        mapping: dict = {}
        for m in models:
            prefix = "预训练" if m.is_pretrained else "训练"
            label = f"[{prefix}] {m.name} ({m.task_display})"
            labels.append(label)
            mapping[label] = str(m.path) if m.path else ""
        return labels, mapping

    def _refresh_models():
        labels, _map = _get_all_models()
        return gr.update(choices=labels, value=(labels[0] if labels else None))

    # 预加载模型列表
    initial_labels, _ = _get_all_models()

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            img_in = gr.Image(label="选择图片", type="numpy")
            refresh_btn = gr.Button("刷新模型列表", variant="secondary")
            model_dd = gr.Dropdown(
                label="选择模型",
                choices=initial_labels,
                value=(initial_labels[0] if initial_labels else None),
            )
            with gr.Row():
                conf_sl = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.25, step=0.01, label="置信度阈值"
                )
                device_dd = gr.Dropdown(
                    label="设备", choices=["auto", "cpu", "cuda:0"], value="auto"
                )
            infer_btn = gr.Button("推理", variant="primary")

        with gr.Column(scale=2, min_width=520):
            out_img = gr.Image(label="结果图像")
            cols = [
                "label",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "x_center",
                "y_center",
                "w",
                "h",
                "theta",
            ]
            out_tbl = gr.Dataframe(headers=cols, row_count=(0, "dynamic"), wrap=True)

    refresh_btn.click(fn=_refresh_models, outputs=[model_dd])

    def _infer(image: Any, model_label: str, conf: float, device: str):
        if image is None:
            return None, []
        labels, mapping = _get_all_models()
        if not model_label or model_label not in mapping:
            return image, []
        model_path = mapping[model_label]
        if not model_path:
            return image, []

        try:
            from ultralytics import YOLO

            yolo = YOLO(model_path)

            # 执行推理
            try:
                preds = yolo(
                    image,
                    conf=float(conf),
                    device=(device if device and device != "auto" else None),
                    verbose=False,
                )
            except TypeError:
                # 兼容部分版本采用 predict 接口
                preds = yolo.predict(
                    source=image,
                    conf=float(conf),
                    device=(device if device and device != "auto" else None),
                    verbose=False,
                )

            result = preds[0]

            # 绘制结果
            plotted = result.plot()  # BGR ndarray
            if (
                isinstance(plotted, np.ndarray)
                and plotted.ndim == 3
                and plotted.shape[2] == 3
            ):
                rgb = plotted[:, :, ::-1]
            else:
                rgb = plotted

            # names 映射
            names = None
            if hasattr(result, "names") and isinstance(result.names, (list, dict)):
                names = result.names
            elif hasattr(yolo, "names"):
                names = yolo.names

            def _cls_name(cid: int) -> str:
                if names is None:
                    return str(cid)
                if isinstance(names, dict):
                    return str(names.get(int(cid), cid))
                if 0 <= int(cid) < len(names):
                    return str(names[int(cid)])
                return str(cid)

            rows: List[List[Any]] = []

            # 优先使用 axis-aligned boxes
            if hasattr(result, "boxes") and result.boxes is not None:
                try:
                    for b in result.boxes:
                        xyxy = b.xyxy[0].tolist()
                        conf_v = float(b.conf[0]) if hasattr(b, "conf") else None
                        cls_v = int(b.cls[0]) if hasattr(b, "cls") else -1
                        rows.append(
                            [
                                _cls_name(cls_v),
                                conf_v,
                                float(xyxy[0]),
                                float(xyxy[1]),
                                float(xyxy[2]),
                                float(xyxy[3]),
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )
                except Exception:
                    pass

            # 若为 OBB，补充相应字段
            if hasattr(result, "obb") and getattr(result, "obb") is not None:
                try:
                    obb = result.obb
                    # 可能提供 xywhr 与 conf/cls
                    xywhr = getattr(obb, "xywhr", None)
                    confs = getattr(obb, "conf", None)
                    clss = getattr(obb, "cls", None)
                    if xywhr is not None:
                        for i in range(len(xywhr)):
                            vals = xywhr[i].tolist()
                            conf_v = float(confs[i]) if confs is not None else None
                            cls_v = int(clss[i]) if clss is not None else -1
                            rows.append(
                                [
                                    _cls_name(cls_v),
                                    conf_v,
                                    "",
                                    "",
                                    "",
                                    "",
                                    float(vals[0]),
                                    float(vals[1]),
                                    float(vals[2]),
                                    float(vals[3]),
                                    float(vals[4]),
                                ]
                            )
                except Exception:
                    pass

            return rgb, rows
        except ImportError:
            # 未安装 ultralytics
            return image, []
        except Exception:
            # 任何错误返回原图与空表
            return image, []

    infer_btn.click(
        fn=_infer,
        inputs=[img_in, model_dd, conf_sl, device_dd],
        outputs=[out_img, out_tbl],
    )
