from __future__ import annotations

import gradio as gr

from .paths import MODELS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR
from .utils import (
    MODEL_TASK_MAP,
    refresh_model_lists,
    download_and_register_pretrained,
    delete_model,
    rename_model,
)


def render() -> None:
    gr.Markdown(f"模型目录: `{MODELS_DIR}`\n\n- 预训练模型: `{MODELS_PRETRAINED_DIR}`\n- 训练模型: `{MODELS_TRAINED_DIR}`")

    # 初始列表（页面打开时即填充）
    pre_init, tr_init = refresh_model_lists()

    with gr.Row():
        # 左侧：下载 + 统一管理面板（放在下载下方）
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 获取预训练模型（从 Ultralytics 下载）")
            task_dd = gr.Dropdown(choices=list(MODEL_TASK_MAP.keys()), value="目标检测", label="任务类型")
            ver_dd = gr.Dropdown(choices=["v8", "11"], value="11", label="YOLO 版本")
            size_dd = gr.Dropdown(choices=["n", "s", "m", "l", "x"], value="n", label="模型大小")
            mdl_name_in = gr.Textbox(label="模型名称（保存名，可选）", placeholder="例如：yolo11n-det")
            mdl_desc_in = gr.Textbox(label="模型描述（可选）", lines=3)
            fetch_btn = gr.Button("下载并登记", variant="primary")
            fetch_status = gr.Markdown(visible=True)

            gr.Markdown("---")
            gr.Markdown("### 模型管理")
            cat_radio = gr.Radio(choices=["预训练", "训练"], value="预训练", label="模型类别")
            mdl_sel = gr.Dropdown(choices=pre_init, value=(pre_init[0] if pre_init else None), label="选择模型")
            new_name_in = gr.Textbox(label="新名称", placeholder="留空不改名")
            with gr.Row():
                rename_btn = gr.Button("重命名")
                delete_btn = gr.Button("删除", variant="stop")
            op_status = gr.Markdown(visible=True)

        # 右侧：两个列表，共用一个刷新按钮
        with gr.Column(scale=2, min_width=760):
            refresh_btn = gr.Button("刷新列表", variant="secondary")
            gr.Markdown("### 预训练模型")
            pretrained_df = gr.Dataframe(headers=["文件名"], value=[[n] for n in pre_init], interactive=False)
            gr.Markdown("### 训练模型")
            trained_df = gr.Dataframe(headers=["文件名"], value=[[n] for n in tr_init], interactive=False)

    def _refresh_models_for_category(category: str):
        pre, tr = refresh_model_lists()
        choices = pre if category == "预训练" else tr
        value = (choices[0] if choices else None)
        return (
            gr.update(value=[[n] for n in pre]),
            gr.update(value=[[n] for n in tr]),
            gr.update(choices=choices, value=value),
        )

    def _on_category_change(category: str):
        pre, tr = refresh_model_lists()
        choices = pre if category == "预训练" else tr
        value = (choices[0] if choices else None)
        return gr.update(choices=choices, value=value)

    def _fetch(task: str, ver: str, size: str, name: str, desc: str, progress=gr.Progress(track_tqdm=True)):
        def _cb(p, txt):
            try:
                progress(p, desc=txt)
            except Exception:
                pass
        ok, msg = download_and_register_pretrained(task, ver, size, name, desc, progress_cb=_cb)
        return msg

    def _rename(category: str, old: str, new: str):
        if not old or not new:
            return ("请输入新名称",) + _refresh_models_for_category(category)
        is_pre = category == "预训练"
        ok, final_name, m = rename_model(is_pre, old, new)
        return (m,) + _refresh_models_for_category(category)

    def _delete(category: str, old: str):
        if not old:
            return ("请选择要删除的模型",) + _refresh_models_for_category(category)
        is_pre = category == "预训练"
        ok, m = delete_model(is_pre, old)
        return (m,) + _refresh_models_for_category(category)

    # 事件绑定
    fetch_btn.click(fn=_fetch, inputs=[task_dd, ver_dd, size_dd, mdl_name_in, mdl_desc_in], outputs=[fetch_status])
    # 下载后刷新列表与选择器（保持当前类别）
    fetch_btn.click(fn=_refresh_models_for_category, inputs=[cat_radio], outputs=[pretrained_df, trained_df, mdl_sel])
    refresh_btn.click(fn=_refresh_models_for_category, inputs=[cat_radio], outputs=[pretrained_df, trained_df, mdl_sel])
    cat_radio.change(fn=_on_category_change, inputs=[cat_radio], outputs=[mdl_sel])
    rename_btn.click(fn=_rename, inputs=[cat_radio, mdl_sel, new_name_in], outputs=[op_status, pretrained_df, trained_df, mdl_sel])
    delete_btn.click(fn=_delete, inputs=[cat_radio, mdl_sel], outputs=[op_status, pretrained_df, trained_df, mdl_sel])
