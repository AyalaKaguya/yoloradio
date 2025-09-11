"""模型管理页面模块"""

from __future__ import annotations

import gradio as gr

from ..core import (
    delete_model,
    download_pretrained_if_missing,
    get_all_models_info,
    get_model_detail,
    get_model_details,
    rename_model,
)
from ..core.paths import MODELS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR

# 模型任务映射与数据集一致
MODEL_TASK_MAP = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}


def refresh_model_lists():
    """刷新模型列表"""
    all_models = get_all_models_info()
    pre_models = [model["stem"] for model in all_models["pretrained"]]
    tr_models = [model["stem"] for model in all_models["trained"]]
    return pre_models, tr_models


def refresh_model_details():
    """刷新模型详细信息"""
    all_models = get_all_models_info()

    # 构建表格数据
    pre_details = [
        [model["filename"], model["description"], model["created_str"]]
        for model in all_models["pretrained"]
    ]
    tr_details = [
        [model["filename"], model["description"], model["created_str"]]
        for model in all_models["trained"]
    ]

    return pre_details, tr_details


def get_model_choices_for_detail():
    """获取用于详细信息下拉框的选项"""
    all_models = get_all_models_info()

    choices = []
    # 添加预训练模型
    for model in all_models["pretrained"]:
        display_name = f"[预训练] {model['stem']}"
        choices.append((display_name, model["stem"], True))

    # 添加训练模型
    for model in all_models["trained"]:
        display_name = f"[训练] {model['stem']}"
        choices.append((display_name, model["stem"], False))

    return choices


def download_and_register_pretrained(
    task: str, ver: str, size: str, name: str, desc: str, progress_cb=None
):
    """下载并注册预训练模型"""
    try:
        # 构建模型文件名
        task_code = MODEL_TASK_MAP.get(task, "detect")
        if task_code == "detect":
            model_name = f"yolo{ver}{size}.pt"
        else:
            model_name = f"yolo{ver}{size}-{task_code}.pt"

        # 调用核心下载函数
        ok, msg, _ = download_pretrained_if_missing(model_name)

        if ok and name:
            # 如果指定了自定义名称，进行重命名
            old_path = MODELS_PRETRAINED_DIR / model_name
            new_name_with_ext = f"{name}.pt"
            new_path = MODELS_PRETRAINED_DIR / new_name_with_ext

            if new_path.exists():
                return False, f"名称 '{name}' 已存在"

            try:
                old_path.rename(new_path)

                # 创建元数据文件
                if desc:
                    meta_content = f"""name: {name}
description: {desc}
task: {task_code}
version: {ver}
size: {size}
created_at: {old_path.stat().st_mtime if old_path.exists() else "unknown"}
"""
                    meta_path = new_path.with_suffix(".yml")
                    meta_path.write_text(meta_content, encoding="utf-8")

                return True, f"模型已下载并重命名为: {new_name_with_ext}"
            except Exception as e:
                return False, f"重命名失败: {e}"

        return ok, msg

    except Exception as e:
        return False, f"下载失败: {e}"


def create_models_tab() -> None:
    """创建模型管理标签页"""
    gr.Markdown(
        f"模型目录: `{MODELS_DIR}`\n\n- 预训练模型: `{MODELS_PRETRAINED_DIR}`\n- 训练模型: `{MODELS_TRAINED_DIR}`"
    )

    # 初始列表（页面打开时即填充）
    pre_init, tr_init = refresh_model_lists()
    pre_details_init, tr_details_init = refresh_model_details()

    # 初始化详细信息下拉框选项
    model_choices = get_model_choices_for_detail()
    detail_choices_init = [choice[0] for choice in model_choices]  # 显示名称
    detail_value_init = detail_choices_init[0] if detail_choices_init else None

    with gr.Row():
        # 左侧：下载 + 统一管理面板
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 获取预训练模型（从 Ultralytics 下载）")
            task_dd = gr.Dropdown(
                choices=list(MODEL_TASK_MAP.keys()), value="目标检测", label="任务类型"
            )
            ver_dd = gr.Dropdown(choices=["v8", "11"], value="11", label="YOLO 版本")
            size_dd = gr.Dropdown(
                choices=["n", "s", "m", "l", "x"], value="n", label="模型大小"
            )
            mdl_name_in = gr.Textbox(
                label="模型名称（保存名，可选）", placeholder="例如：yolo11n-det"
            )
            mdl_desc_in = gr.Textbox(label="模型描述（可选）", lines=3)
            fetch_btn = gr.Button("下载并登记", variant="primary")
            fetch_status = gr.Markdown(visible=True)

            gr.Markdown("---")
            gr.Markdown("### 模型管理")
            cat_radio = gr.Radio(
                choices=["预训练", "训练"], value="预训练", label="模型类别"
            )
            mdl_sel = gr.Dropdown(
                choices=pre_init,
                value=(pre_init[0] if pre_init else None),
                label="选择模型",
            )
            new_name_in = gr.Textbox(label="新名称", placeholder="留空不改名")
            with gr.Row():
                rename_btn = gr.Button("重命名")
                delete_btn = gr.Button("删除", variant="stop")
            op_status = gr.Markdown(visible=True)

        # 右侧：两个列表和详细信息展示
        with gr.Column(scale=2, min_width=760):
            refresh_btn = gr.Button("刷新列表", variant="secondary")

            with gr.Row():
                # 模型列表列
                with gr.Column(scale=1):
                    gr.Markdown("### 预训练模型")
                    pretrained_df = gr.Dataframe(
                        headers=["文件名", "描述", "创建时间"],
                        value=pre_details_init,
                        interactive=False,
                    )
                    gr.Markdown("### 训练模型")
                    trained_df = gr.Dataframe(
                        headers=["文件名", "描述", "创建时间"],
                        value=tr_details_init,
                        interactive=False,
                    )

                    # 详细信息展示列
                    gr.Markdown("### 模型详细信息")
                    model_detail_dropdown = gr.Dropdown(
                        choices=detail_choices_init,
                        value=detail_value_init,
                        label="选择要查看的模型",
                        interactive=True,
                    )
                    model_detail_area = gr.Markdown(
                        value="请选择一个模型查看详细信息", visible=True
                    )

    def _refresh_models_for_category(category: str):
        pre, tr = refresh_model_lists()
        pre_details, tr_details = refresh_model_details()
        choices = pre if category == "预训练" else tr
        value = choices[0] if choices else None

        # 更新详细信息下拉框选项
        model_choices = get_model_choices_for_detail()
        detail_choices = [choice[0] for choice in model_choices]  # 显示名称
        detail_value = detail_choices[0] if detail_choices else None

        return (
            gr.update(value=pre_details),
            gr.update(value=tr_details),
            gr.update(choices=choices, value=value),
            gr.update(choices=detail_choices, value=detail_value),
        )

    def _on_category_change(category: str):
        pre, tr = refresh_model_lists()
        choices = pre if category == "预训练" else tr
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value)

    def _update_model_detail(selected_display: str):
        """更新模型详细信息显示"""
        if not selected_display:
            return gr.update(value="请选择一个模型查看详细信息")

        # 解析选择的模型
        if selected_display.startswith("[预训练] "):
            model_name = selected_display[5:].strip()
            is_pretrained = True
        elif selected_display.startswith("[训练] "):
            model_name = selected_display[4:].strip()
            is_pretrained = False
        else:
            return gr.update(value="无法解析模型信息")

        detail_info = get_model_detail(model_name, is_pretrained)
        return gr.update(value=detail_info)

    def _fetch(
        task: str,
        ver: str,
        size: str,
        name: str,
        desc: str,
        progress=gr.Progress(track_tqdm=True),
    ):
        def _cb(p, txt):
            try:
                progress(p, desc=txt)
            except Exception:
                pass

        ok, msg = download_and_register_pretrained(
            task, ver, size, name, desc, progress_cb=_cb
        )
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
    fetch_btn.click(
        fn=_fetch,
        inputs=[task_dd, ver_dd, size_dd, mdl_name_in, mdl_desc_in],
        outputs=[fetch_status],
    )
    # 下载后刷新列表与选择器（保持当前类别）
    fetch_btn.click(
        fn=_refresh_models_for_category,
        inputs=[cat_radio],
        outputs=[pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    refresh_btn.click(
        fn=_refresh_models_for_category,
        inputs=[cat_radio],
        outputs=[pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    cat_radio.change(fn=_on_category_change, inputs=[cat_radio], outputs=[mdl_sel])
    rename_btn.click(
        fn=_rename,
        inputs=[cat_radio, mdl_sel, new_name_in],
        outputs=[op_status, pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    delete_btn.click(
        fn=_delete,
        inputs=[cat_radio, mdl_sel],
        outputs=[op_status, pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )

    # 模型详细信息相关事件
    model_detail_dropdown.change(
        fn=_update_model_detail,
        inputs=[model_detail_dropdown],
        outputs=[model_detail_area],
    )

    # 页面加载时显示初始模型的详细信息
    if detail_value_init:
        # 解析初始选择的模型
        if detail_value_init.startswith("[预训练] "):
            initial_model_name = detail_value_init[5:].strip()
            initial_is_pretrained = True
        elif detail_value_init.startswith("[训练] "):
            initial_model_name = detail_value_init[4:].strip()
            initial_is_pretrained = False
        else:
            initial_model_name = None
            initial_is_pretrained = True

        if initial_model_name:
            model_detail_area.value = get_model_detail(
                initial_model_name, initial_is_pretrained
            )
