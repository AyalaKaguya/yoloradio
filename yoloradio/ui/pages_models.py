"""模型管理页面模块"""

from __future__ import annotations

import gradio as gr

from ..core import TASK_MAP, model_manager
from ..core.paths import MODELS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR


def create_models_tab() -> None:
    """创建模型管理标签页"""
    gr.Markdown(
        f"## 模型管理\n\n模型目录: `{MODELS_DIR}`\n\n- 预训练模型: `{MODELS_PRETRAINED_DIR}`\n- 训练模型: `{MODELS_TRAINED_DIR}`"
    )

    # 初始列表（页面打开时即填充）
    pre_init, tr_init = model_manager.refresh_model_lists()
    pre_details_init, tr_details_init = model_manager.refresh_model_details()

    # 初始化详细信息下拉框选项
    detail_choices_init = model_manager.get_model_choices_for_detail()
    detail_value_init = detail_choices_init[0] if detail_choices_init else None

    with gr.Row():
        # 左侧：下载 + 统一管理面板
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 获取预训练模型（从 Ultralytics 下载）")
            task_dd = gr.Dropdown(
                choices=list(TASK_MAP.keys()), value="目标检测", label="任务类型"
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

    # 事件处理函数
    def handle_fetch(
        task: str,
        ver: str,
        size: str,
        name: str,
        desc: str,
        progress=gr.Progress(track_tqdm=True),
    ):
        """处理模型下载"""

        def progress_callback(p, txt):
            try:
                progress(p, desc=txt)
            except Exception:
                pass

        ok, msg = model_manager.download_and_register_pretrained(
            task, ver, size, name, desc, progress_callback
        )
        return msg

    def handle_refresh_models(category: str):
        """刷新模型列表和详细信息"""
        pre_models, tr_models = model_manager.refresh_model_lists()
        pre_details, tr_details = model_manager.refresh_model_details()

        choices = pre_models if category == "预训练" else tr_models
        value = choices[0] if choices else None

        # 更新详细信息下拉框选项
        detail_choices = model_manager.get_model_choices_for_detail()
        detail_value = detail_choices[0] if detail_choices else None

        return (
            gr.update(value=pre_details),
            gr.update(value=tr_details),
            gr.update(choices=choices, value=value),
            gr.update(choices=detail_choices, value=detail_value),
        )

    def handle_category_change(category: str):
        """处理模型类别变更"""
        pre_models, tr_models = model_manager.refresh_model_lists()
        choices = pre_models if category == "预训练" else tr_models
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value)

    def handle_model_detail(selected_display: str):
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

        detail_info = model_manager.get_model_detail(model_name, is_pretrained)
        return gr.update(value=detail_info)

    def handle_rename(category: str, old: str, new: str):
        """处理模型重命名"""
        if not old or not new:
            return ("请输入新名称",) + handle_refresh_models(category)

        is_pretrained = category == "预训练"
        # 从文件名中提取模型名称（去掉扩展名）
        old_name = old.split(".")[0] if "." in old else old
        new_name = new.split(".")[0] if "." in new else new

        ok, final_name, msg = model_manager.rename_model(
            old_name, new_name, is_pretrained
        )
        return (msg,) + handle_refresh_models(category)

    def handle_delete(category: str, old: str):
        """处理模型删除"""
        if not old:
            return ("请选择要删除的模型",) + handle_refresh_models(category)

        is_pretrained = category == "预训练"
        # 从文件名中提取模型名称（去掉扩展名）
        model_name = old.split(".")[0] if "." in old else old

        ok, msg = model_manager.delete_model(model_name, is_pretrained)
        return (msg,) + handle_refresh_models(category)

    # 事件绑定
    fetch_btn.click(
        fn=handle_fetch,
        inputs=[task_dd, ver_dd, size_dd, mdl_name_in, mdl_desc_in],
        outputs=[fetch_status],
    )
    # 下载后刷新列表与选择器（保持当前类别）
    fetch_btn.click(
        fn=handle_refresh_models,
        inputs=[cat_radio],
        outputs=[pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    refresh_btn.click(
        fn=handle_refresh_models,
        inputs=[cat_radio],
        outputs=[pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    cat_radio.change(fn=handle_category_change, inputs=[cat_radio], outputs=[mdl_sel])
    rename_btn.click(
        fn=handle_rename,
        inputs=[cat_radio, mdl_sel, new_name_in],
        outputs=[op_status, pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )
    delete_btn.click(
        fn=handle_delete,
        inputs=[cat_radio, mdl_sel],
        outputs=[op_status, pretrained_df, trained_df, mdl_sel, model_detail_dropdown],
    )

    # 模型详细信息相关事件
    model_detail_dropdown.change(
        fn=handle_model_detail,
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
            model_detail_area.value = model_manager.get_model_detail(
                initial_model_name, initial_is_pretrained
            )
