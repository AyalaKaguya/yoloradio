"""数据集管理页面模块"""

from __future__ import annotations

from typing import List

import gradio as gr

from ..core import TASK_MAP, extract_pathlike
from ..core.dataset_manager import dataset_manager
from ..core.paths import DATASETS_DIR


def _show_dataset_detail(name: str):
    """显示数据集详细信息"""
    if not name:
        return "请选择数据集"

    detail = dataset_manager.get_dataset_detail(name)
    if not detail:
        return f"无法获取数据集 '{name}' 的详细信息"

    # 格式化详细信息
    info_lines = [
        f"# 数据集详情: {detail['name']}",
        "",
        f"**路径**: `{detail['path']}`\n",
        f"**类型**: {detail['type']}",
        "",
        "## 描述",
        detail["description"] if detail["description"] else "*无描述*",
        "",
        "## 统计信息",
        f"- **总样本数**: {detail['total_images']}",
        f"- **总标注数**: {detail['total_labels']}",
        "",
        "### 按集合分布",
        f"- **Train**: {detail['train_images']} 样本, {detail['train_labels']} 标注",
        f"- **Val**: {detail['val_images']} 样本, {detail['val_labels']} 标注",
        f"- **Test**: {detail['test_images']} 样本, {detail['test_labels']} 标注",
        "",
        "## 元数据信息",
        f"- **元数据文件**: {'存在' if detail['meta_exists'] else '不存在'}",
    ]

    if detail["meta_modified"]:
        info_lines.append(f"- **最后修改**: {detail['meta_modified']}")

    return "\n".join(info_lines)


def create_datasets_tab() -> None:
    """创建数据集管理标签页"""
    gr.Markdown(
        f"""
        ## 数据集管理
        
        数据集目录: `{DATASETS_DIR}`  
        推荐上传压缩包（.zip/.tar.gz/.tgz/.tar.bz2 等），服务器将自动解压到 `Datasets/子目录`。
        体积较大的数据集建议直接在文件系统中放入 `Datasets/`，以避免网页上传耗时。
        """
    )

    # 初始化数据集选项和详情
    dataset_options = dataset_manager.get_dataset_names()
    first_dataset = dataset_options[0] if dataset_options else None

    # 获取初始汇总表格数据
    sum_headers, sum_rows = dataset_manager.get_datasets_summary()

    # 获取初始详情信息
    initial_detail = ""
    if first_dataset:
        initial_detail = _show_dataset_detail(first_dataset)

    with gr.Row():
        # 左侧：上传 + 管理面板
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 信息与上传")
            name_in = gr.Textbox(
                label="数据集名称", placeholder="例如：screw_long_obbkp_dataset"
            )
            desc_in = gr.Textbox(
                label="数据集描述", placeholder="可填写来源、标注说明等", lines=3
            )
            type_in = gr.Dropdown(
                choices=list(TASK_MAP.keys()),
                label="数据集类型",
                value="目标检测",
            )
            ds_archive = gr.File(
                label="上传压缩包（仅限单个文件）", file_count="single"
            )
            with gr.Row():
                extract_ds_btn = gr.Button("上传并解压到 Datasets/", variant="primary")
            ds_status = gr.Markdown(visible=True)

            gr.Markdown("---")
            gr.Markdown("### 管理所选数据集")
            ds_select = gr.Dropdown(
                choices=dataset_options,
                value=first_dataset,
                label="选择数据集",
                allow_custom_value=True,
            )
            new_name_in = gr.Textbox(label="新名称", placeholder="留空则不改名")
            # 初始描述：根据当前选择自动加载
            dataset = (
                dataset_manager.get_dataset(first_dataset) if first_dataset else None
            )
            _init_desc = dataset.description if dataset else ""
            desc_edit = gr.Textbox(label="描述", lines=4, value=_init_desc)
            with gr.Row():
                save_btn = gr.Button("保存修改", variant="primary")
                del_btn = gr.Button("删除所选", variant="stop")
            op_status = gr.Markdown(visible=True)

        # 右侧：数据集列表和详细信息展示
        with gr.Column(scale=2, min_width=760):
            refresh_btn = gr.Button("刷新列表", variant="secondary")

            with gr.Row():
                # 数据集列表列
                with gr.Column(scale=1):
                    gr.Markdown("### 数据集列表")
                    ds_table = gr.Dataframe(
                        headers=sum_headers,
                        value=sum_rows,
                        interactive=False,
                        wrap=True,
                    )

            # 详细信息展示列
            with gr.Column(scale=1):
                gr.Markdown("### 数据集详细信息")
                detail_ds_select = gr.Dropdown(
                    choices=dataset_options,
                    value=first_dataset,
                    label="选择要查看的数据集",
                    interactive=True,
                )
                detail_info = gr.Markdown(
                    value=(
                        initial_detail if initial_detail else "请选择数据集查看详细信息"
                    ),
                    visible=True,
                )

    # 事件处理函数
    def handle_upload(name: str, desc: str, dtype: str, archive) -> str:
        """处理数据集上传"""
        p = extract_pathlike(archive)
        if not p:
            return "请上传一个压缩包文件"
        ok, msg = dataset_manager.create_dataset_from_upload(
            name, desc, dtype, p, TASK_MAP
        )
        return msg

    def handle_description_load(name: str):
        """加载数据集描述"""
        if not name:
            return gr.update(value="")
        dataset = dataset_manager.get_dataset(name)
        desc = dataset.description if dataset else ""
        return gr.update(value=desc)

    def handle_save_changes(selected: str, new_name: str, new_desc: str):
        """保存数据集更改"""
        msgs: list[str] = []
        final_selected = selected

        # 重命名如果需要
        if selected and new_name and new_name != selected:
            ok, final_name, m = dataset_manager.rename_dataset(selected, new_name)
            msgs.append(m)
            if ok:
                final_selected = final_name

        # 更新描述
        if final_selected:
            ok, m = dataset_manager.update_dataset_description(
                final_selected, new_desc or ""
            )
            msgs.append(m)

        # 刷新数据
        headers, rows = dataset_manager.get_datasets_summary()
        names = dataset_manager.get_dataset_names()

        return (
            "\n".join(msgs) or "已保存",
            gr.update(headers=headers, value=rows),
            gr.update(
                choices=names,
                value=(
                    final_selected
                    if final_selected in names
                    else (names[0] if names else None)
                ),
            ),
        )

    def handle_delete(selected: str):
        """删除数据集"""
        if not selected:
            return "请选择要删除的数据集", gr.update(), gr.update()

        ok, m = dataset_manager.delete_dataset(selected)
        headers, rows = dataset_manager.get_datasets_summary()
        names = dataset_manager.get_dataset_names()

        return (
            m,
            gr.update(headers=headers, value=rows),
            gr.update(choices=names, value=(names[0] if names else None)),
        )

    def handle_refresh():
        """刷新所有界面元素"""
        dataset_options = dataset_manager.get_dataset_names()
        first_dataset = dataset_options[0] if dataset_options else None
        sum_headers, sum_rows = dataset_manager.get_datasets_summary()
        dataset = dataset_manager.get_dataset(first_dataset) if first_dataset else None
        init_desc = dataset.description if dataset else ""
        initial_detail = (
            _show_dataset_detail(first_dataset)
            if first_dataset
            else "请选择数据集查看详细信息"
        )

        return (
            gr.update(headers=sum_headers, value=sum_rows),  # ds_table
            gr.update(choices=dataset_options, value=first_dataset),  # ds_select
            gr.update(value=init_desc),  # desc_edit
            gr.update(choices=dataset_options, value=first_dataset),  # detail_ds_select
            initial_detail,  # detail_info
        )

    # 事件绑定
    extract_ds_btn.click(
        fn=handle_upload,
        inputs=[name_in, desc_in, type_in, ds_archive],
        outputs=[ds_status],
    )
    # 上传后刷新所有界面
    extract_ds_btn.click(
        fn=handle_refresh,
        outputs=[ds_table, ds_select, desc_edit, detail_ds_select, detail_info],
    )
    # 刷新按钮
    refresh_btn.click(
        fn=handle_refresh,
        outputs=[ds_table, ds_select, desc_edit, detail_ds_select, detail_info],
    )
    # 自动在切换选择时加载描述
    ds_select.change(
        fn=handle_description_load, inputs=[ds_select], outputs=[desc_edit]
    )
    save_btn.click(
        fn=handle_save_changes,
        inputs=[ds_select, new_name_in, desc_edit],
        outputs=[op_status, ds_table, ds_select],
    )
    del_btn.click(
        fn=handle_delete, inputs=[ds_select], outputs=[op_status, ds_table, ds_select]
    )

    # 详情查看事件绑定
    detail_ds_select.change(
        fn=_show_dataset_detail, inputs=[detail_ds_select], outputs=[detail_info]
    )
