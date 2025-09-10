from __future__ import annotations

from typing import List

import gradio as gr

from .paths import DATASETS_DIR
from .utils import (
    ensure_unique_dir,
    extract_pathlike,
    is_supported_archive,
    list_dir,
    safe_extract_tar,
    safe_extract_zip,
    strip_archive_suffix,
    dataset_summary_table,
    dataset_detail_table,
    list_dataset_dirs,
    DATASET_TYPE_MAP,
    unwrap_single_root,
    validate_dataset_by_type,
    build_metadata_yaml,
    read_meta_description,
)


def render() -> None:
    gr.Markdown(
        f"""
        数据集目录: `{DATASETS_DIR}`  
        推荐上传压缩包（.zip/.tar.gz/.tgz/.tar.bz2 等），服务器将自动解压到 `Datasets/子目录`。
        体积较大的数据集建议直接在文件系统中放入 `Datasets/`，以避免网页上传耗时。
        """
    )
    with gr.Row():
        # 左侧窄栏：信息与上传 + 管理面板（移到下方）
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 信息与上传")
            name_in = gr.Textbox(label="数据集名称", placeholder="例如：screw_long_obbkp_dataset")
            desc_in = gr.Textbox(label="数据集描述", placeholder="可填写来源、标注说明等", lines=3)
            type_in = gr.Dropdown(choices=list(DATASET_TYPE_MAP.keys()), label="数据集类型", value="目标检测")
            ds_archive = gr.File(label="上传压缩包（仅限单个文件）", file_count="single")
            with gr.Row():
                extract_ds_btn = gr.Button("上传并解压到 Datasets/", variant="primary")
                refresh_ds_btn = gr.Button("刷新统计", variant="secondary")
            ds_status = gr.Markdown(visible=True)

            gr.Markdown("---")
            gr.Markdown("### 管理所选数据集")
            _choices = [p.name for p in list_dataset_dirs()]
            ds_select = gr.Dropdown(choices=_choices, value=(_choices[0] if _choices else None), label="选择数据集", allow_custom_value=True)
            new_name_in = gr.Textbox(label="新名称", placeholder="留空则不改名")
            # 初始描述：根据当前选择自动加载
            _init_desc = read_meta_description(_choices[0]) if _choices else ""
            desc_edit = gr.Textbox(label="描述", lines=4, value=_init_desc)
            with gr.Row():
                save_btn = gr.Button("保存修改", variant="primary")
                del_btn = gr.Button("删除所选", variant="stop")
            op_status = gr.Markdown(visible=True)

        # 右侧宽栏（占两列）：仅显示汇总列表
        with gr.Column(scale=2, min_width=760):
            # 汇总
            with gr.Row():
                gr.Markdown("### 数据集列表")
                refresh_btn2 = gr.Button("刷新")
            sum_headers, sum_rows = dataset_summary_table()
            ds_table = gr.Dataframe(headers=sum_headers, value=sum_rows, interactive=False, wrap=True)

    def upload_and_extract(name: str, desc: str, dtype: str, archive) -> str:
        # 基本校验
        if not name:
            return "请填写数据集名称"
        p = extract_pathlike(archive)
        if not p or not p.exists():
            return "请上传一个压缩包文件"
        if not is_supported_archive(p):
            return f"不支持的压缩包类型: {p.name}"

        # 目标目录
        folder_name = strip_archive_suffix(name.strip())
        dest_dir = ensure_unique_dir(DATASETS_DIR / folder_name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 解压
        try:
            if p.suffix.lower() == ".zip":
                import zipfile
                with zipfile.ZipFile(p, "r") as zf:
                    safe_extract_zip(zf, dest_dir)
            else:
                import tarfile
                mode = "r:gz" if "gz" in "".join(p.suffixes).lower() or p.name.lower().endswith(".tgz") else None
                if p.name.lower().endswith((".bz2", ".tbz")):
                    mode = "r:bz2"
                if p.name.lower().endswith((".xz", ".txz")):
                    mode = "r:xz"
                with tarfile.open(p, mode or "r:") as tf:
                    safe_extract_tar(tf, dest_dir)
        except Exception as e:
            return f"解压失败: {p.name} -> {e}"

        # 展平单根目录
        unwrap_single_root(dest_dir)

        # 结构与类型校验
        type_code = DATASET_TYPE_MAP.get(dtype, "detect")
        ok, structure, msg, splits = validate_dataset_by_type(dest_dir, type_code)
        if not ok:
            # 无法判定为有效数据集，清理空目录（如果没有任何文件）
            try:
                if not any(dest_dir.rglob("*")):
                    dest_dir.rmdir()
            except Exception:
                pass
            return f"数据集校验失败：{msg}"

        # 写入同名 yml 元数据（位于 Datasets 下，与数据集目录同名）
        meta_yaml = build_metadata_yaml(name=folder_name, type_code=type_code, type_display=dtype, description=desc or "", structure=structure, splits=splits)
        meta_path = DATASETS_DIR / f"{folder_name}.yml"
        try:
            meta_path.write_text(meta_yaml, encoding="utf-8")
        except Exception as e:
            return f"已导入但写元数据失败：{e}"

        return f"已导入数据集：{dest_dir.name}（类型：{dtype}，结构：{structure}）\n元数据：{meta_path.name}"

    def _refresh_tables():
        sh, sr = dataset_summary_table()
        choices = [p.name for p in list_dataset_dirs()]
        selected = choices[0] if choices else None
        desc = read_meta_description(selected) if selected else ""
        return (
            gr.update(headers=sh, value=sr),
            gr.update(choices=choices, value=selected),
            gr.update(value=desc),
        )

    def _load_desc(name: str):
        from .utils import read_meta_description
        if not name:
            return gr.update(value="")
        return gr.update(value=read_meta_description(name))

    def _save_changes(selected: str, new_name: str, new_desc: str):
        from .utils import update_meta_description, rename_dataset
        msgs: list[str] = []
        # rename if needed
        if selected and new_name and new_name != selected:
            ok, final_name, m = rename_dataset(selected, new_name)
            msgs.append(m)
            if not ok:
                # refresh table & select for consistency
                sh, sr = dataset_summary_table()
                choices = [p.name for p in list_dataset_dirs()]
                return "\n".join(msgs), gr.update(headers=sh, value=sr), gr.update(choices=choices, value=(choices[0] if choices else None))
            selected = final_name
        # update description
        if selected:
            ok, m = update_meta_description(selected, new_desc or "")
            msgs.append(m)
        # refresh
        sh, sr = dataset_summary_table()
        choices = [p.name for p in list_dataset_dirs()]
        return (
            "\n".join(msgs) or "已保存",
            gr.update(headers=sh, value=sr),
            gr.update(choices=choices, value=(selected if selected in choices else (choices[0] if choices else None))),
        )

    def _delete(selected: str):
        from .utils import delete_dataset
        if not selected:
            return "请选择要删除的数据集", gr.update(), gr.update()
        ok, m = delete_dataset(selected)
        # refresh
        sh, sr = dataset_summary_table()
        choices = [p.name for p in list_dataset_dirs()]
        return (
            m,
            gr.update(headers=sh, value=sr),
            gr.update(choices=choices, value=(choices[0] if choices else None)),
        )

    extract_ds_btn.click(
        fn=upload_and_extract,
        inputs=[name_in, desc_in, type_in, ds_archive],
        outputs=[ds_status],
    )
    # After upload, also refresh summary & selector
    extract_ds_btn.click(fn=_refresh_tables, outputs=[ds_table, ds_select, desc_edit])
    refresh_ds_btn.click(fn=_refresh_tables, outputs=[ds_table, ds_select, desc_edit])
    refresh_btn2.click(fn=_refresh_tables, outputs=[ds_table, ds_select, desc_edit])
    # 自动在切换选择时加载描述
    ds_select.change(fn=_load_desc, inputs=[ds_select], outputs=[desc_edit])
    save_btn.click(fn=_save_changes, inputs=[ds_select, new_name_in, desc_edit], outputs=[op_status, ds_table, ds_select])
    del_btn.click(fn=_delete, inputs=[ds_select], outputs=[op_status, ds_table, ds_select])
