"""数据集管理页面模块"""

from __future__ import annotations

from typing import List

import gradio as gr

from ..core import (
    dataset_summary_table,
    delete_dataset,
    ensure_unique_dir,
    extract_pathlike,
    get_dataset_detail,
    is_supported_archive,
    list_datasets_for_task,
    list_dir,
    read_meta_description,
    rename_dataset,
    safe_extract_tar,
    safe_extract_zip,
    strip_archive_suffix,
    summarize_dataset,
    unwrap_single_root,
    validate_dataset_by_type,
)
from ..core.paths import DATASETS_DIR

# 数据集类型映射
DATASET_TYPE_MAP = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}


def list_dataset_dirs():
    """列出数据集目录"""
    if not DATASETS_DIR.exists():
        return []
    return [d for d in DATASETS_DIR.iterdir() if d.is_dir()]


def build_metadata_yaml(
    name: str,
    type_code: str,
    type_display: str,
    description: str,
    structure: str,
    splits: dict,
):
    """构建元数据YAML"""
    return f"""name: {name}
type: {type_code}
type_display: {type_display}
description: {description}
structure: {structure}
splits: {splits}
"""


def read_meta_description(name: str) -> str:
    """读取数据集描述"""
    if not name:
        return ""
    meta_path = DATASETS_DIR / f"{name}.yml"
    if not meta_path.exists():
        return ""

    try:
        content = meta_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("description:"):
                desc = line.split(":", 1)[1].strip()
                if desc.startswith('"') and desc.endswith('"'):
                    desc = desc[1:-1]
                return desc
    except:
        pass
    return ""


def update_meta_description(name: str, description: str) -> tuple[bool, str]:
    """更新数据集描述"""
    meta_path = DATASETS_DIR / f"{name}.yml"
    if not meta_path.exists():
        return False, f"元数据文件不存在: {name}.yml"

    try:
        lines = meta_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        desc_updated = False

        for line in lines:
            if line.strip().startswith("description:"):
                new_lines.append(f'description: "{description}"')
                desc_updated = True
            else:
                new_lines.append(line)

        if not desc_updated:
            new_lines.append(f'description: "{description}"')

        meta_path.write_text("\n".join(new_lines), encoding="utf-8")
        return True, "描述已更新"
    except Exception as e:
        return False, f"更新失败: {e}"


def create_datasets_tab() -> None:
    """创建数据集管理标签页"""
    gr.Markdown(
        f"""
        数据集目录: `{DATASETS_DIR}`  
        推荐上传压缩包（.zip/.tar.gz/.tgz/.tar.bz2 等），服务器将自动解压到 `Datasets/子目录`。
        体积较大的数据集建议直接在文件系统中放入 `Datasets/`，以避免网页上传耗时。
        """
    )
    with gr.Row():
        # 左侧窄栏：信息与上传 + 管理面板
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("### 信息与上传")
            name_in = gr.Textbox(
                label="数据集名称", placeholder="例如：screw_long_obbkp_dataset"
            )
            desc_in = gr.Textbox(
                label="数据集描述", placeholder="可填写来源、标注说明等", lines=3
            )
            type_in = gr.Dropdown(
                choices=list(DATASET_TYPE_MAP.keys()),
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
            _choices = [p.name for p in list_dataset_dirs()]
            ds_select = gr.Dropdown(
                choices=_choices,
                value=(_choices[0] if _choices else None),
                label="选择数据集",
                allow_custom_value=True,
            )
            new_name_in = gr.Textbox(label="新名称", placeholder="留空则不改名")
            # 初始描述：根据当前选择自动加载
            _init_desc = read_meta_description(_choices[0]) if _choices else ""
            desc_edit = gr.Textbox(label="描述", lines=4, value=_init_desc)
            with gr.Row():
                save_btn = gr.Button("保存修改", variant="primary")
                del_btn = gr.Button("删除所选", variant="stop")
            op_status = gr.Markdown(visible=True)

        # 右侧宽栏：显示汇总列表
        with gr.Column(scale=2, min_width=760):
            # 汇总
            with gr.Row():
                gr.Markdown("### 数据集列表")
                refresh_btn2 = gr.Button("刷新")
            sum_headers, sum_rows = dataset_summary_table()
            ds_table = gr.Dataframe(
                headers=sum_headers, value=sum_rows, interactive=False, wrap=True
            )

            # 详情查看区域
            gr.Markdown("### 数据集详情")
            with gr.Row():
                detail_ds_select = gr.Dropdown(
                    choices=[p.name for p in list_dataset_dirs()],
                    label="选择数据集查看详情",
                    value=None,
                )
                show_detail_btn = gr.Button("查看详情", variant="secondary")

            detail_info = gr.Markdown("请选择数据集查看详细信息", visible=True)

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

                mode = (
                    "r:gz"
                    if "gz" in "".join(p.suffixes).lower()
                    or p.name.lower().endswith(".tgz")
                    else None
                )
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
            # 无法判定为有效数据集，清理空目录
            try:
                if not any(dest_dir.rglob("*")):
                    dest_dir.rmdir()
            except Exception:
                pass
            return f"数据集校验失败：{msg}"

        # 写入同名 yml 元数据
        meta_yaml = build_metadata_yaml(
            name=folder_name,
            type_code=type_code,
            type_display=dtype,
            description=desc or "",
            structure=structure,
            splits=splits,
        )
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
            gr.update(choices=choices, value=selected),  # 详情选择下拉框
        )

    def _load_desc(name: str):
        if not name:
            return gr.update(value="")
        return gr.update(value=read_meta_description(name))

    def _save_changes(selected: str, new_name: str, new_desc: str):
        msgs: list[str] = []
        # rename if needed
        if selected and new_name and new_name != selected:
            ok, final_name, m = rename_dataset(selected, new_name)
            msgs.append(m)
            if not ok:
                # refresh table & select for consistency
                sh, sr = dataset_summary_table()
                choices = [p.name for p in list_dataset_dirs()]
                return (
                    "\n".join(msgs),
                    gr.update(headers=sh, value=sr),
                    gr.update(choices=choices, value=(choices[0] if choices else None)),
                )
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
            gr.update(
                choices=choices,
                value=(
                    selected
                    if selected in choices
                    else (choices[0] if choices else None)
                ),
            ),
        )

    def _delete(selected: str):
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

    def _show_dataset_detail(name: str):
        """显示数据集详细信息"""
        if not name:
            return "请选择数据集"

        detail = get_dataset_detail(name)
        if not detail:
            return f"无法获取数据集 '{name}' 的详细信息"

        # 格式化详细信息
        info_lines = [
            f"# 数据集详情: {detail['name']}",
            "",
            f"**路径**: `{detail['path']}`",
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

    def _refresh_detail_choices():
        """刷新详情选择下拉框"""
        choices = [p.name for p in list_dataset_dirs()]
        return gr.update(choices=choices, value=choices[0] if choices else None)

    extract_ds_btn.click(
        fn=upload_and_extract,
        inputs=[name_in, desc_in, type_in, ds_archive],
        outputs=[ds_status],
    )
    # After upload, also refresh summary & selector
    extract_ds_btn.click(
        fn=_refresh_tables, outputs=[ds_table, ds_select, desc_edit, detail_ds_select]
    )
    refresh_btn2.click(
        fn=_refresh_tables, outputs=[ds_table, ds_select, desc_edit, detail_ds_select]
    )
    # 自动在切换选择时加载描述
    ds_select.change(fn=_load_desc, inputs=[ds_select], outputs=[desc_edit])
    save_btn.click(
        fn=_save_changes,
        inputs=[ds_select, new_name_in, desc_edit],
        outputs=[op_status, ds_table, ds_select],
    )
    del_btn.click(
        fn=_delete, inputs=[ds_select], outputs=[op_status, ds_table, ds_select]
    )

    # 详情查看事件绑定
    show_detail_btn.click(
        fn=_show_dataset_detail, inputs=[detail_ds_select], outputs=[detail_info]
    )
    detail_ds_select.change(
        fn=_show_dataset_detail, inputs=[detail_ds_select], outputs=[detail_info]
    )


# 保持向后兼容性
def render() -> None:
    """向后兼容的渲染函数"""
    create_datasets_tab()
