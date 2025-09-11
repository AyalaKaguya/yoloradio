"""模型管理模块"""

from __future__ import annotations

# 配置日志
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .paths import MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR

logger = logging.getLogger(__name__)

# 常量定义
MODEL_EXTS = {".pt", ".onnx", ".engine", ".xml", ".bin"}


def _read_yaml_top_level_value(p: Path, key: str) -> Optional[str]:
    """从YAML文件中读取顶级键值"""
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and key in data:
            return str(data[key])
    except Exception:
        return None
    return None


def model_task_code_from_sidecar(model_path: Path) -> Optional[str]:
    """从模型的sidecar文件中获取任务类型代码"""
    side = model_path.with_suffix("")
    yml = side.parent / f"{side.name}.yml"
    return _read_yaml_top_level_value(yml, "task")


def list_models_for_task(task_code: str) -> List[Tuple[str, str]]:
    """返回指定任务类型的模型列表，格式为(标签, 路径)"""
    results: List[Tuple[str, str]] = []
    for is_pre, base in ((True, MODELS_PRETRAINED_DIR), (False, MODELS_TRAINED_DIR)):
        if not base.exists():
            continue
        for f in sorted(base.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_EXTS:
                t = model_task_code_from_sidecar(f)
                if t == task_code:
                    label = ("[预训练] " if is_pre else "[训练] ") + f.name
                    results.append((label, str(f)))
    return results


# Ultralytics预训练模型选择
ULTRALYTICS_PRETRAINED_CHOICES = {
    # detect
    "YOLOv8n (detect)": "yolov8n.pt",
    "YOLOv8s (detect)": "yolov8s.pt",
    "YOLOv8m (detect)": "yolov8m.pt",
    "YOLOv8l (detect)": "yolov8l.pt",
    "YOLOv8x (detect)": "yolov8x.pt",
    # YOLO11
    "YOLO11n (detect)": "yolo11n.pt",
    "YOLO11s (detect)": "yolo11s.pt",
    "YOLO11m (detect)": "yolo11m.pt",
    "YOLO11l (detect)": "yolo11l.pt",
    "YOLO11x (detect)": "yolo11x.pt",
}

# 模型任务映射
MODEL_TASK_MAP = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}


def list_models(dir_path: Path) -> List[str]:
    """获取目录下的模型文件名列表"""
    if not dir_path.exists():
        return []
    return [
        p.name
        for p in sorted(dir_path.iterdir())
        if p.is_file()
        and p.suffix.lower() in {".pt", ".onnx", ".engine", ".xml", ".bin"}
    ]


def get_model_details(dir_path: Path) -> List[List[str]]:
    """获取模型详细信息列表，包含文件名、描述、创建日期"""
    if not dir_path.exists():
        return []

    details = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in {
            ".pt",
            ".onnx",
            ".engine",
            ".xml",
            ".bin",
        }:
            name = p.name

            # 读取描述信息
            meta_file = p.with_suffix("")
            yml_path = meta_file.parent / f"{meta_file.name}.yml"
            description = "无描述"
            if yml_path.exists():
                desc = _read_yaml_top_level_value(yml_path, "description")
                if desc:
                    description = desc

            # 获取创建时间
            try:
                created_time = datetime.fromtimestamp(p.stat().st_mtime)
                created_str = created_time.strftime("%Y-%m-%d %H:%M")
            except Exception:
                created_str = "未知"

            details.append([name, description, created_str])

    return details


def get_all_models_info() -> dict:
    """获取所有模型的完整信息"""
    result = {"pretrained": [], "trained": []}

    # 处理预训练模型
    if MODELS_PRETRAINED_DIR.exists():
        for p in sorted(MODELS_PRETRAINED_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in {
                ".pt",
                ".onnx",
                ".engine",
                ".xml",
                ".bin",
            }:
                model_info = _extract_model_info(p, is_pretrained=True)
                if model_info:
                    result["pretrained"].append(model_info)

    # 处理训练模型
    if MODELS_TRAINED_DIR.exists():
        for p in sorted(MODELS_TRAINED_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in {
                ".pt",
                ".onnx",
                ".engine",
                ".xml",
                ".bin",
            }:
                model_info = _extract_model_info(p, is_pretrained=False)
                if model_info:
                    result["trained"].append(model_info)

    return result


def _extract_model_info(model_path: Path, is_pretrained: bool) -> Optional[dict]:
    """提取单个模型的完整信息"""
    try:
        # 基本文件信息
        filename = model_path.name
        stem = model_path.stem

        # 文件统计信息
        stat = model_path.stat()
        file_size = stat.st_size / (1024 * 1024)  # MB
        created_time = datetime.fromtimestamp(stat.st_mtime)

        # 读取元数据
        yml_path = model_path.with_suffix(".yml")
        metadata = {}
        description = "无描述"
        if yml_path.exists():
            try:
                with open(yml_path, "r", encoding="utf-8") as f:
                    metadata = yaml.safe_load(f) or {}
                # 获取描述，如果太长则在表格中显示为 "|"
                desc = metadata.get("description", "")
                if desc:
                    description = "|" if len(desc) > 50 else desc
            except Exception as e:
                logger.warning(f"读取模型元数据失败 {yml_path}: {e}")

        return {
            "filename": filename,
            "stem": stem,
            "file_size_mb": file_size,
            "created_time": created_time,
            "created_str": created_time.strftime("%Y-%m-%d %H:%M"),
            "description": description,
            "full_description": metadata.get("description", ""),
            "metadata": metadata,
            "is_pretrained": is_pretrained,
            "model_path": model_path,
            "yml_path": yml_path,
        }
    except Exception as e:
        logger.error(f"提取模型信息失败 {model_path}: {e}")
        return None


def get_model_detail(model_name: str, is_pretrained: bool = True) -> str:
    """获取模型的详细信息"""
    if not model_name:
        return ""

    # 选择目录
    base_dir = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR

    # 查找模型文件
    model_file = None
    for suffix in [".pt", ".onnx", ".engine", ".xml", ".bin"]:
        candidate = base_dir / f"{model_name}{suffix}"
        if candidate.exists():
            model_file = candidate
            break

    if not model_file:
        return f"❌ 模型文件不存在: {model_name}"

    # 读取元数据
    yml_path = model_file.with_suffix(".yml")
    metadata = {}
    if yml_path.exists():
        try:
            with open(yml_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"读取模型元数据失败 {yml_path}: {e}")

    # 获取文件信息
    try:
        file_size = model_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        created_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        modified_time = datetime.fromtimestamp(model_file.stat().st_ctime)
    except Exception:
        size_mb = 0
        created_time = datetime.now()
        modified_time = datetime.now()

    # 构建详细信息
    info_lines = [
        f"# 📦 模型详细信息",
        f"",
        f"**模型名称:** {model_name}",
        f"**文件路径:** `{model_file}`",
        f"**文件大小:** {size_mb:.2f} MB",
        f"**创建时间:** {created_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**修改时间:** {modified_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模型类型:** {'预训练模型' if is_pretrained else '训练模型'}",
        f"",
    ]

    # 添加元数据信息
    if metadata:
        info_lines.append("## 📋 元数据信息")

        # 基本信息
        if "task" in metadata:
            task_display = metadata.get("task_display", metadata["task"])
            info_lines.append(f"**任务类型:** {task_display}")

        if "version" in metadata:
            info_lines.append(f"**YOLO版本:** {metadata['version']}")

        if "size" in metadata:
            info_lines.append(f"**模型大小:** {metadata['size']}")

        if "description" in metadata:
            desc = metadata["description"]
            info_lines.append(f"**描述:**")
            # 处理多行描述
            if isinstance(desc, str):
                # 清理描述内容，去掉多余的空行和空格
                cleaned_desc = desc.strip()
                for line in cleaned_desc.split("\n"):
                    line = line.strip()
                    if line:  # 只添加非空行
                        info_lines.append(f"> {line}")
            else:
                info_lines.append(f"> {desc}")

        # 训练相关信息（仅训练模型）
        if not is_pretrained:
            if "base_model" in metadata:
                info_lines.append(f"**基础模型:** {metadata['base_model']}")
            if "dataset" in metadata:
                info_lines.append(f"**训练数据集:** {metadata['dataset']}")
            if "epochs_trained" in metadata:
                info_lines.append(f"**训练轮次:** {metadata['epochs_trained']}")
            if "training_date" in metadata:
                info_lines.append(f"**训练日期:** {metadata['training_date']}")
            if "model_type" in metadata:
                model_type_display = {"best": "最佳权重", "latest": "最新权重"}.get(
                    metadata["model_type"], metadata["model_type"]
                )
                info_lines.append(f"**权重类型:** {model_type_display}")

        info_lines.append("")

        # 完整元数据
        info_lines.append("## 🔧 完整元数据")
        info_lines.append("```yaml")
        try:
            yaml_content = yaml.safe_dump(
                metadata, allow_unicode=True, default_flow_style=False
            )
            info_lines.append(yaml_content)
        except Exception:
            info_lines.append("无法显示元数据")
        info_lines.append("```")
    else:
        info_lines.append("## ⚠️ 元数据信息")
        info_lines.append("")
        info_lines.append("此模型没有关联的元数据文件(.yml)")

    return "\n\n".join(info_lines)


def refresh_model_lists() -> tuple[list[str], list[str]]:
    """获取模型名称列表 (用于下拉框)"""
    return list_models(MODELS_PRETRAINED_DIR), list_models(MODELS_TRAINED_DIR)


def refresh_model_details() -> tuple[list[list[str]], list[list[str]]]:
    """获取模型详细信息列表 (用于表格显示)"""
    return get_model_details(MODELS_PRETRAINED_DIR), get_model_details(
        MODELS_TRAINED_DIR
    )


def download_pretrained_if_missing(name_or_path: str) -> tuple[bool, str, str]:
    """通过ultralytics下载预训练模型"""
    try:
        from ultralytics import YOLO

        m = YOLO(name_or_path)
        ckpt_path = getattr(m, "ckpt_path", None)
        if not ckpt_path:
            ckpt_path = name_or_path

        src = Path(ckpt_path)
        if not src.exists():
            return False, "未找到下载后的权重文件", ""

        MODELS_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_PRETRAINED_DIR / src.name

        if not dest.exists():
            shutil.copy2(src, dest)

        return True, f"已准备预训练模型: {dest.name}", dest.name

    except Exception as e:
        return False, f"下载失败：{e}", ""


def delete_model(is_pretrained: bool, filename: str) -> tuple[bool, str]:
    """删除模型文件"""
    base = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
    p = base / filename
    if not p.exists():
        return False, "文件不存在"

    try:
        p.unlink()
        # 删除对应的YAML文件
        side = base / f"{p.stem}.yml"
        if side.exists():
            try:
                side.unlink()
            except Exception:
                pass
        return True, "已删除"
    except Exception as e:
        return False, f"删除失败：{e}"


def rename_model(is_pretrained: bool, old: str, new: str) -> tuple[bool, str, str]:
    """重命名模型文件"""
    base = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
    src = base / old
    if not src.exists():
        return False, old, "文件不存在"

    dest = base / new
    if dest.exists():
        # 创建唯一名称
        i = 1
        stem, suf = dest.stem, dest.suffix
        while (base / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = base / f"{stem}_{i}{suf}"

    try:
        shutil.move(str(src), str(dest))

        # 重命名对应的YAML文件
        old_yaml = base / f"{src.stem}.yml"
        new_yaml = base / f"{dest.stem}.yml"
        if old_yaml.exists():
            try:
                shutil.move(str(old_yaml), str(new_yaml))
            except Exception:
                pass

        return True, dest.name, "已重命名"
    except Exception as e:
        return False, old, f"重命名失败：{e}"


def model_meta_path(is_pretrained: bool, stem: str) -> Path:
    """获取模型元数据文件路径"""
    base = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
    return base / f"{stem}.yml"


def build_model_metadata_yaml(
    name: str,
    task_display: str,
    task_code: str,
    version: str,
    size: str,
    description: str,
) -> str:
    """构建模型元数据YAML"""
    created_at = datetime.utcnow().isoformat() + "Z"
    data = {
        "name": name,
        "task": task_code,
        "task_display": task_display,
        "version": version,
        "size": size,
        "description": description,
        "created_at": created_at,
    }
    return yaml.safe_dump(data, allow_unicode=True, default_flow_style=False)


def compute_ultra_weight_name(task_code: str, version: str, size: str) -> str:
    """计算Ultralytics权重文件名"""
    v = version.lower().strip()
    if v not in {"v8", "11"}:
        v = "v8"
    prefix = "yolov8" if v == "v8" else "yolo11"

    suffix = {
        "classify": "",
        "detect": "",
        "segment": "-seg",
        "pose": "-pose",
        "obb": "-obb",
    }.get(task_code, "")

    size_code = size.lower().strip()[0]  # n/s/m/l/x
    return f"{prefix}{size_code}{suffix}.pt"


def download_and_register_pretrained(
    task_display: str,
    version: str,
    size: str,
    custom_name: str,
    description: str,
    progress_cb=None,
) -> tuple[bool, str]:
    """下载并注册预训练模型"""
    task_code = MODEL_TASK_MAP.get(task_display, "detect")
    weight = compute_ultra_weight_name(task_code, version, size)

    if progress_cb:
        try:
            progress_cb(0.1, f"准备下载 {weight}")
        except Exception:
            pass

    ok, msg, saved = download_pretrained_if_missing(weight)
    if not ok:
        return False, msg

    # 重命名文件
    src = MODELS_PRETRAINED_DIR / saved
    final_file = src
    if custom_name:
        desired = MODELS_PRETRAINED_DIR / (custom_name.strip() + src.suffix)
        if desired.exists():
            i = 1
            while (
                MODELS_PRETRAINED_DIR / f"{custom_name.strip()}_{i}{src.suffix}"
            ).exists():
                i += 1
            desired = MODELS_PRETRAINED_DIR / f"{custom_name.strip()}_{i}{src.suffix}"
        shutil.move(str(src), str(desired))
        final_file = desired

    # 写入元数据
    stem = final_file.stem
    meta_yaml = build_model_metadata_yaml(
        name=stem,
        task_display=task_display,
        task_code=task_code,
        version=version,
        size=size,
        description=description or "",
    )
    model_meta_path(True, stem).write_text(meta_yaml, encoding="utf-8")

    if progress_cb:
        try:
            progress_cb(1.0, f"完成：{final_file.name}")
        except Exception:
            pass

    return True, f"已准备预训练模型：{final_file.name}"


__all__ = [
    "ULTRALYTICS_PRETRAINED_CHOICES",
    "MODEL_TASK_MAP",
    "list_models",
    "get_model_details",
    "refresh_model_lists",
    "refresh_model_details",
    "download_pretrained_if_missing",
    "delete_model",
    "rename_model",
    "download_and_register_pretrained",
    "build_model_metadata_yaml",
]
