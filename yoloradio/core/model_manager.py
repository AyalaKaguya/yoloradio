"""模型管理模块"""

from __future__ import annotations

# 配置日志
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .paths import MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR

logger = logging.getLogger(__name__)

# 常量定义
MODEL_EXTS = {".pt", ".onnx", ".engine", ".xml", ".bin"}


def _read_yaml_top_level_value(p: Path, key: str) -> Optional[str]:
    """从YAML文件中读取顶级键值"""
    if not p.exists():
        return None
    try:
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith(f"{key}:"):
                val = s.split(":", 1)[1].strip()
                # strip possible quotes
                if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                    val = val[1:-1]
                return val
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
    lines = [
        f"name: {name}",
        f"task: {task_code}",
        f"task_display: {task_display}",
        f"version: {version}",
        f"size: {size}",
        "description: |",
    ]
    for ln in description.splitlines() or [""]:
        lines.append(f"  {ln}")
    lines.append(f"created_at: {created_at}")
    return "\n".join(lines) + "\n"


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
