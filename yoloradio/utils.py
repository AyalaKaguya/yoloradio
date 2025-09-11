"""YoloRadio 工具函数模块

提供文件操作、数据集处理、模型管理等核心功能。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from .paths import DATASETS_DIR, MODELS_DIR, MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR

# 配置日志
logger = logging.getLogger(__name__)


def list_dir(
    dir_path: Path, exts: tuple[str, ...] | None = None, max_items: int = 100
) -> List[str]:
    items: List[str] = []
    if not dir_path.exists():
        return [f"目录不存在: {dir_path}"]
    try:
        for child in sorted(dir_path.iterdir()):
            if child.is_dir():
                items.append(f"📁 {child.name}/")
            else:
                if exts is None or child.suffix.lower() in exts:
                    items.append(f"📄 {child.name}")
            if len(items) >= max_items:
                items.append("…（已截断）")
                break
    except PermissionError:
        items.append("无权限读取该目录")
    return items if items else ["（空）"]


def extract_pathlike(obj: Any) -> Optional[Path]:
    """从各种对象中提取路径对象

    Args:
        obj: 可能包含路径信息的对象

    Returns:
        提取到的路径对象，失败返回 None
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, str):
            return Path(obj)
        if isinstance(obj, bytes):
            try:
                return Path(obj.decode("utf-8"))
            except UnicodeDecodeError:
                logger.warning(f"无法解码字节对象: {obj}")
                return None
        if isinstance(obj, os.PathLike):
            fs = os.fspath(obj)
            if isinstance(fs, bytes):
                try:
                    return Path(fs.decode("utf-8"))
                except UnicodeDecodeError:
                    logger.warning(f"无法解码路径对象: {fs}")
                    return None
            return Path(cast(str, fs))

        # 尝试从对象属性获取路径
        path_attr = getattr(obj, "name", None) or getattr(obj, "path", None)
        if isinstance(path_attr, str):
            return Path(path_attr)

        # 尝试从字典获取路径
        if isinstance(obj, dict):
            p = obj.get("name") or obj.get("path")
            if isinstance(p, str):
                return Path(p)

    except Exception as e:
        logger.error(f"提取路径失败: {e}")
        return None
    return None


def safe_move(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = src.name
    dest = dst_dir / name
    if dest.exists():
        stem, suf = dest.stem, dest.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{suf}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    shutil.move(str(src), str(dest))
    return dest.name


def is_supported_archive(p: Path) -> bool:
    suffs = "".join(p.suffixes).lower()
    if suffs.endswith(".zip"):
        return True
    if any(
        suffs.endswith(x)
        for x in [".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar", ".tar.xz", ".txz"]
    ):
        return True
    return False


def strip_archive_suffix(name: str) -> str:
    lower = name.lower()
    for suf in [
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".tgz",
        ".tbz",
        ".txz",
        ".zip",
        ".tar",
    ]:
        if lower.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem


def ensure_unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.parent / f"{base.name}_{i}"
        if not cand.exists():
            return cand
        i += 1


def safe_extract_zip(zp: zipfile.ZipFile, dest: Path):
    root = dest.resolve()
    for member in zp.infolist():
        mpath = Path(member.filename)
        target = (root / mpath).resolve()
        if not str(target).startswith(str(root)):
            continue
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zp.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def safe_extract_tar(tf: tarfile.TarFile, dest: Path):
    root = dest.resolve()
    for member in tf.getmembers():
        mpath = Path(member.name)
        target = (root / mpath).resolve()
        if not str(target).startswith(str(root)):
            continue
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            with f as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


# ---------- Dataset summarization utilities ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_files(dir_path: Path, exts: set[str]) -> int:
    if not dir_path.exists():
        return 0
    return sum(
        1 for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def _count_label_txt(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*.txt") if p.is_file())


def summarize_dataset(ds_dir: Path) -> dict:
    """Return dataset summary with split-level counts.
    Keys: name, total_images, total_labels, train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    name = ds_dir.name
    # 支持两种结构：
    # 1) 分割结构：train/val/test 下有 images/ 与 labels/
    # 2) 扁平结构：数据集根目录下直接有 images/ 与 labels/
    has_split = any((ds_dir / s).exists() for s in ("train", "val", "test"))
    if has_split:
        splits = {
            "train": (ds_dir / "train" / "images", ds_dir / "train" / "labels"),
            "val": (ds_dir / "val" / "images", ds_dir / "val" / "labels"),
            "test": (ds_dir / "test" / "images", ds_dir / "test" / "labels"),
        }
    else:
        # 将根目录视作 train，其余为空
        splits = {
            "train": (ds_dir / "images", ds_dir / "labels"),
            "val": (ds_dir / "val_images", ds_dir / "val_labels"),  # 兼容性占位
            "test": (ds_dir / "test_images", ds_dir / "test_labels"),  # 兼容性占位
        }
    counts = {}
    total_images = 0
    total_labels = 0
    for split, (img_dir, lbl_dir) in splits.items():
        img_cnt = _count_files(img_dir, IMG_EXTS)
        lbl_cnt = _count_label_txt(lbl_dir)
        counts[f"{split}_images"] = img_cnt
        counts[f"{split}_labels"] = lbl_cnt
        total_images += img_cnt
        total_labels += lbl_cnt
    return {
        "name": name,
        "total_images": total_images,
        "total_labels": total_labels,
        **counts,
    }


def list_dataset_dirs() -> List[Path]:
    if not DATASETS_DIR.exists():
        return []
    return [p for p in sorted(DATASETS_DIR.iterdir()) if p.is_dir()]


def dataset_summary_table() -> tuple[list[str], list[list]]:
    """Return headers and rows suitable for gr.Dataframe to list dataset summaries."""
    headers = [
        "数据集",
        "总样本",
        "总标注",
        "train 样本",
        "val 样本",
        "test 样本",
        "train 标注",
        "val 标注",
        "test 标注",
    ]
    rows: list[list] = []
    for ds in list_dataset_dirs():
        s = summarize_dataset(ds)
        rows.append(
            [
                s["name"],
                s["total_images"],
                s["total_labels"],
                s["train_images"],
                s["val_images"],
                s["test_images"],
                s["train_labels"],
                s["val_labels"],
                s["test_labels"],
            ]
        )
    return headers, rows


# ---------- Dataset import helpers ----------

DATASET_TYPE_MAP: Dict[str, str] = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",  # Ultralytics 中关键点任务为 pose
    "旋转检测框识别": "obb",
}

# 模型任务映射与数据集一致
MODEL_TASK_MAP = DATASET_TYPE_MAP


def _only_one_dir_no_files(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    files = [p for p in base.iterdir() if p.is_file()]
    if len(dirs) == 1 and len(files) == 0:
        return dirs[0]
    return None


def unwrap_single_root(dest_dir: Path, max_depth: int = 3) -> None:
    """If dest_dir contains a single directory and no files, move its contents up repeatedly."""
    depth = 0
    while depth < max_depth:
        inner = _only_one_dir_no_files(dest_dir)
        if inner is None:
            break
        # move children of inner to dest_dir
        for item in inner.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        try:
            inner.rmdir()
        except OSError:
            pass
        depth += 1


def find_split_dirs(base: Path) -> Dict[str, Tuple[Path, Path]]:
    splits = {
        "train": (base / "train" / "images", base / "train" / "labels"),
        "val": (base / "val" / "images", base / "val" / "labels"),
        "test": (base / "test" / "images", base / "test" / "labels"),
    }
    return splits


def detect_structure(base: Path) -> Tuple[str, Dict[str, Tuple[Path, Path]]]:
    """Return (structure, splits) where structure in {"split","flat"}."""
    # split
    split_dirs = find_split_dirs(base)
    if any(img.exists() or lbl.exists() for img, lbl in split_dirs.values()):
        return "split", split_dirs
    # flat
    flat = {
        "train": (base / "images", base / "labels"),
        "val": (base / "val_images", base / "val_labels"),
        "test": (base / "test_images", base / "test_labels"),
    }
    if any(img.exists() or lbl.exists() for img, lbl in flat.values()):
        return "flat", flat
    return "unknown", {}


def _has_any_images(p: Path) -> bool:
    return _count_files(p, IMG_EXTS) > 0


def _has_any_labels(p: Path) -> bool:
    return _count_label_txt(p) > 0


def validate_dataset_by_type(
    base: Path, ds_type_code: str
) -> Tuple[bool, str, str, Dict[str, Tuple[Path, Path]]]:
    """Minimal validation. Returns (ok, structure, message, splits)."""
    structure, splits = detect_structure(base)
    if structure == "unknown":
        return (
            False,
            structure,
            "无法识别数据集结构（未发现 images/ 或 labels/ 目录）",
            {},
        )

    # For each split, check presence based on task
    def split_ok(img_dir: Path, lbl_dir: Path) -> bool:
        if ds_type_code == "classify":
            # 分类不强制 labels，要求有图片即可
            return _has_any_images(img_dir)
        else:
            # 其他任务：要求 images 存在且 labels 至少存在一些
            return _has_any_images(img_dir) and _has_any_labels(lbl_dir)

    good = False
    for img_dir, lbl_dir in splits.values():
        if split_ok(img_dir, lbl_dir):
            good = True
            break
    if not good:
        if ds_type_code == "classify":
            return False, structure, "未检测到任何图片文件", splits
        else:
            return False, structure, "未检测到匹配的图片与标注(txt)文件", splits

    return True, structure, "检测到有效的数据集结构", splits


def build_metadata_yaml(
    name: str,
    type_code: str,
    type_display: str,
    description: str,
    structure: str,
    splits: Dict[str, Tuple[Path, Path]],
) -> str:
    """Create a simple YAML string without external deps."""
    created_at = datetime.utcnow().isoformat() + "Z"

    # relativeize paths to dataset root
    def rel(p: Path) -> str:
        return str(p).replace("\\", "/")

    lines: List[str] = []
    lines.append(f"name: {name}")
    lines.append(f"type: {type_code}")
    lines.append(f"type_display: {type_display}")
    lines.append("description: |")
    for ln in description.splitlines() or [""]:
        lines.append(f"  {ln}")
    lines.append(f"created_at: {created_at}")
    lines.append(f"structure: {structure}")
    lines.append("splits:")
    for s, (img_dir, lbl_dir) in splits.items():
        lines.append(f"  {s}:")
        lines.append(
            f"    images: {rel(Path(img_dir.name) if img_dir.is_absolute() else img_dir)}"
        )
        lines.append(
            f"    labels: {rel(Path(lbl_dir.name) if lbl_dir.is_absolute() else lbl_dir)}"
        )
    return "\n".join(lines) + "\n"


def dataset_detail_table(ds_name: str) -> tuple[list[str], list[list]]:
    """Return split-level detail table for selected dataset name."""
    ds_dir = DATASETS_DIR / ds_name
    headers = ["分割", "样本数", "标注数"]
    if not ds_dir.exists():
        return headers, []
    s = summarize_dataset(ds_dir)
    rows = [
        ["train", s["train_images"], s["train_labels"]],
        ["val", s["val_images"], s["val_labels"]],
        ["test", s["test_images"], s["test_labels"]],
        ["总计", s["total_images"], s["total_labels"]],
    ]
    return headers, rows


# ---------- Dataset management (metadata, rename, delete) ----------


def meta_path_for(name: str) -> Path:
    return DATASETS_DIR / f"{name}.yml"


def read_meta_description(name: str) -> str:
    p = meta_path_for(name)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return ""
    lines = text.splitlines()
    desc = []
    i = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("description:"):
            i = idx
            break
    if i is None:
        return ""
    # If multiline (|) then capture indented block
    if "|" in lines[i]:
        for j in range(i + 1, len(lines)):
            ln = lines[j]
            if ln.startswith("  "):
                desc.append(ln[2:])
            else:
                break
        return "\n".join(desc).rstrip("\n")
    # single-line value (unlikely in our writer)
    val = lines[i].split(":", 1)[1].strip()
    return val


def update_meta_description(name: str, new_desc: str) -> tuple[bool, str]:
    p = meta_path_for(name)
    if not p.exists():
        return False, "未找到元数据文件"
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"读取元数据失败：{e}"
    lines = text.splitlines()
    # find description block
    i = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("description:"):
            i = idx
            break
    block = ["description: |"]
    for ln in new_desc.splitlines() or [""]:
        block.append(f"  {ln}")
    if i is None:
        # insert after type_display or after name
        insert_idx = 0
        for idx, ln in enumerate(lines):
            if ln.strip().startswith("type_display:"):
                insert_idx = idx + 1
                break
            if ln.strip().startswith("name:"):
                insert_idx = idx + 1
        new_lines = lines[:insert_idx] + block + lines[insert_idx:]
    else:
        # replace existing description block (including indented lines)
        j = i + 1
        while j < len(lines) and lines[j].startswith("  "):
            j += 1
        new_lines = lines[:i] + block + lines[j:]
    try:
        p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as e:
        return False, f"写入元数据失败：{e}"
    return True, "已更新描述"


def _replace_yaml_name(text: str, new_name: str) -> str:
    lines = text.splitlines()
    for idx, ln in enumerate(lines):
        if ln.startswith("name:"):
            lines[idx] = f"name: {new_name}"
            break
    return "\n".join(lines) + "\n"


def rename_dataset(old_name: str, new_name: str) -> tuple[bool, str, str]:
    """Rename dataset directory and its metadata file. Returns (ok, final_name, message)."""
    src_dir = DATASETS_DIR / old_name
    if not src_dir.exists() or not src_dir.is_dir():
        return False, old_name, "数据集目录不存在"
    desired = DATASETS_DIR / new_name
    if desired.exists():
        desired = ensure_unique_dir(desired)
    final_name = desired.name
    try:
        shutil.move(str(src_dir), str(desired))
    except Exception as e:
        return False, old_name, f"重命名目录失败：{e}"
    # rename meta
    old_meta = meta_path_for(old_name)
    new_meta = meta_path_for(final_name)
    if old_meta.exists():
        try:
            # also update name: field
            try:
                txt = old_meta.read_text(encoding="utf-8")
                txt2 = _replace_yaml_name(txt, final_name)
            except Exception:
                txt2 = None
            shutil.move(str(old_meta), str(new_meta))
            if txt2 is not None:
                new_meta.write_text(txt2, encoding="utf-8")
        except Exception as e:
            return False, old_name, f"重命名元数据失败：{e}"
    return True, final_name, "已重命名"


def delete_dataset(name: str) -> tuple[bool, str]:
    ds_dir = DATASETS_DIR / name
    meta = meta_path_for(name)
    if not ds_dir.exists():
        return False, "数据集目录不存在"
    try:
        shutil.rmtree(ds_dir, ignore_errors=False)
    except Exception as e:
        return False, f"删除目录失败：{e}"
    try:
        if meta.exists():
            meta.unlink()
    except Exception:
        pass
    return True, "已删除"


# ---------- Model utilities ----------

ULTRALYTICS_PRETRAINED_CHOICES = {
    # detect
    "YOLOv8n (detect)": "yolov8n.pt",
    "YOLOv8s (detect)": "yolov8s.pt",
    "YOLOv8m (detect)": "yolov8m.pt",
    "YOLOv8l (detect)": "yolov8l.pt",
    "YOLOv8x (detect)": "yolov8x.pt",
    # new yolo11
    "YOLO11n (detect)": "yolo11n.pt",
    "YOLO11s (detect)": "yolo11s.pt",
    "YOLO11m (detect)": "yolo11m.pt",
    "YOLO11l (detect)": "yolo11l.pt",
    "YOLO11x (detect)": "yolo11x.pt",
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
    """Download via ultralytics and copy weight file into Models/pretrained."""
    try:
        from ultralytics import YOLO

        m = YOLO(name_or_path)
        # YOLO(...).ckpt_path points to downloaded weight file path
        ckpt_path = getattr(m, "ckpt_path", None)
        if not ckpt_path:
            # fallback: the path argument could be a local file already
            ckpt_path = name_or_path
        src = Path(ckpt_path)
        if not src.exists():
            return False, "未找到下载后的权重文件", ""
        MODELS_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_PRETRAINED_DIR / src.name
        # If file already exists, keep existing
        if not dest.exists():
            shutil.copy2(src, dest)
        return True, f"已准备预训练模型: {dest.name}", dest.name
    except Exception as e:
        return False, f"下载失败：{e}", ""


def delete_model(is_pretrained: bool, filename: str) -> tuple[bool, str]:
    base = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
    p = base / filename
    if not p.exists():
        return False, "文件不存在"
    try:
        p.unlink()
        # remove sidecar yaml if present
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
    base = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
    src = base / old
    if not src.exists():
        return False, old, "文件不存在"
    dest = base / new
    if dest.exists():
        # create a unique name
        i = 1
        stem, suf = dest.stem, dest.suffix
        while (base / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = base / f"{stem}_{i}{suf}"
    try:
        shutil.move(str(src), str(dest))
        # also rename YAML sidecar if present
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
    # Examples: detect: yolov8n.pt / yolo11n.pt ; segment: yolov8n-seg.pt ; pose: yolov8n-pose.pt ; obb: yolov8n-obb.pt
    v = version.lower().strip()
    if v not in {"v8", "11"}:
        v = "v8"
    prefix = "yolov8" if v == "v8" else "yolo11"
    suffix = {
        "classify": "",  # ultralytics 分类一般是 *.pt（分类专用也可能是 cls），这里按空后缀处理
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
    task_code = MODEL_TASK_MAP.get(task_display, "detect")
    weight = compute_ultra_weight_name(task_code, version, size)
    # optional progress is not natively exposed by ultralytics; we can only post milestones
    if progress_cb:
        try:
            progress_cb(0.1, f"准备下载 {weight}")
        except Exception:
            pass
    ok, msg, saved = download_pretrained_if_missing(weight)
    if not ok:
        return False, msg
    # rename if user provided custom_name
    src = MODELS_PRETRAINED_DIR / saved
    final_file = src
    if custom_name:
        desired = MODELS_PRETRAINED_DIR / (custom_name.strip() + src.suffix)
        if desired.exists():
            # make unique
            i = 1
            while (
                MODELS_PRETRAINED_DIR / f"{custom_name.strip()}_{i}{src.suffix}"
            ).exists():
                i += 1
            desired = MODELS_PRETRAINED_DIR / f"{custom_name.strip()}_{i}{src.suffix}"
        shutil.move(str(src), str(desired))
        final_file = desired
    # write metadata yaml
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


# ---------- Training helpers (filter datasets/models by task, data.yaml builder) ----------


def _read_yaml_top_level_value(p: Path, key: str) -> Optional[str]:
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


def dataset_type_code(ds_name: str) -> Optional[str]:
    p = meta_path_for(ds_name)
    return _read_yaml_top_level_value(p, "type")


def list_datasets_for_task(task_code: str) -> List[str]:
    if not DATASETS_DIR.exists():
        return []
    names: List[str] = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            t = dataset_type_code(d.name)
            if t == task_code:
                names.append(d.name)
    return names


def model_task_code_from_sidecar(model_path: Path) -> Optional[str]:
    side = model_path.with_suffix("")
    yml = side.parent / f"{side.name}.yml"
    return _read_yaml_top_level_value(yml, "task")


MODEL_EXTS = {".pt", ".onnx", ".engine", ".xml", ".bin"}


def list_models_for_task(task_code: str) -> list[tuple[str, str]]:
    """Return list of (label, path) filtered by model task code across pretrained/trained dirs."""
    results: list[tuple[str, str]] = []
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


def read_class_names(ds_dir: Path) -> list[str]:
    p = ds_dir / "classes.txt"
    if not p.exists():
        return []
    try:
        lines = [
            ln.strip()
            for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()
        ]
        return [ln for ln in lines if ln]
    except Exception:
        return []


def build_ultra_data_yaml(ds_name: str) -> Path:
    """Create a minimal ultralytics data.yaml beside dataset for training and return its path."""
    ds_dir = DATASETS_DIR / ds_name
    structure, splits = detect_structure(ds_dir)

    # Build paths
    def rel(p: Path) -> str:
        # Paths in ultralytics yaml can be absolute or combined with 'path'
        return str(p.relative_to(ds_dir)) if p.exists() else str(p)

    img_paths = {}
    for s, (img_dir, _lbl_dir) in splits.items():
        img_paths[s] = rel(img_dir)
    # default val/test to train if missing
    if not (ds_dir / img_paths.get("val", "")).exists():
        img_paths["val"] = img_paths.get("train", "")
    if not (ds_dir / img_paths.get("test", "")).exists():
        img_paths["test"] = img_paths.get("val", img_paths.get("train", ""))
    names = read_class_names(ds_dir)
    lines: list[str] = []
    lines.append(f"path: {str(ds_dir).replace('\\','/')}")
    lines.append(f"train: {img_paths.get('train','')}")
    lines.append(f"val: {img_paths.get('val','')}")
    if img_paths.get("test"):
        lines.append(f"test: {img_paths.get('test')}")
    if names:
        lines.append("names:")
        for i, n in enumerate(names):
            # list form
            lines.append(f"  - {n}")
    out = ds_dir / f"{ds_name}.ultra.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---------- Training management ----------


class TrainingState:
    """训练状态管理类 - 单例模式"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingState, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.is_running = False
        self.is_paused = False
        self.process: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.log_lines: List[str] = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.model_path = ""
        self.run_id = ""

    def reset(self):
        """重置训练状态"""
        self.is_running = False
        self.is_paused = False
        self.process = None
        self.thread = None
        self.log_lines = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.model_path = ""
        self.run_id = ""


# 全局训练状态
training_state = TrainingState()


def start_training(
    task_code: str,
    dataset_name: str,
    model_path: str,
    epochs: int,
    lr0: float,
    imgsz: int,
    batch: int,
    device: str = "auto",
    **kwargs,
) -> tuple[bool, str]:
    """开始训练"""
    global training_state

    if training_state.is_running:
        return False, "训练已在进行中"

    try:
        # 检查 ultralytics 是否可用
        try:
            import ultralytics
        except ImportError:
            return False, "未安装 ultralytics，请先安装: pip install ultralytics"

        # 验证输入
        if not Path(model_path).exists():
            return False, f"模型文件不存在: {model_path}"

        if not (DATASETS_DIR / dataset_name).exists():
            return False, f"数据集不存在: {dataset_name}"

        # 生成运行ID
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建数据配置文件
        data_yaml = build_ultra_data_yaml(dataset_name)

        # 准备输出目录
        from .paths import PROJECT_DIR

        runs_dir = PROJECT_DIR / "runs" / "train"
        runs_dir.mkdir(parents=True, exist_ok=True)
        project_dir = runs_dir / f"{task_code}_{timestamp}_{run_id}"

        # 准备训练参数
        train_args = [
            "yolo",
            "train",
            f"data={data_yaml}",
            f"model={model_path}",
            f"epochs={epochs}",
            f"lr0={lr0}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"project={runs_dir}",
            f"name={task_code}_{timestamp}_{run_id}",
            f"device={device}",
            "save=True",
            "plots=True",
            "val=True",
        ]

        # 添加额外参数
        for key, value in kwargs.items():
            if value not in [None, "", "auto"]:
                train_args.append(f"{key}={value}")

        # 重置状态
        training_state.reset()
        training_state.is_running = True
        training_state.total_epochs = epochs
        training_state.run_id = run_id
        training_state.model_path = model_path

        # 启动训练线程
        def run_training():
            try:
                training_state.process = subprocess.Popen(
                    train_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    universal_newlines=True,
                    cwd=PROJECT_DIR,
                )

                # 读取输出
                if training_state.process.stdout:
                    for line in iter(training_state.process.stdout.readline, ""):
                        if not training_state.is_running:
                            break

                        line = line.strip()
                        if line:
                            training_state.log_lines.append(
                                f"[{datetime.now().strftime('%H:%M:%S')}] {line}"
                            )

                            # 解析epoch信息
                            if "Epoch" in line and "/" in line:
                                try:
                                    parts = line.split()
                                    for i, part in enumerate(parts):
                                        if "Epoch" in part and i + 1 < len(parts):
                                            epoch_info = parts[i + 1]
                                            if "/" in epoch_info:
                                                current, total = epoch_info.split("/")
                                                training_state.current_epoch = int(
                                                    current
                                                )
                                                break
                                except:
                                    pass

                # 等待进程结束
                training_state.process.wait()

                # 训练完成后处理模型文件
                if training_state.process.returncode == 0:
                    _handle_training_completion(
                        task_code, dataset_name, model_path, project_dir, timestamp
                    )
                    training_state.log_lines.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 训练完成，模型已保存到 Models/trained/"
                    )
                else:
                    training_state.log_lines.append(
                        f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 训练失败"
                    )

            except Exception as e:
                training_state.log_lines.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 训练错误: {e}"
                )
            finally:
                training_state.is_running = False
                training_state.process = None

        training_state.thread = threading.Thread(target=run_training, daemon=True)
        training_state.thread.start()

        return True, f"训练已开始 (ID: {run_id})"

    except Exception as e:
        training_state.reset()
        return False, f"启动训练失败: {e}"


def _handle_training_completion(
    task_code: str,
    dataset_name: str,
    model_path: str,
    project_dir: Path,
    timestamp: str,
):
    """处理训练完成后的模型保存"""
    try:
        # 查找best和last权重
        weights_dir = project_dir / "weights"
        if not weights_dir.exists():
            return

        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        # 准备模型名称
        base_model_name = Path(model_path).stem
        date_str = datetime.now().strftime("%Y%m%d")

        MODELS_TRAINED_DIR.mkdir(parents=True, exist_ok=True)

        # 复制并重命名模型
        saved_models = []

        if best_pt.exists():
            # UUID文件名
            uuid_name = f"{str(uuid.uuid4())[:8]}.pt"
            uuid_dest = MODELS_TRAINED_DIR / uuid_name
            shutil.copy2(best_pt, uuid_dest)

            # 显示名称
            display_name = f"{task_code}-{base_model_name}-{date_str}-best"

            # 创建元数据
            meta_content = f"""task: {task_code}
name: {display_name}
description: "基于 {base_model_name} 在 {dataset_name} 数据集上训练的最佳权重"
base_model: {base_model_name}
dataset: {dataset_name}
training_date: {datetime.now().isoformat()}
model_type: best
epochs_trained: {training_state.total_epochs}
created_at: {datetime.now().isoformat()}
"""
            meta_path = MODELS_TRAINED_DIR / f"{uuid_dest.stem}.yml"
            meta_path.write_text(meta_content, encoding="utf-8")

            saved_models.append((uuid_name, display_name))

        if last_pt.exists():
            # UUID文件名
            uuid_name = f"{str(uuid.uuid4())[:8]}.pt"
            uuid_dest = MODELS_TRAINED_DIR / uuid_name
            shutil.copy2(last_pt, uuid_dest)

            # 显示名称
            display_name = f"{task_code}-{base_model_name}-{date_str}-latest"

            # 创建元数据
            meta_content = f"""task: {task_code}
name: {display_name}
description: "基于 {base_model_name} 在 {dataset_name} 数据集上训练的最新权重"
base_model: {base_model_name}
dataset: {dataset_name}
training_date: {datetime.now().isoformat()}
model_type: latest
epochs_trained: {training_state.total_epochs}
created_at: {datetime.now().isoformat()}
"""
            meta_path = MODELS_TRAINED_DIR / f"{uuid_dest.stem}.yml"
            meta_path.write_text(meta_content, encoding="utf-8")

            saved_models.append((uuid_name, display_name))

        logger.info(f"训练完成，保存了 {len(saved_models)} 个模型: {saved_models}")

    except Exception as e:
        logger.error(f"处理训练完成失败: {e}")


def pause_training() -> tuple[bool, str]:
    """暂停训练"""
    global training_state

    if not training_state.is_running:
        return False, "没有正在运行的训练"

    if training_state.is_paused:
        return False, "训练已经暂停"

    # 注意：YOLO训练暂停比较复杂，这里实现基本的进程暂停
    training_state.is_paused = True
    training_state.log_lines.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] ⏸️ 训练已暂停"
    )
    return True, "训练已暂停"


def resume_training() -> tuple[bool, str]:
    """恢复训练"""
    global training_state

    if not training_state.is_running:
        return False, "没有正在运行的训练"

    if not training_state.is_paused:
        return False, "训练没有暂停"

    training_state.is_paused = False
    training_state.log_lines.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] ▶️ 训练已恢复"
    )
    return True, "训练已恢复"


def stop_training() -> tuple[bool, str]:
    """停止训练"""
    global training_state

    if not training_state.is_running:
        return False, "没有正在运行的训练"

    try:
        training_state.is_running = False
        if training_state.process and training_state.process.poll() is None:
            training_state.process.terminate()
            training_state.process.wait(timeout=10)

        training_state.log_lines.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] ⏹️ 训练已停止"
        )
        return True, "训练已停止"

    except Exception as e:
        return False, f"停止训练失败: {e}"


def get_training_status() -> dict:
    """获取训练状态"""
    global training_state

    return {
        "is_running": training_state.is_running,
        "is_paused": training_state.is_paused,
        "current_epoch": training_state.current_epoch,
        "total_epochs": training_state.total_epochs,
        "progress": training_state.current_epoch
        / max(training_state.total_epochs, 1)
        * 100,
        "run_id": training_state.run_id,
        "log_lines": training_state.log_lines.copy(),
        "log_count": len(training_state.log_lines),
    }


def get_training_logs() -> List[str]:
    """获取训练日志"""
    global training_state
    return training_state.log_lines.copy()


def clear_training_logs():
    """清空训练日志"""
    global training_state
    training_state.log_lines.clear()


def get_device_info() -> str:
    """获取可用设备信息"""
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            return f"CUDA 可用 - {device_count} 个GPU - {device_name}"
        else:
            return "仅 CPU 可用 (推荐使用 GPU 进行训练)"
    except ImportError:
        return "PyTorch 未安装"
    except Exception as e:
        return f"设备检查失败: {e}"


def validate_training_environment() -> tuple[bool, str]:
    """验证训练环境"""
    issues = []

    # 检查 ultralytics
    try:
        import ultralytics

        issues.append("✅ Ultralytics 已安装")
    except ImportError:
        issues.append("❌ Ultralytics 未安装")
        return False, "\n".join(issues)

    # 检查 PyTorch
    try:
        import torch

        issues.append("✅ PyTorch 已安装")
        if torch.cuda.is_available():
            issues.append(f"✅ CUDA 可用 - {torch.cuda.device_count()} 个GPU")
        else:
            issues.append("⚠️ CUDA 不可用，将使用 CPU")
    except ImportError:
        issues.append("❌ PyTorch 未安装")
        return False, "\n".join(issues)

    # 检查目录
    if DATASETS_DIR.exists():
        issues.append("✅ 数据集目录存在")
    else:
        issues.append("❌ 数据集目录不存在")

    if MODELS_TRAINED_DIR.exists():
        issues.append("✅ 训练模型目录存在")
    else:
        issues.append("⚠️ 训练模型目录不存在，将自动创建")

    return True, "\n".join(issues)
