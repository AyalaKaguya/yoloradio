"""数据集管理模块"""

from __future__ import annotations

# 配置日志
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .file_utils import ensure_unique_dir
from .paths import DATASETS_DIR

logger = logging.getLogger(__name__)


def read_class_names(ds_dir: Path) -> list[str]:
    """读取数据集的类别名称"""
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


# 常量定义
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASET_TYPE_MAP: Dict[str, str] = {
    "图像分类": "classify",
    "目标检测": "detect",
    "图像分割": "segment",
    "关键点跟踪": "pose",
    "旋转检测框识别": "obb",
}


def _count_files(dir_path: Path, exts: set[str]) -> int:
    """统计指定扩展名的文件数量"""
    if not dir_path.exists():
        return 0
    return sum(
        1 for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def _count_label_txt(dir_path: Path) -> int:
    """统计标注文件数量"""
    if not dir_path.exists():
        return 0
    return sum(1 for p in dir_path.rglob("*.txt") if p.is_file())


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


def dataset_type_code(ds_name: str) -> Optional[str]:
    """获取数据集的任务类型代码"""
    p = meta_path_for(ds_name)
    return _read_yaml_top_level_value(p, "type")


def list_datasets_for_task(task_code: str) -> List[str]:
    """列出指定任务类型的数据集"""
    if not DATASETS_DIR.exists():
        return []
    names: List[str] = []
    for d in sorted(DATASETS_DIR.iterdir()):
        if d.is_dir():
            t = dataset_type_code(d.name)
            if t == task_code:
                names.append(d.name)
    return names


def summarize_dataset(ds_dir: Path) -> dict:
    """数据集摘要统计"""
    name = ds_dir.name
    # 支持两种结构：分割结构和扁平结构
    has_split = any((ds_dir / s).exists() for s in ("train", "val", "test"))
    if has_split:
        splits = {
            "train": (ds_dir / "train" / "images", ds_dir / "train" / "labels"),
            "val": (ds_dir / "val" / "images", ds_dir / "val" / "labels"),
            "test": (ds_dir / "test" / "images", ds_dir / "test" / "labels"),
        }
    else:
        splits = {
            "train": (ds_dir / "images", ds_dir / "labels"),
            "val": (ds_dir / "val_images", ds_dir / "val_labels"),
            "test": (ds_dir / "test_images", ds_dir / "test_labels"),
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
    """获取数据集目录列表"""
    if not DATASETS_DIR.exists():
        return []
    return [p for p in sorted(DATASETS_DIR.iterdir()) if p.is_dir()]


def dataset_summary_table() -> tuple[list[str], list[list]]:
    """生成数据集摘要表格"""
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


def detect_structure(base: Path) -> Tuple[str, Dict[str, Tuple[Path, Path]]]:
    """检测数据集结构"""
    # 分割结构
    split_dirs = {
        "train": (base / "train" / "images", base / "train" / "labels"),
        "val": (base / "val" / "images", base / "val" / "labels"),
        "test": (base / "test" / "images", base / "test" / "labels"),
    }
    if any(img.exists() or lbl.exists() for img, lbl in split_dirs.values()):
        return "split", split_dirs

    # 扁平结构
    flat_dirs = {
        "train": (base / "images", base / "labels"),
        "val": (base / "val_images", base / "val_labels"),
        "test": (base / "test_images", base / "test_labels"),
    }
    if any(img.exists() or lbl.exists() for img, lbl in flat_dirs.values()):
        return "flat", flat_dirs

    return "unknown", {}


def _has_any_images(p: Path) -> bool:
    """检查是否包含图像文件"""
    return _count_files(p, IMG_EXTS) > 0


def _has_any_labels(p: Path) -> bool:
    """检查是否包含标注文件"""
    return _count_label_txt(p) > 0


def validate_dataset_by_type(
    base: Path, ds_type_code: str
) -> Tuple[bool, str, str, Dict[str, Tuple[Path, Path]]]:
    """根据类型验证数据集"""
    structure, splits = detect_structure(base)
    if structure == "unknown":
        return (
            False,
            structure,
            "无法识别数据集结构（未发现 images/ 或 labels/ 目录）",
            {},
        )

    def split_ok(img_dir: Path, lbl_dir: Path) -> bool:
        if ds_type_code == "classify":
            return _has_any_images(img_dir)
        else:
            return _has_any_images(img_dir) and _has_any_labels(lbl_dir)

    good = any(split_ok(img_dir, lbl_dir) for img_dir, lbl_dir in splits.values())
    if not good:
        msg = (
            "未检测到任何图片文件"
            if ds_type_code == "classify"
            else "未检测到匹配的图片与标注文件"
        )
        return False, structure, msg, splits

    return True, structure, "检测到有效的数据集结构", splits


def build_metadata_yaml(
    name: str,
    type_code: str,
    type_display: str,
    description: str,
    structure: str,
    splits: Dict[str, Tuple[Path, Path]],
) -> str:
    """构建元数据YAML"""
    created_at = datetime.utcnow().isoformat() + "Z"

    def rel(p: Path) -> str:
        return str(p).replace("\\", "/")

    lines = [
        f"name: {name}",
        f"type: {type_code}",
        f"type_display: {type_display}",
        "description: |",
    ]
    for ln in description.splitlines() or [""]:
        lines.append(f"  {ln}")

    lines.extend([f"created_at: {created_at}", f"structure: {structure}", "splits:"])

    for s, (img_dir, lbl_dir) in splits.items():
        lines.extend(
            [
                f"  {s}:",
                f"    images: {rel(Path(img_dir.name) if img_dir.is_absolute() else img_dir)}",
                f"    labels: {rel(Path(lbl_dir.name) if lbl_dir.is_absolute() else lbl_dir)}",
            ]
        )

    return "\n".join(lines) + "\n"


def dataset_detail_table(ds_name: str) -> tuple[list[str], list[list]]:
    """生成数据集详情表格"""
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


# ---------- 数据集元数据管理 ----------


def meta_path_for(name: str) -> Path:
    """获取数据集元数据文件路径"""
    return DATASETS_DIR / f"{name}.yml"


def read_meta_description(name: str) -> str:
    """读取数据集描述"""
    p = meta_path_for(name)
    if not p.exists():
        return ""

    try:
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()

        # 查找description行
        desc_start = None
        for idx, ln in enumerate(lines):
            if ln.strip().startswith("description:"):
                desc_start = idx
                break

        if desc_start is None:
            return ""

        # 处理多行描述
        if "|" in lines[desc_start]:
            desc = []
            for j in range(desc_start + 1, len(lines)):
                ln = lines[j]
                if ln.startswith("  "):
                    desc.append(ln[2:])
                else:
                    break
            return "\n".join(desc).rstrip("\n")

        # 单行描述
        val = lines[desc_start].split(":", 1)[1].strip()
        return val

    except Exception:
        return ""


def update_meta_description(name: str, new_desc: str) -> tuple[bool, str]:
    """更新数据集描述"""
    p = meta_path_for(name)
    if not p.exists():
        return False, "未找到元数据文件"

    try:
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()

        # 查找description块
        desc_start = None
        for idx, ln in enumerate(lines):
            if ln.strip().startswith("description:"):
                desc_start = idx
                break

        block = ["description: |"]
        for ln in new_desc.splitlines() or [""]:
            block.append(f"  {ln}")

        if desc_start is None:
            # 插入新描述块
            insert_idx = 0
            for idx, ln in enumerate(lines):
                if ln.strip().startswith("type_display:"):
                    insert_idx = idx + 1
                    break
                if ln.strip().startswith("name:"):
                    insert_idx = idx + 1
            new_lines = lines[:insert_idx] + block + lines[insert_idx:]
        else:
            # 替换现有描述块
            desc_end = desc_start + 1
            while desc_end < len(lines) and lines[desc_end].startswith("  "):
                desc_end += 1
            new_lines = lines[:desc_start] + block + lines[desc_end:]

        p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True, "已更新描述"

    except Exception as e:
        return False, f"更新失败：{e}"


def _replace_yaml_name(text: str, new_name: str) -> str:
    """替换YAML中的name字段"""
    lines = text.splitlines()
    for idx, ln in enumerate(lines):
        if ln.startswith("name:"):
            lines[idx] = f"name: {new_name}"
            break
    return "\n".join(lines) + "\n"


def rename_dataset(old_name: str, new_name: str) -> tuple[bool, str, str]:
    """重命名数据集"""
    src_dir = DATASETS_DIR / old_name
    if not src_dir.exists() or not src_dir.is_dir():
        return False, old_name, "数据集目录不存在"

    desired = DATASETS_DIR / new_name
    if desired.exists():
        desired = ensure_unique_dir(desired)
    final_name = desired.name

    try:
        shutil.move(str(src_dir), str(desired))

        # 重命名元数据文件
        old_meta = meta_path_for(old_name)
        new_meta = meta_path_for(final_name)
        if old_meta.exists():
            try:
                txt = old_meta.read_text(encoding="utf-8")
                txt2 = _replace_yaml_name(txt, final_name)
                shutil.move(str(old_meta), str(new_meta))
                new_meta.write_text(txt2, encoding="utf-8")
            except Exception as e:
                return False, old_name, f"重命名元数据失败：{e}"

        return True, final_name, "已重命名"

    except Exception as e:
        return False, old_name, f"重命名失败：{e}"


def delete_dataset(name: str) -> tuple[bool, str]:
    """删除数据集"""
    ds_dir = DATASETS_DIR / name
    meta = meta_path_for(name)

    if not ds_dir.exists():
        return False, "数据集目录不存在"

    try:
        shutil.rmtree(ds_dir, ignore_errors=False)
        if meta.exists():
            meta.unlink()
        return True, "已删除"
    except Exception as e:
        return False, f"删除失败：{e}"


# ---------- 数据集导入辅助函数 ----------


def _only_one_dir_no_files(base: Path) -> Optional[Path]:
    """检查目录是否只包含一个子目录且无文件"""
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    files = [p for p in base.iterdir() if p.is_file()]
    if len(dirs) == 1 and len(files) == 0:
        return dirs[0]
    return None


def unwrap_single_root(dest_dir: Path, max_depth: int = 3) -> None:
    """展开单一根目录结构"""
    depth = 0
    while depth < max_depth:
        inner = _only_one_dir_no_files(dest_dir)
        if inner is None:
            break
        # 移动内部目录的所有内容到外层
        for item in inner.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        try:
            inner.rmdir()
        except OSError:
            pass
        depth += 1


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


__all__ = [
    "DATASET_TYPE_MAP",
    "IMG_EXTS",
    "summarize_dataset",
    "list_dataset_dirs",
    "dataset_summary_table",
    "detect_structure",
    "validate_dataset_by_type",
    "build_metadata_yaml",
    "dataset_detail_table",
    "read_meta_description",
    "update_meta_description",
    "rename_dataset",
    "delete_dataset",
    "unwrap_single_root",
]
