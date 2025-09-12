"""数据集管理模块"""

from __future__ import annotations

# 配置日志
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .file_utils import ensure_unique_dir
from .paths import DATASETS_DIR

logger = logging.getLogger(__name__)

# 常量定义
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Dataset:
    """数据集原型类 - 封装数据集的所有属性和行为"""

    def __init__(self, name: str, path: Optional[Path] = None):
        """初始化数据集实例

        Args:
            name: 数据集名称
            path: 数据集路径，如果为None则从name推导
        """
        self.name = name
        self.path = path or (DATASETS_DIR / name)
        self._metadata: Optional[dict] = None
        self._metadata_path = DATASETS_DIR / f"{name}.yml"

    @property
    def metadata_path(self) -> Path:
        """获取元数据文件路径"""
        return self._metadata_path

    @property
    def exists(self) -> bool:
        """检查数据集是否存在"""
        return self.path.exists() and self.path.is_dir()

    @property
    def metadata_exists(self) -> bool:
        """检查元数据文件是否存在"""
        return self._metadata_path.exists()

    @property
    def metadata(self) -> dict:
        """获取数据集元数据"""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata or {}

    def _load_metadata(self) -> None:
        """加载元数据"""
        if not self._metadata_path.exists():
            self._metadata = {}
            return

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                self._metadata = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"读取数据集元数据失败 {self._metadata_path}: {e}")
            self._metadata = {}

    def save_metadata(self, metadata: dict) -> bool:
        """保存元数据"""
        try:
            # 清理元数据中的Path对象
            cleaned_metadata = self._clean_metadata_for_yaml(metadata)
            with open(self._metadata_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    cleaned_metadata, f, allow_unicode=True, default_flow_style=False
                )
            self._metadata = cleaned_metadata.copy()
            return True
        except Exception as e:
            logger.error(f"保存数据集元数据失败 {self._metadata_path}: {e}")
            return False

    def _clean_metadata_for_yaml(self, metadata: dict) -> dict:
        """清理元数据中的Path对象，使其可以被YAML序列化"""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, Path):
                cleaned[key] = str(value)
            elif isinstance(value, dict):
                # 递归处理字典
                cleaned_dict = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, Path):
                        cleaned_dict[sub_key] = str(sub_value)
                    elif isinstance(sub_value, (list, tuple)):
                        cleaned_dict[sub_key] = [
                            str(item) if isinstance(item, Path) else item
                            for item in sub_value
                        ]
                    else:
                        cleaned_dict[sub_key] = sub_value
                cleaned[key] = cleaned_dict
            elif isinstance(value, (list, tuple)):
                # 处理列表中的Path对象
                cleaned[key] = [
                    str(item) if isinstance(item, Path) else item for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    @property
    def description(self) -> str:
        """获取数据集描述"""
        return self.metadata.get("description", "")

    @description.setter
    def description(self, value: str) -> None:
        """设置数据集描述"""
        meta = self.metadata.copy()
        meta["description"] = value
        self.save_metadata(meta)

    @property
    def dataset_type(self) -> str:
        """获取数据集类型"""
        return self.metadata.get("type", "detect")

    @property
    def dataset_type_display(self) -> str:
        """获取数据集类型显示名称"""
        return self.metadata.get("type_display", "目标检测")

    @property
    def structure(self) -> str:
        """获取数据集结构"""
        return self.metadata.get("structure", "")

    @property
    def splits(self) -> dict:
        """获取数据集分割信息"""
        return self.metadata.get("splits", {})

    @property
    def created_time(self) -> Optional[datetime]:
        """获取创建时间"""
        if not self.exists:
            return None
        try:
            return datetime.fromtimestamp(self.path.stat().st_ctime)
        except Exception:
            return None

    @property
    def modified_time(self) -> Optional[datetime]:
        """获取修改时间"""
        if not self.metadata_exists:
            return None
        try:
            return datetime.fromtimestamp(self._metadata_path.stat().st_mtime)
        except Exception:
            return None

    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        stats = {
            "total_images": 0,
            "total_labels": 0,
            "train_images": 0,
            "train_labels": 0,
            "val_images": 0,
            "val_labels": 0,
            "test_images": 0,
            "test_labels": 0,
        }

        if not self.exists:
            return stats

        # 统计各分割的图像和标注数量
        for split_name, split_info in self.splits.items():
            img_count = 0
            lbl_count = 0

            if isinstance(split_info, dict):
                img_spec = split_info.get("images")
                lbl_spec = split_info.get("labels")

                # images 可能是数字（直接记录）或目录名（需要统计）
                if isinstance(img_spec, int):
                    img_count = img_spec
                elif isinstance(img_spec, str):
                    img_dir = self._resolve_dir(img_spec)
                    img_count = _count_files(img_dir, IMG_EXTS)
                else:
                    img_count = self._count_images_in_split(split_name)

                # labels 同理：数字或目录名
                if isinstance(lbl_spec, int):
                    lbl_count = lbl_spec
                elif isinstance(lbl_spec, str):
                    lbl_dir = self._resolve_dir(lbl_spec)
                    lbl_count = _count_files(lbl_dir, {".txt", ".json", ".xml"})
                else:
                    lbl_count = self._count_labels_in_split(split_name)
            elif isinstance(split_info, (list, tuple)) and len(split_info) == 2:
                # 兼容旧格式：列表/元组 [images_dir, labels_dir]
                img_dir_spec, lbl_dir_spec = split_info
                img_dir = self._resolve_dir(img_dir_spec)
                lbl_dir = self._resolve_dir(lbl_dir_spec)
                img_count = _count_files(img_dir, IMG_EXTS)
                lbl_count = _count_files(lbl_dir, {".txt", ".json", ".xml"})
            else:
                # 兼容旧格式：按默认结构统计
                img_count = self._count_images_in_split(split_name)
                lbl_count = self._count_labels_in_split(split_name)

            stats[f"{split_name}_images"] = img_count
            stats[f"{split_name}_labels"] = lbl_count
            stats["total_images"] += img_count
            stats["total_labels"] += lbl_count

        return stats

    def _resolve_dir(self, spec: str | Path) -> Path:
        """根据元数据中的相对/绝对目录描述解析为目录路径"""
        p = Path(spec)
        if not p.is_absolute():
            p = self.path / p
        return p

    def _count_images_in_split(self, split_name: str) -> int:
        """统计特定分割中的图像数量（兼容旧格式与基于 splits 的目录映射）"""
        info = self.splits.get(split_name)
        if isinstance(info, dict) and isinstance(info.get("images"), str):
            split_dir = self._resolve_dir(info["images"])
        else:
            split_dir = self.path / split_name / "images"
            if not split_dir.exists():
                split_dir = self.path / split_name
        return _count_files(split_dir, IMG_EXTS)

    def _count_labels_in_split(self, split_name: str) -> int:
        """统计特定分割中的标注数量（兼容旧格式与基于 splits 的目录映射）"""
        info = self.splits.get(split_name)
        if isinstance(info, dict) and isinstance(info.get("labels"), str):
            split_dir = self._resolve_dir(info["labels"])
        else:
            split_dir = self.path / split_name / "labels"
            if not split_dir.exists():
                split_dir = self.path / split_name
        return _count_files(split_dir, {".txt", ".json", ".xml"})

    def rename(self, new_name: str) -> tuple[bool, str]:
        """重命名数据集"""
        if not self.exists:
            return False, "数据集目录不存在"

        if new_name == self.name:
            return True, "名称未改变"

        new_path = DATASETS_DIR / new_name
        new_meta_path = DATASETS_DIR / f"{new_name}.yml"

        # 检查目标是否存在
        if new_path.exists():
            return False, f"目标目录已存在: {new_name}"

        try:
            # 重命名目录
            self.path.rename(new_path)

            # 重命名元数据文件
            if self._metadata_path.exists():
                self._metadata_path.rename(new_meta_path)

            # 更新内部状态
            old_name = self.name
            self.name = new_name
            self.path = new_path
            self._metadata_path = new_meta_path

            # 更新元数据中的名称
            if self._metadata:
                meta = self._metadata.copy()
                meta["name"] = new_name
                self.save_metadata(meta)

            return True, f"已重命名: {old_name} -> {new_name}"
        except Exception as e:
            logger.error(f"重命名数据集失败: {e}")
            return False, f"重命名失败: {e}"

    def delete(self) -> tuple[bool, str]:
        """删除数据集"""
        if not self.exists:
            return False, "数据集目录不存在"

        try:
            # 删除目录
            shutil.rmtree(self.path)

            # 删除元数据文件
            if self._metadata_path.exists():
                self._metadata_path.unlink()

            return True, f"已删除数据集: {self.name}"
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            return False, f"删除失败: {e}"

    def to_dict(self) -> dict:
        """转换为字典格式（用于API返回）"""
        stats = self.get_statistics()

        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.dataset_type,
            "type_display": self.dataset_type_display,
            "description": self.description,
            "structure": self.structure,
            "meta_exists": self.metadata_exists,
            "meta_modified": (
                self.modified_time.strftime("%Y-%m-%d %H:%M:%S")
                if self.modified_time
                else None
            ),
            "created_time": self.created_time,
            **stats,
        }


class DatasetManager:
    """数据集管理器 - 提供统一的数据集操作接口"""

    def __init__(self):
        """初始化数据集管理器"""
        self._datasets_cache: Dict[str, Dataset] = {}
        self._ensure_datasets_dir()

    def _ensure_datasets_dir(self) -> None:
        """确保数据集目录存在"""
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[Dataset]:
        """获取所有数据集列表"""
        self._refresh_cache()
        return list(self._datasets_cache.values())

    def list_datasets_with_type(self, type: str) -> List[Dataset]:
        """获取所有数据集列表"""
        self._refresh_cache()
        if type:
            return [
                ds for ds in self._datasets_cache.values() if ds.dataset_type == type
            ]
        return list(self._datasets_cache.values())

    def _refresh_cache(self) -> None:
        """刷新数据集缓存"""
        if not DATASETS_DIR.exists():
            self._datasets_cache.clear()
            return

        # 获取现有目录
        existing_dirs = {p.name for p in DATASETS_DIR.iterdir() if p.is_dir()}

        # 移除不存在的数据集
        to_remove = [
            name for name in self._datasets_cache.keys() if name not in existing_dirs
        ]
        for name in to_remove:
            del self._datasets_cache[name]

        # 添加新的数据集
        for dir_name in existing_dirs:
            if dir_name not in self._datasets_cache:
                self._datasets_cache[dir_name] = Dataset(dir_name)

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """获取特定数据集"""
        self._refresh_cache()
        return self._datasets_cache.get(name)

    def get_dataset_names(self) -> List[str]:
        """获取所有数据集名称"""
        return [ds.name for ds in self.list_datasets()]

    def get_datasets_summary(self) -> tuple[List[str], List[List[str]]]:
        """获取数据集汇总表格数据"""
        datasets = self.list_datasets()

        headers = ["名称", "类型", "结构", "图像", "标注", "描述"]
        rows = []

        for ds in datasets:
            stats = ds.get_statistics()
            desc = ds.description

            # 处理长描述
            if len(desc) > 30:
                desc = desc[:27] + "..."
            elif not desc:
                desc = "*无描述*"

            row = [
                ds.name,
                ds.dataset_type_display,
                ds.structure or "*未知*",
                str(stats["total_images"]),
                str(stats["total_labels"]),
                desc,
            ]
            rows.append(row)

        return headers, rows

    def get_dataset_detail(self, name: str) -> Optional[dict]:
        """获取数据集详细信息"""
        dataset = self.get_dataset(name)
        if not dataset:
            return None
        return dataset.to_dict()

    def create_dataset_from_upload(
        self,
        name: str,
        description: str,
        dataset_type: str,
        archive_path: Path,
        type_map: Dict[str, str],
    ) -> tuple[bool, str]:
        """从上传的压缩包创建数据集"""
        try:
            # 导入必要的函数
            from .file_utils import (
                is_supported_archive,
                safe_extract_tar,
                safe_extract_zip,
                strip_archive_suffix,
                unwrap_single_root,
            )

            # 基本校验
            if not name:
                return False, "请填写数据集名称"

            if not archive_path or not archive_path.exists():
                return False, "请上传一个压缩包文件"

            if not is_supported_archive(archive_path):
                return False, f"不支持的压缩包类型: {archive_path.name}"

            # 目标目录
            folder_name = strip_archive_suffix(name.strip())
            dest_dir = ensure_unique_dir(DATASETS_DIR / folder_name)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # 解压
            try:
                if archive_path.suffix.lower() == ".zip":
                    import zipfile

                    with zipfile.ZipFile(archive_path, "r") as zf:
                        safe_extract_zip(zf, dest_dir)
                else:
                    import tarfile

                    mode = self._get_tar_mode(archive_path)
                    # 直接使用tarfile.open，忽略类型检查
                    tf = tarfile.open(str(archive_path), mode or "r:")  # type: ignore
                    try:
                        safe_extract_tar(tf, dest_dir)
                    finally:
                        tf.close()
            except Exception as e:
                return False, f"解压失败: {archive_path.name} -> {e}"

            # 展平单根目录
            unwrap_single_root(dest_dir)

            # 结构与类型校验
            type_code = type_map.get(dataset_type, "detect")
            ok, structure, msg, splits = validate_dataset_by_type(dest_dir, type_code)
            if not ok:
                # 清理失败的目录
                try:
                    if not any(dest_dir.rglob("*")):
                        dest_dir.rmdir()
                except Exception:
                    pass
                return False, f"数据集校验失败：{msg}"

            # 创建数据集实例并保存元数据
            dataset = Dataset(folder_name, dest_dir)

            # 清理splits，统一为 screw_long_obb.yml 风格且使用相对路径
            cleaned_splits = self._clean_splits_for_yaml(splits, base_dir=dest_dir)

            metadata = {
                "name": folder_name,
                "type": type_code,
                "type_display": dataset_type,
                "description": description or "",
                "structure": structure,
                "splits": cleaned_splits,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }

            if not dataset.save_metadata(metadata):
                return False, "已导入但写元数据失败"

            # 添加到缓存
            self._datasets_cache[folder_name] = dataset

            return (
                True,
                f"已导入数据集：{folder_name}（类型：{dataset_type}，结构：{structure}）",
            )

        except Exception as e:
            logger.error(f"创建数据集失败: {e}")
            return False, f"创建数据集失败: {e}"

    def _get_tar_mode(self, archive_path: Path) -> Optional[str]:
        """获取tar文件的打开模式"""
        name_lower = archive_path.name.lower()
        if "gz" in "".join(archive_path.suffixes).lower() or name_lower.endswith(
            ".tgz"
        ):
            return "r:gz"
        elif name_lower.endswith((".bz2", ".tbz")):
            return "r:bz2"
        elif name_lower.endswith((".xz", ".txz")):
            return "r:xz"
        return None

    def _clean_splits_for_yaml(
        self, splits: dict, base_dir: Optional[Path] = None
    ) -> dict:
        """将内部 splits 映射清理为 screw_long_obb.yml 风格：
        { split: { images: <relative_dir>, labels: <relative_dir> } }

        - 接受 tuple/list/dict 三种输入形式
        - 输出一律为相对 base_dir 的相对路径（若可相对）
        """
        if not splits:
            return {}

        def rel_str(p: Path | str) -> str:
            try:
                path = Path(p)
            except Exception:
                return str(p)
            # 若提供了 base_dir，尽量转为相对
            if base_dir is not None:
                try:
                    # Windows 兼容：as_posix 统一分隔符
                    return path.relative_to(base_dir).as_posix()
                except Exception:
                    return path.as_posix()
            return path.as_posix()

        cleaned: dict = {}
        for split_name, split_data in splits.items():
            img_dir_s: str
            lbl_dir_s: str
            if isinstance(split_data, dict):
                img_v = split_data.get("images")
                lbl_v = split_data.get("labels")
                if img_v is None or lbl_v is None:
                    # 兜底：尝试从任意值推断（不严格）
                    items = list(split_data.values())
                    if len(items) >= 2:
                        img_dir_s = rel_str(items[0])
                        lbl_dir_s = rel_str(items[1])
                    else:
                        # 回退默认结构
                        img_dir_s = f"{split_name}/images"
                        lbl_dir_s = f"{split_name}/labels"
                else:
                    img_dir_s = rel_str(img_v)
                    lbl_dir_s = rel_str(lbl_v)
            elif isinstance(split_data, (list, tuple)) and len(split_data) == 2:
                img_dir_s = rel_str(split_data[0])
                lbl_dir_s = rel_str(split_data[1])
            else:
                # 无法识别，按默认结构
                img_dir_s = f"{split_name}/images"
                lbl_dir_s = f"{split_name}/labels"

            cleaned[split_name] = {"images": img_dir_s, "labels": lbl_dir_s}

        return cleaned

    def update_dataset_description(
        self, name: str, description: str
    ) -> tuple[bool, str]:
        """更新数据集描述"""
        dataset = self.get_dataset(name)
        if not dataset:
            return False, f"数据集不存在: {name}"

        try:
            dataset.description = description
            return True, "描述已更新"
        except Exception as e:
            logger.error(f"更新描述失败: {e}")
            return False, f"更新描述失败: {e}"

    def rename_dataset(self, old_name: str, new_name: str) -> tuple[bool, str, str]:
        """重命名数据集"""
        dataset = self.get_dataset(old_name)
        if not dataset:
            return False, old_name, f"数据集不存在: {old_name}"

        ok, msg = dataset.rename(new_name)
        if ok:
            # 更新缓存
            del self._datasets_cache[old_name]
            self._datasets_cache[new_name] = dataset
            return True, new_name, msg
        else:
            return False, old_name, msg

    def delete_dataset(self, name: str) -> tuple[bool, str]:
        """删除数据集"""
        dataset = self.get_dataset(name)
        if not dataset:
            return False, f"数据集不存在: {name}"

        ok, msg = dataset.delete()
        if ok:
            # 从缓存中移除
            if name in self._datasets_cache:
                del self._datasets_cache[name]

        return ok, msg


# 全局数据集管理器实例
dataset_manager = DatasetManager()


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


__all__ = [
    "IMG_EXTS",
    "dataset_manager",
    "Dataset",
    "DatasetManager",
]
