"""文件操作工具模块"""

from __future__ import annotations

# 配置日志
import logging
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional

from .paths import DATASETS_DIR

logger = logging.getLogger(__name__)


def extract_pathlike(obj: Any) -> Optional[Path]:
    """从各种对象中提取路径对象"""
    try:
        if obj is None:
            return None
        if isinstance(obj, str):
            return Path(obj)
        if isinstance(obj, bytes):
            try:
                return Path(obj.decode("utf-8"))
            except UnicodeDecodeError:
                return None
        if isinstance(obj, os.PathLike):
            fs = os.fspath(obj)
            if isinstance(fs, bytes):
                try:
                    return Path(fs.decode("utf-8"))
                except UnicodeDecodeError:
                    return None
            return Path(fs)

        # 尝试从对象属性获取路径
        path_attr = getattr(obj, "name", None) or getattr(obj, "path", None)
        if isinstance(path_attr, str):
            return Path(path_attr)

        # 尝试从字典获取路径
        if isinstance(obj, dict):
            p = obj.get("name") or obj.get("path")
            if isinstance(p, str):
                return Path(p)

    except Exception:
        return None
    return None


def is_supported_archive(p: Path) -> bool:
    """检查是否为支持的压缩包格式"""
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
    """移除文件名中的压缩包后缀"""
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


def unwrap_single_root(dest_dir: Path, max_depth: int = 3) -> None:
    """如果目录只包含一个子目录且无文件，将其内容上移"""

    def _only_one_dir_no_files(d: Path) -> Optional[Path]:
        if not d.exists() or not d.is_dir():
            return None
        items = list(d.iterdir())
        dirs = [x for x in items if x.is_dir()]
        files = [x for x in items if x.is_file()]
        if len(dirs) == 1 and len(files) == 0:
            return dirs[0]
        return None

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


def list_dir(
    dir_path: Path, exts: tuple[str, ...] | None = None, max_items: int = 100
) -> List[str]:
    """列出目录内容"""
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


def safe_move(src: Path, dst_dir: Path) -> str:
    """安全移动文件，避免覆盖"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    counter = 1
    while dst.exists():
        name = src.stem + f"_({counter})" + src.suffix
        dst = dst_dir / name
        counter += 1
    shutil.move(str(src), str(dst))
    return dst.name


def ensure_unique_dir(base: Path) -> Path:
    """确保目录名唯一"""
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = Path(f"{base}_({counter})")
        if not candidate.exists():
            return candidate
        counter += 1


def safe_extract_zip(zp: zipfile.ZipFile, dest: Path):
    """安全解压ZIP文件"""
    for info in zp.infolist():
        if info.filename.startswith("/") or ".." in info.filename:
            continue
        zp.extract(info, dest)


def safe_extract_tar(tf: tarfile.TarFile, dest: Path):
    """安全解压TAR文件"""

    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory

    for member in tf.getmembers():
        member_path = dest / member.name
        if not is_within_directory(dest, member_path):
            continue
        tf.extract(member, dest)


__all__ = [
    "list_dir",
    "extract_pathlike",
    "safe_move",
    "is_supported_archive",
    "strip_archive_suffix",
    "ensure_unique_dir",
    "safe_extract_zip",
    "safe_extract_tar",
]
