"""测试配置和公共工具"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录用于测试"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset_structure(temp_dir: Path) -> Path:
    """创建示例数据集结构用于测试"""
    ds_dir = temp_dir / "test_dataset"

    # 创建标准 YOLO 结构
    (ds_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 创建示例文件
    (ds_dir / "images" / "train" / "image1.jpg").touch()
    (ds_dir / "images" / "train" / "image2.jpg").touch()
    (ds_dir / "labels" / "train" / "image1.txt").touch()
    (ds_dir / "labels" / "train" / "image2.txt").touch()

    (ds_dir / "images" / "val" / "image3.jpg").touch()
    (ds_dir / "labels" / "val" / "image3.txt").touch()

    # 创建类别文件
    (ds_dir / "classes.txt").write_text("person\ncar\nbike\n")

    return ds_dir
