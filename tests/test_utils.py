"""测试工具函数"""

from __future__ import annotations

from pathlib import Path

import pytest

from yoloradio.utils import (
    detect_structure,
    extract_pathlike,
    is_supported_archive,
    list_dir,
)


class TestExtractPathlike:
    """测试路径提取函数"""

    def test_string_path(self):
        """测试字符串路径"""
        result = extract_pathlike("/path/to/file")
        assert result == Path("/path/to/file")

    def test_pathlib_path(self):
        """测试 Path 对象"""
        path = Path("/path/to/file")
        result = extract_pathlike(path)
        assert result == path

    def test_none_input(self):
        """测试 None 输入"""
        result = extract_pathlike(None)
        assert result is None

    def test_dict_with_name(self):
        """测试包含 name 键的字典"""
        result = extract_pathlike({"name": "/path/to/file"})
        assert result == Path("/path/to/file")

    def test_invalid_input(self):
        """测试无效输入"""
        result = extract_pathlike(123)
        assert result is None


class TestListDir:
    """测试目录列表函数"""

    def test_nonexistent_directory(self):
        """测试不存在的目录"""
        result = list_dir(Path("/nonexistent/path"))
        assert len(result) == 1
        assert "目录不存在" in result[0]

    def test_empty_directory(self, temp_dir: Path):
        """测试空目录"""
        result = list_dir(temp_dir)
        assert result == ["（空）"]

    def test_directory_with_files(self, sample_dataset_structure: Path):
        """测试包含文件的目录"""
        result = list_dir(sample_dataset_structure)
        assert len(result) > 0
        assert any("📁" in item for item in result)  # 至少有一个目录
        assert any("📄" in item for item in result)  # 至少有一个文件


class TestDetectStructure:
    """测试数据集结构检测"""

    def test_yolo_structure(self, sample_dataset_structure: Path):
        """测试 YOLO 结构检测"""
        structure, splits = detect_structure(sample_dataset_structure)
        assert structure == "yolo"
        assert "train" in splits
        assert "val" in splits

        train_img, train_lbl = splits["train"]
        assert train_img.name == "train"
        assert train_lbl.name == "train"


class TestArchiveSupport:
    """测试压缩包支持"""

    def test_supported_extensions(self):
        """测试支持的压缩格式"""
        assert is_supported_archive(Path("test.zip"))
        assert is_supported_archive(Path("test.tar.gz"))
        assert is_supported_archive(Path("test.tar"))
        assert not is_supported_archive(Path("test.txt"))
        assert not is_supported_archive(Path("test.jpg"))
