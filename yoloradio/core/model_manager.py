"""模型管理模块"""

from __future__ import annotations

# 配置日志
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple

import yaml

from .paths import MODELS_PRETRAINED_DIR, MODELS_TRAINED_DIR
from .task_types import TASK_MAP as MODEL_TASK_MAP

logger = logging.getLogger(__name__)

# 常量定义
MODEL_EXTS = {".pt", ".onnx", ".engine", ".xml", ".bin"}


class Model:
    """模型原型类 - 封装模型的所有属性和行为"""

    def __init__(
        self, name: str, is_pretrained: bool = True, path: Optional[Path] = None
    ):
        """初始化模型实例

        Args:
            name: 模型名称（不含扩展名）
            is_pretrained: 是否为预训练模型
            path: 模型路径，如果为None则从name推导
        """
        self.name = name
        self.is_pretrained = is_pretrained
        self.base_dir = MODELS_PRETRAINED_DIR if is_pretrained else MODELS_TRAINED_DIR
        self.path = path or self._find_model_file()
        self._metadata: Optional[dict] = None
        self._metadata_path = self.base_dir / f"{name}.yml"

    def _find_model_file(self) -> Optional[Path]:
        """查找模型文件"""
        for ext in MODEL_EXTS:
            candidate = self.base_dir / f"{self.name}{ext}"
            if candidate.exists():
                return candidate
        return None

    @property
    def exists(self) -> bool:
        """检查模型文件是否存在"""
        return self.path is not None and self.path.exists()

    @property
    def metadata_exists(self) -> bool:
        """检查元数据文件是否存在"""
        return self._metadata_path.exists()

    @property
    def filename(self) -> str:
        """获取文件名"""
        return self.path.name if self.path else ""

    @property
    def stem(self) -> str:
        """获取文件主名（不含扩展名）"""
        return self.path.stem if self.path else self.name

    @property
    def metadata(self) -> dict:
        """获取模型元数据"""
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
            logger.warning(f"读取模型元数据失败 {self._metadata_path}: {e}")
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
            logger.error(f"保存模型元数据失败 {self._metadata_path}: {e}")
            return False

    def _clean_metadata_for_yaml(self, metadata: dict) -> dict:
        """清理元数据中的Path对象，使其可以被YAML序列化"""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, Path):
                cleaned[key] = str(value)
            elif isinstance(value, dict):
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
                cleaned[key] = [
                    str(item) if isinstance(item, Path) else item for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned

    @property
    def description(self) -> str:
        """获取模型描述"""
        return self.metadata.get("description", "")

    @description.setter
    def description(self, value: str) -> None:
        """设置模型描述"""
        meta = self.metadata.copy()
        meta["description"] = value
        self.save_metadata(meta)

    @property
    def task_code(self) -> str:
        """获取任务类型代码"""
        return self.metadata.get("task", "detect")

    @property
    def task_display(self) -> str:
        """获取任务类型显示名称"""
        return self.metadata.get("task_display", "目标检测")

    @property
    def version(self) -> str:
        """获取YOLO版本"""
        return self.metadata.get("version", "")

    @property
    def size(self) -> str:
        """获取模型大小"""
        return self.metadata.get("size", "")

    @property
    def file_size_mb(self) -> float:
        """获取文件大小(MB)"""
        if not self.exists or not self.path:
            return 0.0
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0

    @property
    def created_time(self) -> Optional[datetime]:
        """获取创建时间"""
        if not self.exists or not self.path:
            return None
        try:
            return datetime.fromtimestamp(self.path.stat().st_mtime)
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

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "filename": self.filename,
            "stem": self.stem,
            "is_pretrained": self.is_pretrained,
            "file_size_mb": self.file_size_mb,
            "created_time": self.created_time,
            "created_str": (
                self.created_time.strftime("%Y-%m-%d %H:%M")
                if self.created_time
                else "未知"
            ),
            "description": self.description,
            "full_description": self.metadata.get("description", ""),
            "task_code": self.task_code,
            "task_display": self.task_display,
            "version": self.version,
            "size": self.size,
            "metadata": self.metadata,
            "model_path": self.path,
            "yml_path": self._metadata_path,
        }

    def get_detail_info(self) -> str:
        """获取详细信息文本"""
        if not self.exists:
            return f"❌ 模型文件不存在: {self.name}"

        # 构建详细信息
        info_lines = [
            f"# 📦 模型详细信息",
            f"",
            f"**模型名称:** {self.name}",
            f"**文件路径:** `{self.path}`",
            f"**文件大小:** {self.file_size_mb:.2f} MB",
            f"**创建时间:** {self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else '未知'}",
            f"**修改时间:** {self.modified_time.strftime('%Y-%m-%d %H:%M:%S') if self.modified_time else '未知'}",
            f"**模型类型:** {'预训练模型' if self.is_pretrained else '训练模型'}",
            f"",
        ]

        metadata = self.metadata
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
                if isinstance(desc, str):
                    cleaned_desc = desc.strip()
                    for line in cleaned_desc.split("\n"):
                        line = line.strip()
                        if line:
                            info_lines.append(f"> {line}")
                else:
                    info_lines.append(f"> {desc}")

            # 训练相关信息（仅训练模型）
            if not self.is_pretrained:
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

        return "\n".join(info_lines)

    def delete(self) -> tuple[bool, str]:
        """删除模型文件"""
        if not self.exists or not self.path:
            return False, "文件不存在"

        try:
            self.path.unlink()
            # 删除对应的YAML文件
            if self._metadata_path.exists():
                try:
                    self._metadata_path.unlink()
                except Exception:
                    pass
            return True, "已删除"
        except Exception as e:
            return False, f"删除失败：{e}"

    def rename(self, new_name: str) -> tuple[bool, str, str]:
        """重命名模型文件"""
        if not self.exists or not self.path:
            return False, self.name, "文件不存在"

        old_path = self.path
        new_path = self.base_dir / f"{new_name}{old_path.suffix}"

        if new_path.exists():
            # 创建唯一名称
            i = 1
            while (self.base_dir / f"{new_name}_{i}{old_path.suffix}").exists():
                i += 1
            new_path = self.base_dir / f"{new_name}_{i}{old_path.suffix}"

        try:
            shutil.move(str(old_path), str(new_path))

            # 重命名对应的YAML文件
            old_yaml = self._metadata_path
            new_yaml = self.base_dir / f"{new_path.stem}.yml"
            if old_yaml.exists():
                try:
                    shutil.move(str(old_yaml), str(new_yaml))
                except Exception:
                    pass

            # 更新实例属性
            self.name = new_path.stem
            self.path = new_path
            self._metadata_path = new_yaml

            return True, new_path.name, "已重命名"
        except Exception as e:
            return False, self.name, f"重命名失败：{e}"


class ModelManager:
    """模型管理器 - 提供统一的模型操作接口"""

    def __init__(self):
        """初始化模型管理器"""
        self._models_cache: Dict[str, Model] = {}
        self._ensure_model_dirs()

    def _ensure_model_dirs(self) -> None:
        """确保模型目录存在"""
        MODELS_PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    def list_models(self, is_pretrained: Optional[bool] = None) -> List[Model]:
        """获取模型列表"""
        self._refresh_cache()
        if is_pretrained is None:
            return list(self._models_cache.values())
        return [
            model
            for model in self._models_cache.values()
            if model.is_pretrained == is_pretrained
        ]

    def list_models_with_task(self, task_code: str) -> List[Model]:
        """获取指定任务类型的模型列表"""
        self._refresh_cache()
        return [
            model
            for model in self._models_cache.values()
            if model.task_code == task_code
        ]

    def _refresh_cache(self) -> None:
        """刷新模型缓存"""
        # 清空缓存
        self._models_cache.clear()

        # 扫描预训练模型
        if MODELS_PRETRAINED_DIR.exists():
            self._scan_directory(MODELS_PRETRAINED_DIR, is_pretrained=True)

        # 扫描训练模型
        if MODELS_TRAINED_DIR.exists():
            self._scan_directory(MODELS_TRAINED_DIR, is_pretrained=False)

    def _scan_directory(self, directory: Path, is_pretrained: bool) -> None:
        """扫描目录中的模型文件"""
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in MODEL_EXTS:
                model_name = file_path.stem
                cache_key = f"{'pre' if is_pretrained else 'train'}_{model_name}"

                # 避免重复（如果有同名模型）
                if cache_key not in self._models_cache:
                    model = Model(model_name, is_pretrained, file_path)
                    self._models_cache[cache_key] = model

    def get_model(self, name: str, is_pretrained: bool = True) -> Optional[Model]:
        """获取特定模型"""
        self._refresh_cache()
        cache_key = f"{'pre' if is_pretrained else 'train'}_{name}"
        return self._models_cache.get(cache_key)

    def get_model_names(self, is_pretrained: Optional[bool] = None) -> List[str]:
        """获取模型名称列表"""
        models = self.list_models(is_pretrained)
        return [model.name for model in models]

    def get_models_summary(self) -> tuple[List[str], List[List[str]], List[List[str]]]:
        """获取模型汇总表格数据"""
        pretrained_models = self.list_models(is_pretrained=True)
        trained_models = self.list_models(is_pretrained=False)

        headers = ["文件名", "描述", "创建时间"]

        pretrained_rows = []
        for model in pretrained_models:
            desc = model.description
            if len(desc) > 50:
                desc = "|"
            elif not desc:
                desc = "无描述"

            row = [
                model.filename,
                desc,
                (
                    model.created_time.strftime("%Y-%m-%d %H:%M")
                    if model.created_time
                    else "未知"
                ),
            ]
            pretrained_rows.append(row)

        trained_rows = []
        for model in trained_models:
            desc = model.description
            if len(desc) > 50:
                desc = "|"
            elif not desc:
                desc = "无描述"

            row = [
                model.filename,
                desc,
                (
                    model.created_time.strftime("%Y-%m-%d %H:%M")
                    if model.created_time
                    else "未知"
                ),
            ]
            trained_rows.append(row)

        return headers, pretrained_rows, trained_rows

    def get_model_detail(self, name: str, is_pretrained: bool = True) -> str:
        """获取模型详细信息"""
        model = self.get_model(name, is_pretrained)
        if not model:
            return f"❌ 模型不存在: {name}"
        return model.get_detail_info()

    def get_model_choices_for_detail(self) -> List[str]:
        """获取用于详细信息下拉框的选项"""
        self._refresh_cache()
        choices = []

        # 添加预训练模型
        for model in self.list_models(is_pretrained=True):
            display_name = f"[预训练] {model.name}"
            choices.append(display_name)

        # 添加训练模型
        for model in self.list_models(is_pretrained=False):
            display_name = f"[训练] {model.name}"
            choices.append(display_name)

        return choices

    def download_pretrained_model(self, name_or_path: str) -> tuple[bool, str, str]:
        """下载预训练模型"""
        try:
            from ultralytics import YOLO

            m = YOLO(name_or_path)
            ckpt_path = getattr(m, "ckpt_path", None)
            if not ckpt_path:
                ckpt_path = name_or_path

            src = Path(ckpt_path)
            if not src.exists():
                return False, "未找到下载后的权重文件", ""

            dest = MODELS_PRETRAINED_DIR / src.name

            if not dest.exists():
                shutil.copy2(src, dest)

            # 刷新缓存
            self._refresh_cache()

            return True, f"已准备预训练模型: {dest.name}", dest.name

        except Exception as e:
            return False, f"下载失败：{e}", ""

    def download_and_register_pretrained(
        self,
        task_display: str,
        version: str,
        size: str,
        custom_name: str,
        description: str,
        progress_cb=None,
    ) -> tuple[bool, str]:
        """下载并注册预训练模型"""
        task_code = MODEL_TASK_MAP.get(task_display, "detect")
        weight = self._compute_ultra_weight_name(task_code, version, size)

        if progress_cb:
            try:
                progress_cb(0.1, f"准备下载 {weight}")
            except Exception:
                pass

        ok, msg, saved = self.download_pretrained_model(weight)
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
                desired = (
                    MODELS_PRETRAINED_DIR / f"{custom_name.strip()}_{i}{src.suffix}"
                )
            shutil.move(str(src), str(desired))
            final_file = desired

        # 创建模型实例并写入元数据
        model = Model(final_file.stem, is_pretrained=True, path=final_file)
        metadata = {
            "name": final_file.stem,
            "task": task_code,
            "task_display": task_display,
            "version": version,
            "size": size,
            "description": description or "",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        model.save_metadata(metadata)

        # 刷新缓存
        self._refresh_cache()

        if progress_cb:
            try:
                progress_cb(1.0, f"完成：{final_file.name}")
            except Exception:
                pass

        return True, f"已准备预训练模型：{final_file.name}"

    def _compute_ultra_weight_name(
        self, task_code: str, version: str, size: str
    ) -> str:
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

    def delete_model(self, name: str, is_pretrained: bool) -> tuple[bool, str]:
        """删除模型"""
        model = self.get_model(name, is_pretrained)
        if not model:
            return False, "模型不存在"

        ok, msg = model.delete()
        if ok:
            # 从缓存中移除
            cache_key = f"{'pre' if is_pretrained else 'train'}_{name}"
            self._models_cache.pop(cache_key, None)

        return ok, msg

    def rename_model(
        self, old_name: str, new_name: str, is_pretrained: bool
    ) -> tuple[bool, str, str]:
        """重命名模型"""
        model = self.get_model(old_name, is_pretrained)
        if not model:
            return False, old_name, "模型不存在"

        ok, final_name, msg = model.rename(new_name)
        if ok:
            # 更新缓存
            self._refresh_cache()

        return ok, final_name, msg

    def list_models_for_task_display(self, task_code: str) -> List[Tuple[str, str]]:
        """返回指定任务类型的模型列表，格式为(标签, 路径)"""
        models = self.list_models_with_task(task_code)
        results = []
        for model in models:
            if model.path:
                label = (
                    "[预训练] " if model.is_pretrained else "[训练] "
                ) + model.filename
                results.append((label, str(model.path)))
        return results

    def refresh_model_lists(self) -> tuple[list[str], list[str]]:
        """获取模型名称列表 (用于下拉框)"""
        pretrained_names = [
            model.filename for model in self.list_models(is_pretrained=True)
        ]
        trained_names = [
            model.filename for model in self.list_models(is_pretrained=False)
        ]
        return pretrained_names, trained_names

    def refresh_model_details(self) -> tuple[list[list[str]], list[list[str]]]:
        """获取模型详细信息列表 (用于表格显示)"""
        _, pretrained_rows, trained_rows = self.get_models_summary()
        return pretrained_rows, trained_rows


# 创建全局模型管理器实例
model_manager = ModelManager()


__all__ = [
    "Model",
    "ModelManager",
    "model_manager",
]
