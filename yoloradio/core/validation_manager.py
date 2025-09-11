"""验证管理器 - YOLO模型验证功能"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .dataset_manager import Dataset, dataset_manager
from .paths import PROJECT_DIR

logger = logging.getLogger(__name__)


def create_yolo_config(dataset_obj: Dataset, task_code: str) -> Optional[Path]:
    """
    为数据集创建YOLO兼容的配置文件

    Args:
        dataset_obj: 数据集对象
        task_code: 任务代码

    Returns:
        YOLO配置文件路径，如果失败则返回None
    """
    try:
        # 检查是否已有ultra.yaml文件（优先使用）
        ultra_yaml_path = dataset_obj.path / f"{dataset_obj.name}.ultra.yaml"
        if ultra_yaml_path.exists():
            logger.info(f"发现现有的ultra.yaml配置文件: {ultra_yaml_path}")
            return ultra_yaml_path

        # 读取数据集元数据
        metadata = dataset_obj.metadata
        if not metadata:
            logger.error(f"无法读取数据集元数据: {dataset_obj.name}")
            return None

        # 获取数据集路径
        dataset_path = dataset_obj.path

        # 检查数据集的实际目录结构
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        if not images_dir.exists():
            logger.error(f"数据集 {dataset_obj.name} 缺少images目录: {images_dir}")
            return None

        # 构建YOLO配置（使用相对路径）
        yolo_config = {
            "path": str(dataset_path).replace("\\", "/"),  # YOLO使用正斜杠
            "train": "images",  # 相对于path的路径
            "val": "images",  # 相对于path的路径
            "test": "images",  # 相对于path的路径
            "names": {},
        }

        # 处理类别名称
        if "class_names" in metadata:
            # 如果class_names是列表，转换为字典
            class_names = metadata["class_names"]
            if isinstance(class_names, list):
                yolo_config["names"] = {i: name for i, name in enumerate(class_names)}
            elif isinstance(class_names, dict):
                yolo_config["names"] = class_names
        elif "classes" in metadata:
            yolo_config["names"] = metadata["classes"]
        else:
            # 尝试从classes.txt文件读取
            classes_file = dataset_path / "classes.txt"
            if classes_file.exists():
                with open(classes_file, "r", encoding="utf-8") as f:
                    class_list = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                    yolo_config["names"] = {
                        i: name for i, name in enumerate(class_list)
                    }
            else:
                # 默认类别
                yolo_config["names"] = {0: "object"}

        # 添加类别数量
        yolo_config["nc"] = len(yolo_config["names"])

        # 创建临时配置文件
        temp_config_path = PROJECT_DIR / "temp" / f"{dataset_obj.name}_yolo_config.yaml"
        temp_config_path.parent.mkdir(exist_ok=True)

        # 写入配置文件
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(yolo_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"为数据集 {dataset_obj.name} 创建YOLO配置: {temp_config_path}")
        logger.info(f"配置内容: {yolo_config}")
        return temp_config_path

    except Exception as e:
        logger.error(f"创建YOLO配置失败: {e}")
        return None


def validate_environment() -> Tuple[bool, str]:
    """验证YOLO验证环境"""
    try:
        # 检查ultralytics
        import ultralytics

        return True, f"验证环境正常 (ultralytics {ultralytics.__version__})"
    except ImportError:
        return False, "ultralytics库未安装，请运行: pip install ultralytics"
    except Exception as e:
        return False, f"验证环境检查失败: {str(e)}"


# 验证状态类
class ValidationState:
    """验证状态管理"""

    def __init__(self):
        self.is_running = False
        self.current_task: Optional[str] = None
        self.results = {}
        self.error_message = ""
        self.completed_successfully = False
        self.log_lines: List[str] = []
        self.current_epoch = 0
        self.total_epochs = 1
        self.start_time = None
        self.end_time = None

    def reset(self):
        """重置状态"""
        self.is_running = False
        self.current_task = None
        self.results = {}
        self.error_message = ""
        self.completed_successfully = False
        self.log_lines = []
        self.current_epoch = 0
        self.total_epochs = 1
        self.start_time = None
        self.end_time = None

    def add_log(self, message: str):
        """添加日志"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.log_lines.append(log_line)
        logger.info(message)

    def get_recent_logs(self, count: int = 50) -> List[str]:
        """获取最近的日志"""
        return self.log_lines[-count:] if self.log_lines else []


# 全局验证状态实例
validation_state = ValidationState()


def start_validation(
    task_code: str,
    dataset_name: str,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    batch: int = 32,
    device: str = "auto",
    workers: int = 8,
    save_txt: bool = True,
    save_conf: bool = True,
    save_crop: bool = False,
    verbose: bool = True,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    启动YOLO模型验证

    Args:
        task_code: 任务代码
        dataset_name: 数据集名称
        model_path: 模型路径
        conf: 置信度阈值
        iou: IoU阈值
        imgsz: 图像尺寸
        batch: 批大小
        device: 设备
        workers: 工作进程数
        save_txt: 保存预测结果文本
        save_conf: 保存置信度
        save_crop: 保存裁剪图片
        verbose: 详细输出

    Returns:
        (success: bool, message: str, results: Dict | None)
    """
    global validation_state

    try:
        # 重置验证状态
        validation_state.reset()
        validation_state.is_running = True
        validation_state.current_task = task_code
        validation_state.add_log(f"开始验证任务: {task_code}")

        # 获取数据集信息
        dataset_obj = dataset_manager.get_dataset(dataset_name)
        if dataset_obj is None:
            validation_state.error_message = f"数据集未找到: {dataset_name}"
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

        if not dataset_obj.exists:
            validation_state.error_message = f"数据集目录不存在: {dataset_obj.path}"
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

        # 检查数据集元数据
        if not dataset_obj.metadata_exists:
            validation_state.error_message = (
                f"数据集配置文件不存在: {dataset_obj.metadata_path}"
            )
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

        validation_state.add_log(f"✅ 数据集检查通过: {dataset_name}")

        # 创建YOLO兼容的配置文件
        validation_state.add_log("🔄 创建YOLO配置文件...")
        yolo_config = create_yolo_config(dataset_obj, task_code)
        if yolo_config is None:
            validation_state.error_message = f"无法为数据集 {dataset_name} 创建YOLO配置"
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

        validation_state.add_log(f"✅ YOLO配置文件创建成功: {yolo_config}")

        # 检查模型路径
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            validation_state.error_message = f"模型文件不存在: {model_path}"
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

        validation_state.add_log(f"✅ 模型文件检查通过: {model_path}")

        # 创建运行目录
        runs_dir = PROJECT_DIR / "runs" / "val" / task_code
        runs_dir.mkdir(parents=True, exist_ok=True)
        validation_state.add_log(f"📁 输出目录: {runs_dir}")

        # 记录验证参数
        validation_state.add_log("📋 验证参数:")
        validation_state.add_log(f"  - 置信度阈值: {conf}")
        validation_state.add_log(f"  - IoU阈值: {iou}")
        validation_state.add_log(f"  - 图像尺寸: {imgsz}")
        validation_state.add_log(f"  - 批大小: {batch}")
        validation_state.add_log(f"  - 设备: {device}")

        try:
            # 导入YOLO
            validation_state.add_log("🔄 加载YOLO库...")
            from ultralytics import YOLO

            # 加载模型
            validation_state.add_log("🔄 加载模型...")
            model = YOLO(model_path)
            validation_state.add_log("✅ 模型加载成功")

            # 开始验证
            validation_state.add_log("🚀 开始验证...")
            validation_state.current_epoch = 0
            validation_state.total_epochs = 1

            # 执行验证
            results = model.val(
                data=str(yolo_config),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=workers,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                project=str(runs_dir.parent),
                name=task_code,
                exist_ok=True,
                verbose=verbose,
            )

            validation_state.current_epoch = 1
            validation_state.add_log("✅ 验证执行完成")

            # 提取验证结果
            validation_state.add_log("🔄 提取验证结果...")
            validation_results = {}
            if hasattr(results, "results_dict"):
                validation_results = results.results_dict
            elif hasattr(results, "box"):
                # 提取常见的验证指标
                box_results = results.box
                validation_results = {
                    "mAP50": (
                        float(box_results.map50)
                        if hasattr(box_results, "map50")
                        else 0.0
                    ),
                    "mAP50-95": (
                        float(box_results.map) if hasattr(box_results, "map") else 0.0
                    ),
                    "precision": (
                        float(box_results.mp) if hasattr(box_results, "mp") else 0.0
                    ),
                    "recall": (
                        float(box_results.mr) if hasattr(box_results, "mr") else 0.0
                    ),
                    "f1": float(box_results.f1) if hasattr(box_results, "f1") else 0.0,
                }

            # 格式化结果
            formatted_results = {}
            for key, value in validation_results.items():
                if isinstance(value, (int, float)):
                    formatted_results[key] = round(float(value), 4)
                else:
                    formatted_results[key] = str(value)

            validation_state.results = formatted_results
            validation_state.completed_successfully = True
            validation_state.is_running = False

            # 记录结果
            validation_state.add_log("🎉 验证结果:")
            for key, value in formatted_results.items():
                validation_state.add_log(f"  - {key}: {value}")

            validation_state.add_log("✅ 验证任务完成")

            return True, "验证完成", formatted_results

        except ImportError:
            validation_state.error_message = (
                "ultralytics库未安装，请运行: pip install ultralytics"
            )
            validation_state.add_log(f"❌ {validation_state.error_message}")
            validation_state.is_running = False
            return False, validation_state.error_message, None
        except Exception as e:
            validation_state.error_message = f"验证失败: {str(e)}"
            validation_state.add_log(f"❌ 验证过程出错: {e}")
            validation_state.is_running = False
            return False, validation_state.error_message, None

    except Exception as e:
        validation_state.error_message = f"验证启动失败: {str(e)}"
        validation_state.add_log(f"❌ {validation_state.error_message}")
        validation_state.is_running = False
        return False, validation_state.error_message, None
