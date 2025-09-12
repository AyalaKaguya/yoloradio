"""验证管理器 - YOLO模型验证功能

变更要点：
- 将验证执行改为通过子进程调用 `yolo.exe val` 并异步读取标准输出，实时写入日志。
- 提供更贴近命令行的日志输出，而不是仅用内置提示语。
- 在可用时自动选择设备（GPU/CPU）。
- 结束后尽量从结果文件中解析指标（results.json / results.csv）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .dataset_manager import Dataset, dataset_manager
from .paths import PROJECT_DIR

logger = logging.getLogger(__name__)


def _clean_terminal_output(text: str) -> str:
    """清理终端输出中的控制字符和ANSI转义序列"""
    if not text:
        return text

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    text = text.replace("\r", "")
    return text


def _should_log_line(text: str) -> bool:
    """判断是否应该记录这行日志，过滤纯进度条等"""
    if not text or not text.strip():
        return False

    skip_patterns = [
        r"^\s*\|.*\|\s*\d+%.*$",
        r"^\s*\d+%.*\|.*$",
        r"^\s*[▇█░▉▊▋▌▍▎▏]+\s*$",
        r"^\s*\.{3,}\s*$",
        r"^\s*-{3,}\s*$",
        r"^\s*=+\s*$",
        r"^\s*\.$",
    ]
    for pattern in skip_patterns:
        if re.match(pattern, text):
            return False
    return True


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

    # 明确声明属性类型，便于类型检查器识别
    is_running: bool
    current_task: Optional[str]
    results: Dict[str, Any]
    error_message: str
    completed_successfully: bool
    log_lines: List[str]
    current_epoch: int
    total_epochs: int
    start_time: Optional[float]
    end_time: Optional[float]
    process: Optional[subprocess.Popen]
    thread: Optional[threading.Thread]

    def __init__(self):
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
        self.process = None
        self.thread = None

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
        self.process = None
        self.thread = None

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
        # 重置并标记启动
        validation_state.reset()
        validation_state.is_running = True
        validation_state.current_task = task_code
        validation_state.add_log(f"开始验证任务: {task_code}")

        # 获取数据集信息
        dataset_obj = dataset_manager.get_dataset(dataset_name)
        if dataset_obj is None or not dataset_obj.exists:
            msg = (
                f"数据集未找到: {dataset_name}"
                if dataset_obj is None
                else f"数据集目录不存在: {dataset_obj.path}"
            )
            validation_state.error_message = msg
            validation_state.add_log(f"❌ {msg}")
            validation_state.is_running = False
            return False, msg, None

        if not dataset_obj.metadata_exists:
            msg = f"数据集配置文件不存在: {dataset_obj.metadata_path}"
            validation_state.error_message = msg
            validation_state.add_log(f"❌ {msg}")
            validation_state.is_running = False
            return False, msg, None

        validation_state.add_log(f"✅ 数据集检查通过: {dataset_name}")

        # 创建YOLO配置
        validation_state.add_log("🔄 创建YOLO配置文件...")
        yolo_config = create_yolo_config(dataset_obj, task_code)
        if yolo_config is None:
            msg = f"无法为数据集 {dataset_name} 创建YOLO配置"
            validation_state.error_message = msg
            validation_state.add_log(f"❌ {msg}")
            validation_state.is_running = False
            return False, msg, None
        validation_state.add_log(f"✅ YOLO配置文件创建成功: {yolo_config}")

        # 检查模型
        if not Path(model_path).exists():
            msg = f"模型文件不存在: {model_path}"
            validation_state.error_message = msg
            validation_state.add_log(f"❌ {msg}")
            validation_state.is_running = False
            return False, msg, None
        validation_state.add_log(f"✅ 模型文件检查通过: {model_path}")

        # 创建运行目录
        runs_dir = PROJECT_DIR / "runs" / "val" / task_code
        runs_dir.mkdir(parents=True, exist_ok=True)
        validation_state.add_log(f"📁 输出目录: {runs_dir}")

        # 设备选择
        resolved_device = device
        if device == "auto":
            try:
                import torch  # type: ignore

                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    resolved_device = "0"
                else:
                    resolved_device = "cpu"
            except Exception:
                resolved_device = "cpu"

        # 记录参数
        validation_state.add_log("📋 验证参数:")
        validation_state.add_log(f"  - 置信度阈值: {conf}")
        validation_state.add_log(f"  - IoU阈值: {iou}")
        validation_state.add_log(f"  - 图像尺寸: {imgsz}")
        validation_state.add_log(f"  - 批大小: {batch}")
        validation_state.add_log(f"  - 设备: {resolved_device}")

        # 构建命令
        yolo_exe = Path(sys.executable).parent / "yolo.exe"
        if not yolo_exe.exists():
            # 回退到 Python API，但仍保持异步线程以推送日志
            def run_python_api():
                try:
                    validation_state.add_log("🔄 加载YOLO库...")
                    from ultralytics import YOLO  # type: ignore

                    validation_state.add_log("🔄 加载模型...")
                    model = YOLO(model_path)
                    validation_state.add_log("✅ 模型加载成功")
                    validation_state.add_log("🚀 开始验证...")
                    validation_state.current_epoch = 0
                    validation_state.total_epochs = 1

                    results = model.val(
                        data=str(yolo_config),
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        batch=batch,
                        device=resolved_device,
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

                    # 提取结果
                    validation_state.add_log("🔄 提取验证结果...")
                    validation_results: Dict[str, Any] = {}
                    if hasattr(results, "results_dict"):
                        validation_results = results.results_dict  # type: ignore[attr-defined]
                    elif hasattr(results, "box"):
                        box_results = results.box  # type: ignore[attr-defined]
                        validation_results = {
                            "mAP50": float(getattr(box_results, "map50", 0.0)),
                            "mAP50-95": float(getattr(box_results, "map", 0.0)),
                            "precision": float(getattr(box_results, "mp", 0.0)),
                            "recall": float(getattr(box_results, "mr", 0.0)),
                            "f1": float(getattr(box_results, "f1", 0.0)),
                        }

                    formatted_results: Dict[str, Any] = {}
                    for k, v in validation_results.items():
                        formatted_results[k] = (
                            round(float(v), 4)
                            if isinstance(v, (int, float))
                            else str(v)
                        )

                    validation_state.results = formatted_results
                    validation_state.completed_successfully = True
                    validation_state.add_log("🎉 验证结果:")
                    for key, value in formatted_results.items():
                        validation_state.add_log(f"  - {key}: {value}")
                    validation_state.add_log("✅ 验证任务完成")
                except ImportError:
                    msg = "ultralytics库未安装，请运行: pip install ultralytics"
                    validation_state.error_message = msg
                    validation_state.add_log(f"❌ {msg}")
                    validation_state.completed_successfully = False
                except Exception as e:
                    validation_state.error_message = f"验证失败: {e}"
                    validation_state.add_log(f"❌ 验证过程出错: {e}")
                    validation_state.completed_successfully = False
                finally:
                    validation_state.is_running = False
                    validation_state.process = None

            validation_state.thread = threading.Thread(
                target=run_python_api, daemon=True
            )
            validation_state.thread.start()
            return True, "验证已开始 (Python API)", None

        # 使用CLI方式运行，实时读取输出
        val_args: List[str] = [
            str(yolo_exe),
            "val",
            f"data={yolo_config}",
            f"model={model_path}",
            f"conf={conf}",
            f"iou={iou}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"device={resolved_device}",
            f"workers={workers}",
            f"save_txt={(str(save_txt).lower())}",
            f"save_conf={(str(save_conf).lower())}",
            f"save_crop={(str(save_crop).lower())}",
            f"project={str((PROJECT_DIR / 'runs' / 'val').resolve())}",
            f"name={task_code}",
            "exist_ok=True",
        ]

        def parse_results_files(out_dir: Path) -> Dict[str, Any]:
            # 优先读取 results.json，其次 results.csv
            results_json = out_dir / "results.json"
            if results_json.exists():
                try:
                    with open(results_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    flat: Dict[str, Any] = {}
                    # 常见字段映射
                    mappings = {
                        "metrics/precision(B)": "precision",
                        "metrics/recall(B)": "recall",
                        "metrics/mAP50(B)": "mAP50",
                        "metrics/mAP50-95(B)": "mAP50-95",
                    }
                    for k, v in data.items():
                        key = str(mappings.get(k, k))
                        flat[key] = v
                    return flat
                except Exception:
                    pass

            results_csv = out_dir / "results.csv"
            if results_csv.exists():
                try:
                    lines = (
                        results_csv.read_text(encoding="utf-8", errors="ignore")
                        .strip()
                        .splitlines()
                    )
                    if len(lines) >= 2:
                        headers = [h.strip() for h in lines[0].split(",")]
                        values = [v.strip() for v in lines[-1].split(",")]
                        mapping: Dict[str, Any] = {}
                        for h, v in zip(headers, values):
                            try:
                                mapping[h] = float(v)
                            except Exception:
                                mapping[h] = v
                        # 常见列重命名
                        rename = {
                            "metrics/precision(B)": "precision",
                            "metrics/recall(B)": "recall",
                            "metrics/mAP50(B)": "mAP50",
                            "metrics/mAP50-95(B)": "mAP50-95",
                        }
                        return {rename.get(k, k): v for k, v in mapping.items()}
                except Exception:
                    pass
            return {}

        def run_cli():
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["TERM"] = "dumb"
                env["COLUMNS"] = "120"
                env["LINES"] = "32"

                validation_state.add_log("🚀 调用命令行开始验证...")
                validation_state.current_epoch = 0
                validation_state.total_epochs = 1

                validation_state.process = subprocess.Popen(
                    val_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=PROJECT_DIR,
                    env=env,
                    universal_newlines=True,
                )

                if validation_state.process.stdout:
                    for raw in iter(validation_state.process.stdout.readline, ""):
                        if not validation_state.is_running:
                            break
                        line = _clean_terminal_output(raw.rstrip("\n"))
                        if _should_log_line(line):
                            from datetime import datetime as _dt

                            ts = _dt.now().strftime("%H:%M:%S")
                            validation_state.log_lines.append(f"[{ts}] {line}")

                rc = validation_state.process.wait()

                # 解析结果文件
                out_dir = PROJECT_DIR / "runs" / "val" / task_code
                results_map = parse_results_files(out_dir)
                if results_map:
                    validation_state.results = {
                        k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                        for k, v in results_map.items()
                    }
                    if "mAP50" in validation_state.results:
                        # 供上层显示
                        pass

                validation_state.completed_successfully = rc == 0
                if rc == 0:
                    validation_state.add_log("✅ 验证任务完成")
                else:
                    validation_state.add_log(f"❌ 验证进程返回码: {rc}")

            except Exception as e:
                validation_state.completed_successfully = False
                validation_state.error_message = f"验证执行失败: {e}"
                validation_state.add_log(f"❌ {validation_state.error_message}")
            finally:
                validation_state.is_running = False
                validation_state.process = None

        validation_state.thread = threading.Thread(target=run_cli, daemon=True)
        validation_state.thread.start()
        return True, "验证已开始 (CLI)", None

    except Exception as e:
        msg = f"验证启动失败: {e}"
        validation_state.error_message = msg
        validation_state.add_log(f"❌ {msg}")
        validation_state.is_running = False
        return False, msg, None


def stop_validation() -> Tuple[bool, str]:
    """停止正在进行的验证任务（如果有）"""
    if not validation_state.is_running:
        return False, "没有正在运行的验证任务"
    try:
        if validation_state.process and validation_state.process.poll() is None:
            validation_state.process.terminate()
            try:
                validation_state.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                validation_state.process.kill()
        validation_state.is_running = False
        validation_state.add_log("⏹️ 验证已停止")
        return True, "验证已停止"
    except Exception as e:
        return False, f"停止验证失败: {e}"
