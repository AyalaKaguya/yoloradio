"""
训练管理模块
负责YOLO模型的训练流程控制、状态管理和日志处理
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from .dataset_manager import build_ultra_data_yaml
from .paths import DATASETS_DIR, MODELS_TRAINED_DIR, PROJECT_DIR

# 配置日志
logger = logging.getLogger(__name__)


def _clean_terminal_output(text: str) -> str:
    """清理终端输出中的控制字符和ANSI转义序列"""
    if not text:
        return text

    # 移除ANSI转义序列（颜色、光标控制等）
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    # 移除其他控制字符
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # 移除回车符（保留换行符）
    text = text.replace("\r", "")

    # 清理多余的空格
    text = " ".join(text.split())

    return text


def _should_log_line(text: str) -> bool:
    """判断是否应该记录这行日志"""
    if not text.strip():
        return False

    # 跳过纯进度条和重复的状态信息
    skip_patterns = [
        r"^\s*\|.*\|\s*\d+%.*$",  # 进度条
        r"^\s*\d+%.*\|.*$",  # 百分比进度条
        r"^\s*[▇█░▉▊▋▌▍▎▏]+\s*$",  # 纯进度条字符
        r"^\s*\.{3,}\s*$",  # 多个点（3个或以上）
        r"^\s*-{3,}\s*$",  # 分隔线（3个或以上）
        r"^\s*=+\s*$",  # 等号分隔线
        r"^\s*\.\s*$",  # 单独的点
    ]

    for pattern in skip_patterns:
        if re.match(pattern, text):
            return False

    return True


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

        # 智能设备选择
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    device = "0"  # 使用第一个GPU
                else:
                    device = "cpu"  # 回退到CPU
            except ImportError:
                device = "cpu"  # 如果torch未安装，使用CPU

        # 生成运行ID
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建数据配置文件
        data_yaml = build_ultra_data_yaml(dataset_name)

        # 准备输出目录
        runs_dir = PROJECT_DIR / "runs" / "train"
        runs_dir.mkdir(parents=True, exist_ok=True)
        project_dir = runs_dir / f"{task_code}_{timestamp}_{run_id}"

        # 准备训练参数 - 使用虚拟环境中的yolo命令
        yolo_exe = Path(sys.executable).parent / "yolo.exe"

        train_args = [
            str(yolo_exe),
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
                # 设置环境变量避免控制字符
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["TERM"] = "dumb"  # 禁用颜色和控制字符
                env["COLUMNS"] = "80"  # 固定列宽
                env["LINES"] = "24"  # 固定行数

                training_state.process = subprocess.Popen(
                    train_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",  # 忽略编码错误
                    universal_newlines=True,
                    cwd=PROJECT_DIR,
                    env=env,
                )

                # 读取输出并清理控制字符
                if training_state.process.stdout:
                    for line in iter(training_state.process.stdout.readline, ""):
                        if not training_state.is_running:
                            break

                        # 清理控制字符和ANSI转义序列
                        line = _clean_terminal_output(line.strip())
                        if line and _should_log_line(line):
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
            meta_data = {
                "task": task_code,
                "name": display_name,
                "description": f"基于 {base_model_name} 在 {dataset_name} 数据集上训练的最佳权重",
                "base_model": base_model_name,
                "dataset": dataset_name,
                "training_date": datetime.now().isoformat(),
                "model_type": "best",
                "epochs_trained": training_state.total_epochs,
                "created_at": datetime.now().isoformat(),
            }
            meta_path = MODELS_TRAINED_DIR / f"{uuid_dest.stem}.yml"
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    meta_data, f, allow_unicode=True, default_flow_style=False
                )

            saved_models.append((uuid_name, display_name))

        if last_pt.exists():
            # UUID文件名
            uuid_name = f"{str(uuid.uuid4())[:8]}.pt"
            uuid_dest = MODELS_TRAINED_DIR / uuid_name
            shutil.copy2(last_pt, uuid_dest)

            # 显示名称
            display_name = f"{task_code}-{base_model_name}-{date_str}-latest"

            # 创建元数据
            meta_data = {
                "task": task_code,
                "name": display_name,
                "description": f"基于 {base_model_name} 在 {dataset_name} 数据集上训练的最新权重",
                "base_model": base_model_name,
                "dataset": dataset_name,
                "training_date": datetime.now().isoformat(),
                "model_type": "latest",
                "epochs_trained": training_state.total_epochs,
                "created_at": datetime.now().isoformat(),
            }
            meta_path = MODELS_TRAINED_DIR / f"{uuid_dest.stem}.yml"
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    meta_data, f, allow_unicode=True, default_flow_style=False
                )

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
