"""
YoloRadio 核心模块
集成了所有核心功能的统一访问接口
"""

# 数据集管理模块
from .dataset_manager import Dataset, DatasetManager, dataset_manager

# 文件操作模块
from .file_utils import (
    ensure_unique_dir,
    extract_pathlike,
    is_supported_archive,
    list_dir,
    safe_extract_tar,
    safe_extract_zip,
    strip_archive_suffix,
    unwrap_single_root,
)

# 模型管理模块
from .model_manager import Model, ModelManager, model_manager

# 任务管理模块
from .task_manager import (
    Task,
    TaskManager,
    TaskPriority,
    TaskStatus,
    TrainingTaskConfig,
    task_manager,
)

# 任务类型定义
from .task_types import (
    DEFAULT_TASK,
    DEFAULT_TASK_DISPLAY,
    TASK_CODE_TO_DISPLAY,
    TASK_MAP,
    get_all_task_codes,
    get_all_task_displays,
    get_task_code,
    get_task_display,
    is_valid_task_code,
    is_valid_task_display,
)

# 训练管理模块
from .train_manager import TrainManager, train_manager
from .training_manager import (
    TrainingState,
    clear_training_logs,
    get_device_info,
    get_training_logs,
    get_training_status,
    pause_training,
    resume_training,
    start_training,
    stop_training,
    training_state,
    validate_training_environment,
)

# 导出所有公共API
__all__ = [
    # 任务类型
    "TASK_MAP",
    "TASK_CODE_TO_DISPLAY",
    "DEFAULT_TASK",
    "DEFAULT_TASK_DISPLAY",
    "get_task_code",
    "get_task_display",
    "get_all_task_displays",
    "get_all_task_codes",
    "is_valid_task_code",
    "is_valid_task_display",
    # 文件操作
    "list_dir",
    "safe_extract_zip",
    "safe_extract_tar",
    "ensure_unique_dir",
    "extract_pathlike",
    "is_supported_archive",
    "strip_archive_suffix",
    "unwrap_single_root",
    # 数据集管理
    "dataset_manager",
    "Dataset",
    "DatasetManager",
    # 模型管理
    "Model",
    "ModelManager",
    "model_manager",
    # 训练管理
    "TrainingState",
    "training_state",
    "start_training",
    "pause_training",
    "resume_training",
    "stop_training",
    "get_training_status",
    "get_training_logs",
    "clear_training_logs",
    "get_device_info",
    "validate_training_environment",
]
