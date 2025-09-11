"""
YoloRadio 核心模块
集成了所有核心功能的统一访问接口
"""

# 数据集管理模块
from .dataset_manager import (
    Dataset,
    DatasetManager,
    dataset_manager,
    list_datasets_for_task,
)

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
from .model_manager import (
    delete_model,
    download_pretrained_if_missing,
    get_all_models_info,
    get_model_detail,
    get_model_details,
    list_models_for_task,
    rename_model,
)

# 训练管理模块
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
    # 数据集弃用API
    "list_datasets_for_task",
    # 模型管理
    "download_pretrained_if_missing",
    "list_models_for_task",
    "get_all_models_info",
    "get_model_details",
    "get_model_detail",
    "rename_model",
    "delete_model",
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
