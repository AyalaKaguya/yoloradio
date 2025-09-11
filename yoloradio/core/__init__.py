"""
YoloRadio 核心模块
集成了所有核心功能的统一访问接口
"""

# 数据集管理模块
from .dataset_manager import (
    build_ultra_data_yaml,
    dataset_summary_table,
    delete_dataset,
    detect_structure,
    list_datasets_for_task,
    read_class_names,
    rename_dataset,
    summarize_dataset,
    validate_dataset_by_type,
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
    "build_ultra_data_yaml",
    "extract_pathlike",
    "is_supported_archive",
    "strip_archive_suffix",
    "unwrap_single_root",
    # 数据集管理
    "read_class_names",
    "detect_structure",
    "summarize_dataset",
    "validate_dataset_by_type",
    "list_datasets_for_task",
    "rename_dataset",
    "delete_dataset",
    "dataset_summary_table",
    # 模型管理
    "download_pretrained_if_missing",
    "list_models_for_task",
    "get_model_details",
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
