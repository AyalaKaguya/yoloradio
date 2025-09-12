"""
YoloRadio UI模块
集成了所有UI页面组件的统一访问接口
"""

from .pages_datasets import create_datasets_tab
from .pages_export import create_export_tab

# 页面模块导入
from .pages_home import create_home_tab
from .pages_logs import create_logs_tab
from .pages_models import create_models_tab
from .pages_quick import create_quick_tab
from .pages_test import create_test_tab
from .pages_train import create_train_tab
from .pages_val import create_val_tab

# 导出所有公共API
__all__ = [
    "create_home_tab",
    "create_datasets_tab",
    "create_models_tab",
    "create_train_tab",
    "create_val_tab",
    "create_export_tab",
    "create_quick_tab",
    "create_logs_tab",
    "create_test_tab",
]
