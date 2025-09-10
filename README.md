# YoloRadio - YOLO 可视化训练平台

基于 Gradio 和 Ultralytics 的 YOLO 模型可视化训练管理平台，提供数据集管理、模型下载、训练配置等功能。

## 特性

- 🗂️ **数据集管理**: 支持压缩包上传、结构验证、元数据管理
- 🤖 **模型管理**: 预训练模型下载、训练模型管理、统一元数据
- 🏋️ **训练配置**: 任务约束选择、超参数配置、实时 TOML 预览
- 📊 **可视化界面**: 基于 Gradio 的现代化 Web 界面
- 🔄 **多任务支持**: 图像分类、目标检测、图像分割、关键点跟踪、旋转框检测

## 快速开始

### 环境要求

- Python 3.13+
- PyTorch (建议 CUDA 支持)
- 8GB+ RAM (训练时)

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd yoloradio

# 安装依赖 (使用 uv)
uv sync

# 或使用 pip
pip install -e .
```

### 运行

```bash
# 激活虚拟环境 (uv)
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 启动应用
python main.py
```

访问 http://localhost:7860 开始使用。

## 项目结构

```
yoloradio/
├── main.py                 # 应用入口
├── yoloradio/             # 核心模块
│   ├── __init__.py
│   ├── paths.py           # 路径配置
│   ├── utils.py           # 工具函数
│   ├── pages_home.py      # 主页
│   ├── pages_datasets.py  # 数据集管理
│   ├── pages_models.py    # 模型管理
│   ├── pages_train.py     # 训练配置
│   ├── pages_val.py       # 验证页面
│   ├── pages_export.py    # 导出页面
│   └── pages_quick.py     # 快速应用
├── Datasets/              # 数据集目录 (git ignored)
├── Models/                # 模型目录 (git ignored)
│   ├── pretrained/        # 预训练模型
│   └── trained/           # 训练模型
├── runs/                  # 训练日志 (git ignored)
├── pyproject.toml         # 项目配置
└── README.md              # 项目文档
```

## 使用指南

### 数据集管理

1. **上传数据集**: 支持 .zip/.tar.gz 格式，自动验证结构
2. **数据集类型**:
   - 图像分类 (classify)
   - 目标检测 (detect)
   - 图像分割 (segment)
   - 关键点跟踪 (pose)
   - 旋转检测框 (obb)
3. **元数据**: 自动生成 `.yml` 文件记录类型、描述等信息

### 模型管理

1. **预训练模型**: 从 Ultralytics 下载 YOLOv8/v11 预训练权重
2. **训练模型**: 管理本地训练生成的模型
3. **元数据**: 记录任务类型、版本、描述等信息

### 训练配置

1. **任务选择**: 选择任务类型，自动过滤兼容的数据集和模型
2. **超参数**: 核心参数 + 高级参数折叠面板
3. **实时预览**: 参数变更实时生成 TOML 配置预览

## 开发

### 代码结构

- `paths.py`: 定义项目路径常量
- `utils.py`: 核心工具函数 (文件操作、数据集处理、模型管理)
- `pages_*.py`: 各页面渲染逻辑

### 扩展开发

1. **新增页面**: 在 `yoloradio/` 下创建 `pages_xxx.py`，实现 `render()` 函数
2. **注册路由**: 在 `main.py` 中添加路由配置
3. **工具函数**: 在 `utils.py` 中添加通用功能

### 元数据格式

数据集元数据 (`Datasets/<name>.yml`):

```yaml
type: detect # 任务类型
name: my_dataset # 数据集名称
description: "..." # 描述
created_at: "..." # 创建时间
```

模型元数据 (`Models/*/model.yml`):

```yaml
task: detect # 任务类型
version: v11 # YOLO版本
size: n # 模型大小
description: "..." # 描述
created_at: "..." # 创建时间
```

## 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO 实现
- [Gradio](https://github.com/gradio-app/gradio) - Web 界面框架
