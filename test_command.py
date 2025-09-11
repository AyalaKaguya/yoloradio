#!/usr/bin/env python3
"""
测试训练命令生成
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_training_command():
    """测试训练命令生成"""
    print("=== 训练命令测试 ===")

    # 模拟生成训练命令
    import sys
    from pathlib import Path

    # 模拟参数
    dataset_name = "test_dataset"
    model_path = "test_model.pt"
    epochs = 10
    lr0 = 0.01
    imgsz = 640
    batch = 16
    device = "auto"

    # 智能设备选择
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = "0"  # 使用第一个GPU
                print(f"检测到CUDA，使用GPU: {device}")
            else:
                device = "cpu"  # 回退到CPU
                print(f"未检测到CUDA，使用CPU: {device}")
        except ImportError:
            device = "cpu"  # 如果torch未安装，使用CPU
            print(f"PyTorch未安装，使用CPU: {device}")

    # 生成命令
    from pathlib import Path

    runs_dir = Path("runs/train")
    data_yaml = f"Datasets/{dataset_name}/data.yaml"
    timestamp = "20241211_090000"
    run_id = "abc12345"

    train_args = [
        sys.executable,  # 使用当前Python解释器
        "-m",
        "ultralytics.yolo",  # 作为模块运行
        "train",
        f"data={data_yaml}",
        f"model={model_path}",
        f"epochs={epochs}",
        f"lr0={lr0}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"project={runs_dir}",
        f"name=detect_{timestamp}_{run_id}",
        f"device={device}",
        "save=True",
        "plots=True",
        "val=True",
    ]

    print(f"\n生成的训练命令:")
    print(" ".join(train_args))

    print(f"\n当前Python解释器: {sys.executable}")

    # 测试ultralytics模块是否可用
    try:
        import ultralytics

        print(f"Ultralytics版本: {ultralytics.__version__}")
        print("✅ Ultralytics模块可用")
    except ImportError as e:
        print(f"❌ Ultralytics导入失败: {e}")


if __name__ == "__main__":
    test_training_command()
