#!/usr/bin/env python3
"""
测试设备选择修复
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yoloradio.utils import get_device_info, validate_training_environment


def test_device_detection():
    """测试设备检测修复"""
    print("=== 设备检测测试 ===")

    # 1. 检查 PyTorch CUDA 状态
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        print(f"CUDA 设备数: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            try:
                print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
                print(f"设备名称: {torch.cuda.get_device_name(0)}")
            except:
                print("CUDA 设备信息获取失败")
    except Exception as e:
        print(f"PyTorch 检查错误: {e}")

    print()

    # 2. 验证训练环境
    print("2. 验证训练环境...")
    success, message = validate_training_environment()
    print(f"环境验证: {'✅ 通过' if success else '❌ 失败'}")
    print(f"详情:\n{message}")
    print()

    # 3. 获取设备信息
    print("3. 获取设备信息...")
    device_info = get_device_info()
    print(f"设备信息: {device_info}")
    print()

    # 4. 测试设备选择逻辑
    print("4. 测试设备选择逻辑...")
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "0"
            print(f"建议设备: GPU {device}")
        else:
            device = "cpu"
            print(f"建议设备: {device}")
    except ImportError:
        device = "cpu"
        print(f"建议设备: {device} (PyTorch 未安装)")


if __name__ == "__main__":
    test_device_detection()
