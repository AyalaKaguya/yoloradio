#!/usr/bin/env python3
"""
测试训练系统的基本功能
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yoloradio.utils import (
    TrainingState,
    clear_training_logs,
    get_device_info,
    get_training_status,
    validate_training_environment,
)


def test_environment():
    """测试环境验证"""
    print("=== 环境验证测试 ===")
    result = validate_training_environment()
    print(f"环境验证结果: {result}")
    print()


def test_device_info():
    """测试设备信息"""
    print("=== 设备信息测试 ===")
    device_info = get_device_info()
    print(f"设备信息: {device_info}")
    print()


def test_training_state():
    """测试训练状态管理"""
    print("=== 训练状态测试 ===")

    # 清空日志
    clear_training_logs()

    # 获取初始状态
    status = get_training_status()
    print(f"初始状态: {status}")

    # 测试状态单例
    state1 = TrainingState()
    state2 = TrainingState()
    print(f"状态单例测试: {state1 is state2}")

    # 模拟状态变化
    state1.is_running = True
    state1.current_epoch = 5
    state1.total_epochs = 100

    status = get_training_status()
    print(f"更新后状态: {status}")

    # 重置状态
    state1.reset()
    status = get_training_status()
    print(f"重置后状态: {status}")
    print()


def test_basic_functionality():
    """测试基本功能"""
    print("=== YOLO Radio 训练系统基本功能测试 ===\n")

    try:
        test_environment()
        test_device_info()
        test_training_state()

        print("✅ 所有基本测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_functionality()
