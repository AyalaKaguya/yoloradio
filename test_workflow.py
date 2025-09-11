#!/usr/bin/env python3
"""
模拟训练功能测试脚本
测试YOLO Radio的训练系统是否能正常工作
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yoloradio.utils import (
    clear_training_logs,
    get_device_info,
    get_training_logs,
    get_training_status,
    pause_training,
    resume_training,
    start_training,
    stop_training,
    validate_training_environment,
)


def test_training_workflow():
    """测试完整的训练工作流"""
    print("=== YOLO Radio 训练工作流测试 ===\n")

    # 1. 环境验证
    print("1. 验证训练环境...")
    success, message = validate_training_environment()
    print(f"   环境验证: {'✅ 通过' if success else '❌ 失败'}")
    print(f"   详情: {message}")
    print()

    # 2. 设备信息
    print("2. 获取设备信息...")
    device_info = get_device_info()
    print(f"   设备信息: {device_info}")
    print()

    # 3. 清空日志
    print("3. 清空训练日志...")
    clear_training_logs()
    logs = get_training_logs()
    print(f"   日志行数: {len(logs)}")
    print()

    # 4. 检查初始状态
    print("4. 检查初始训练状态...")
    status = get_training_status()
    print(f"   运行状态: {status['is_running']}")
    print(f"   暂停状态: {status['is_paused']}")
    print(f"   当前轮次: {status['current_epoch']}/{status['total_epochs']}")
    print(f"   进度: {status['progress']:.1f}%")
    print()

    # 5. 测试暂停/恢复（应该失败，因为没有训练运行）
    print("5. 测试训练控制（无运行时）...")

    success, msg = pause_training()
    print(f"   暂停训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

    success, msg = resume_training()
    print(f"   恢复训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

    success, msg = stop_training()
    print(f"   停止训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")
    print()

    # 6. 模拟开始训练（这会失败因为没有真实的模型/数据集）
    print("6. 测试开始训练（模拟）...")
    success, msg = start_training(
        task_code="detect",
        dataset_name="test_dataset",
        model_path="test_model.pt",
        epochs=10,
        lr0=0.01,
        imgsz=640,
        batch=16,
        device="auto",
    )
    print(f"   开始训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

    # 7. 如果训练开始了，测试控制功能
    if success:
        print("\n7. 测试训练控制...")
        time.sleep(1)  # 等待一下

        # 测试暂停
        success, msg = pause_training()
        print(f"   暂停训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

        time.sleep(1)

        # 测试恢复
        success, msg = resume_training()
        print(f"   恢复训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

        time.sleep(1)

        # 测试停止
        success, msg = stop_training()
        print(f"   停止训练: {'✅ 成功' if success else '❌ 失败'} - {msg}")

    # 8. 最终状态检查
    print(f"\n8. 最终状态检查...")
    status = get_training_status()
    logs = get_training_logs()
    print(f"   运行状态: {status['is_running']}")
    print(f"   日志行数: {len(logs)}")
    print(f"   最近日志: {logs[-3:] if logs else '无日志'}")

    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_training_workflow()
