"""训练管理器 - 与任务管理器集成的训练控制器"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .task_manager import (
    Task,
    TaskManager,
    TaskPriority,
    TaskStatus,
    TrainingTaskConfig,
    task_manager,
)
from .task_types import TASK_MAP
from .training_manager import get_device_info, validate_training_environment


class TrainManager:
    """训练管理器 - 基于任务管理器的训练控制器"""

    def __init__(self):
        self.task_manager = task_manager
        self._environment_checked = False
        self._environment_status = ""
        self._status_callbacks = []
        self._log_callbacks = []
        self._monitoring_active = False
        self._monitor_thread = None

        # 启动任务处理器
        self.task_manager.start_processing()

        # 设置回调
        self.task_manager.add_status_callback(self._on_task_status_change)
        self.task_manager.add_progress_callback(self._on_task_progress_update)

    def _on_task_status_change(self, task: Task):
        """任务状态变化回调"""
        for callback in self._status_callbacks:
            try:
                callback(self._format_task_status(task))
            except Exception as e:
                print(f"状态回调错误: {e}")

    def _on_task_progress_update(self, task: Task):
        """任务进度更新回调"""
        for callback in self._log_callbacks:
            try:
                callback(self._format_task_logs(task))
            except Exception as e:
                print(f"日志回调错误: {e}")

    def _format_task_status(self, task: Task) -> Tuple[str, str, str]:
        """格式化任务状态为UI显示格式"""
        if task.status == TaskStatus.RUNNING:
            status_text = f"状态: 训练中 - Epoch {task.progress.current_epoch}/{task.progress.total_epochs} ({task.progress.progress_percent:.1f}%)"
        elif task.status == TaskStatus.QUEUED:
            status_text = f"状态: 排队中 (优先级: {task.priority.name})"
        elif task.status == TaskStatus.COMPLETED:
            status_text = "状态: 已完成"
        elif task.status == TaskStatus.FAILED:
            status_text = f"状态: 失败 - {task.error_message}"
        elif task.status == TaskStatus.CANCELLED:
            status_text = "状态: 已取消"
        else:
            status_text = "状态: 就绪"

        # 训练信息
        if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
            info_text = f"""**任务ID**: {task.id}
**名称**: {task.name}
**进度**: {task.progress.current_epoch}/{task.progress.total_epochs} ({task.progress.progress_percent:.1f}%)
**Loss**: {task.progress.loss:.4f}
**准确率**: {task.progress.accuracy:.4f}
**日志行数**: {len(task.logs)}"""
        else:
            info_text = f"""**任务ID**: {task.id}
**名称**: {task.name}
**状态**: {task.status.value}
**优先级**: {task.priority.name}"""

        # 日志文本（最近20行）
        log_text = "\n".join(task.logs[-20:])

        return status_text, info_text, log_text

    def _format_task_logs(self, task: Task) -> str:
        """格式化任务日志"""
        return "\n".join(task.logs[-50:])  # 最近50行

    def add_status_callback(self, callback):
        """添加状态更新回调"""
        self._status_callbacks.append(callback)

    def add_log_callback(self, callback):
        """添加日志更新回调"""
        self._log_callbacks.append(callback)

    def get_environment_status(self) -> str:
        """获取环境检查状态"""
        if not self._environment_checked:
            self.check_environment()
        return self._environment_status

    def check_environment(self) -> str:
        """检查训练环境"""
        success, message = validate_training_environment()
        self._environment_status = f"🔍 环境检查: {message}"
        self._environment_checked = True
        return self._environment_status

    def get_device_info(self) -> str:
        """获取设备信息"""
        return get_device_info()

    def create_training_config(
        self,
        task_display: str,
        dataset: str,
        model: str,
        epochs: int,
        lr0: float,
        imgsz: int,
        batch: int,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        fliplr: float,
        flipud: float,
        mosaic: float,
        mixup: float,
        optimizer: str,
        momentum: float,
        weight_decay: float,
        device: str,
        workers: int,
    ) -> str:
        """创建训练配置的TOML字符串"""
        task_code = TASK_MAP.get(task_display, "detect")

        lines = []
        lines.append(f'task = "{task_code}"')
        if dataset:
            lines.append(f'dataset = "{dataset}"')
        if model:
            lines.append(f'model = "{model}"')
        lines.append("")
        lines.append("[train]")
        lines.append(f"epochs = {int(epochs)}")
        lines.append(f"lr0 = {float(lr0)}")
        lines.append(f"imgsz = {int(imgsz)}")
        lines.append(f"batch = {int(batch)}")
        lines.append("")
        lines.append("[augment]")
        lines.append(f"degrees = {float(degrees)}")
        lines.append(f"translate = {float(translate)}")
        lines.append(f"scale = {float(scale)}")
        lines.append(f"shear = {float(shear)}")
        lines.append(f"fliplr = {float(fliplr)}")
        lines.append(f"flipud = {float(flipud)}")
        lines.append(f"mosaic = {float(mosaic)}")
        lines.append(f"mixup = {float(mixup)}")
        lines.append("")
        lines.append("[trainer]")
        lines.append(f'optimizer = "{optimizer}"')
        lines.append(f"momentum = {float(momentum)}")
        lines.append(f"weight_decay = {float(weight_decay)}")
        lines.append(f'device = "{device}"')
        lines.append(f"workers = {int(workers)}")
        return "\n".join(lines) + "\n"

    def start_training(
        self,
        task_display: str,
        dataset: str,
        model_label: str,
        epochs: int,
        lr0: float,
        imgsz: int,
        batch: int,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        fliplr: float,
        flipud: float,
        mosaic: float,
        mixup: float,
        optimizer: str,
        momentum: float,
        weight_decay: float,
        device: str,
        workers: int,
        model_map: Dict[str, str],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Tuple[str, str]:
        """创建并添加训练任务到队列"""
        if not dataset or not model_label:
            return "❌ 请选择数据集和模型", "状态: 就绪"

        if model_label not in model_map:
            return "❌ 模型路径无效", "状态: 就绪"

        model_path = model_map[model_label]
        task_code = TASK_MAP.get(task_display, "detect")

        # 创建训练配置
        config = TrainingTaskConfig(
            task_code=task_code,
            dataset_name=dataset,
            model_path=model_path,
            epochs=epochs,
            lr0=lr0,
            imgsz=imgsz,
            batch=batch,
            device=device,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fliplr=fliplr,
            flipud=flipud,
            mosaic=mosaic,
            mixup=mixup,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            workers=workers,
        )

        # 创建任务名称
        task_name = f"{task_display}-{dataset}-{model_label}"

        # 添加任务到队列
        task_id = self.task_manager.create_task(
            name=task_name, config=config, priority=priority
        )

        return f"✅ 训练任务已添加到队列 (ID: {task_id})", "状态: 已排队"

    def stop_training(self, task_id: Optional[str] = None) -> Tuple[str, str]:
        """停止训练任务"""
        if task_id:
            # 停止特定任务
            success = self.task_manager.cancel_task(task_id)
            if success:
                return f"⏹️ 任务 {task_id} 已停止", "状态: 已停止"
            else:
                return f"❌ 无法停止任务 {task_id}", "状态: 错误"
        else:
            # 停止当前任务
            current = self.task_manager.current_task
            if current:
                success = self.task_manager.cancel_task(current.id)
                if success:
                    return f"⏹️ 当前任务已停止", "状态: 已停止"
                else:
                    return f"❌ 无法停止当前任务", "状态: 错误"
            else:
                return "❌ 没有正在运行的任务", "状态: 就绪"

    def promote_task(self, task_id: str) -> bool:
        """提升任务优先级"""
        return self.task_manager.promote_task(task_id)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        return self.task_manager.get_all_tasks()

    def get_current_task(self) -> Optional[Task]:
        """获取当前任务"""
        return self.task_manager.current_task

    def get_task_queue_status(self) -> Dict:
        """获取任务队列状态"""
        return self.task_manager.get_status_summary()

    def clear_logs(self, task_id: Optional[str] = None) -> Tuple[str, str]:
        """清空日志"""
        if task_id:
            task = self.task_manager.get_task(task_id)
            if task:
                task.logs.clear()
                return "", f"任务 {task_id} 日志已清空"
            else:
                return "", "任务不存在"
        else:
            # 清空当前任务日志
            current = self.task_manager.current_task
            if current:
                current.logs.clear()
                return "", "当前任务日志已清空"
            else:
                return "", "没有当前任务"

    def get_current_status(self) -> Tuple[str, str, str]:
        """获取当前训练状态显示"""
        current_task = self.task_manager.current_task
        if current_task:
            return self._format_task_status(current_task)
        else:
            # 没有当前任务，显示队列状态
            status_summary = self.task_manager.get_status_summary()
            status_text = f"状态: 就绪 - 队列中有 {status_summary['queued']} 个任务"
            info_text = f"""**队列状态**:
排队: {status_summary['queued']}
运行: {status_summary['running']}
完成: {status_summary['completed']}
失败: {status_summary['failed']}"""
            log_text = ""
            return status_text, info_text, log_text

    def start_monitoring(self):
        """开始监控训练状态"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """停止监控训练状态"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread = None

    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring_active:
            try:
                status_text, info_text, log_text = self.get_current_status()
                for callback in self._status_callbacks:
                    try:
                        callback(status_text, info_text, log_text)
                    except Exception as e:
                        print(f"监控回调错误: {e}")
                time.sleep(2.0)  # 每2秒更新一次
            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(5.0)  # 出错时等待更长时间


# 全局训练管理器实例
train_manager = TrainManager()
