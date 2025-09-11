"""
任务管理模块
负责管理训练任务队列、调度和执行
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"  # 等待中
    QUEUED = "queued"  # 已排队
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    PAUSED = "paused"  # 暂停


class TaskPriority(Enum):
    """任务优先级枚举"""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class TrainingTaskConfig:
    """训练任务配置"""

    task_code: str
    dataset_name: str
    model_path: str
    epochs: int
    lr0: float
    imgsz: int
    batch: int
    device: str
    # 数据增强参数
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    # 训练器参数
    optimizer: str = "auto"
    momentum: float = 0.937
    weight_decay: float = 0.0005
    workers: int = 8


@dataclass
class TaskProgress:
    """任务进度信息"""

    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    lr: float = 0.0
    eta: str = ""

    @property
    def progress_percent(self) -> float:
        """进度百分比"""
        if self.total_epochs <= 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100


@dataclass
class Task:
    """训练任务"""

    id: str
    name: str
    config: TrainingTaskConfig
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    logs: List[str] = field(default_factory=list)
    error_message: str = ""

    def __lt__(self, other):
        """优先级比较（用于优先队列）"""
        return self.priority.value < other.priority.value

    @property
    def duration(self) -> Optional[float]:
        """任务持续时间（秒）"""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class OutputParser:
    """训练输出解析器"""

    def __init__(self):
        self.patterns = {
            "epoch": r"Epoch\s+(\d+)/(\d+)",
            "loss": r"loss:\s*([0-9.]+)",
            "accuracy": r"acc:\s*([0-9.]+)",
            "lr": r"lr:\s*([0-9.e-]+)",
            "eta": r"ETA:\s*([0-9:]+)",
        }

    def parse_line(self, line: str) -> Dict[str, Any]:
        """解析单行输出"""
        import re

        result = {}

        # 解析各种指标
        for key, pattern in self.patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if key == "epoch":
                    result["current_epoch"] = int(match.group(1))
                    result["total_epochs"] = int(match.group(2))
                elif key in ["loss", "accuracy", "lr"]:
                    result[key] = float(match.group(1))
                elif key == "eta":
                    result[key] = match.group(1)

        return result

    def update_progress(self, progress: TaskProgress, parsed_data: Dict[str, Any]):
        """更新进度信息"""
        for key, value in parsed_data.items():
            if hasattr(progress, key):
                setattr(progress, key, value)


class TaskManager:
    """任务管理器"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue = PriorityQueue()
        self.current_task: Optional[Task] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.output_parser = OutputParser()

        # 回调函数
        self.status_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []

    def add_status_callback(self, callback: Callable):
        """添加状态更新回调"""
        self.status_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable):
        """添加进度更新回调"""
        self.progress_callbacks.append(callback)

    def _notify_status_change(self, task: Task):
        """通知状态变化"""
        for callback in self.status_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"状态回调错误: {e}")

    def _notify_progress_update(self, task: Task):
        """通知进度更新"""
        for callback in self.progress_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"进度回调错误: {e}")

    def create_task(
        self,
        name: str,
        config: TrainingTaskConfig,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """创建新任务"""
        task_id = str(uuid.uuid4())[:8]
        task = Task(id=task_id, name=name, config=config, priority=priority)

        self.tasks[task_id] = task
        task.status = TaskStatus.QUEUED
        self.task_queue.put(task)

        logger.info(f"创建任务: {name} (ID: {task_id})")
        self._notify_status_change(task)

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        return list(self.tasks.values())

    def get_queued_tasks(self) -> List[Task]:
        """获取排队中的任务"""
        return [
            task for task in self.tasks.values() if task.status == TaskStatus.QUEUED
        ]

    def promote_task(self, task_id: str) -> bool:
        """提升任务优先级到下一个执行"""
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.QUEUED:
            return False

        # 将优先级设为紧急
        task.priority = TaskPriority.URGENT

        # 重新构建队列
        self._rebuild_queue()

        logger.info(f"任务 {task_id} 已提升到下一个执行")
        self._notify_status_change(task)
        return True

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status == TaskStatus.RUNNING:
            # 如果是当前运行的任务，需要停止执行
            if self.current_task and self.current_task.id == task_id:
                self._stop_current_task()

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()

        logger.info(f"任务 {task_id} 已取消")
        self._notify_status_change(task)
        return True

    def _rebuild_queue(self):
        """重建任务队列"""
        # 获取所有排队中的任务
        queued_tasks = self.get_queued_tasks()

        # 清空当前队列
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except:
                break

        # 重新添加任务
        for task in queued_tasks:
            self.task_queue.put(task)

    def start_processing(self):
        """开始处理任务队列"""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("任务处理器已启动")

    def stop_processing(self):
        """停止处理任务队列"""
        self.is_running = False

        # 停止当前任务
        if self.current_task:
            self._stop_current_task()

        logger.info("任务处理器已停止")

    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get_nowait()
                    if task.status == TaskStatus.QUEUED:
                        self._execute_task(task)
                else:
                    time.sleep(1)  # 没有任务时等待
            except Exception as e:
                logger.error(f"工作循环错误: {e}")
                time.sleep(5)

    def _execute_task(self, task: Task):
        """执行任务"""
        self.current_task = task
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.progress.total_epochs = task.config.epochs

        logger.info(f"开始执行任务: {task.name} (ID: {task.id})")
        self._notify_status_change(task)

        try:
            # 这里调用实际的训练函数
            self._run_training(task)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(f"任务完成: {task.name} (ID: {task.id})")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()
            logger.error(f"任务失败: {task.name} (ID: {task.id}) - {e}")

        finally:
            self.current_task = None
            self._notify_status_change(task)

    def _run_training(self, task: Task):
        """运行训练（这里需要集成实际的训练逻辑）"""
        # 这里应该调用原来的训练逻辑
        # 暂时用模拟训练代替
        config = task.config

        for epoch in range(1, config.epochs + 1):
            if not self.is_running or task.status != TaskStatus.RUNNING:
                break

            # 模拟训练输出
            loss = 1.0 - (epoch / config.epochs) * 0.8
            accuracy = (epoch / config.epochs) * 0.9

            # 更新进度
            task.progress.current_epoch = epoch
            task.progress.loss = loss
            task.progress.accuracy = accuracy

            # 添加日志
            log_line = f"Epoch {epoch}/{config.epochs} - loss: {loss:.4f} - acc: {accuracy:.4f}"
            task.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_line}")

            self._notify_progress_update(task)

            # 模拟训练时间
            time.sleep(1)

    def _stop_current_task(self):
        """停止当前任务"""
        if self.current_task:
            self.current_task.status = TaskStatus.CANCELLED
            self.current_task.completed_at = datetime.now()

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        total_tasks = len(self.tasks)
        queued_count = len(self.get_queued_tasks())
        running_count = 1 if self.current_task else 0
        completed_count = len(
            [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        )
        failed_count = len(
            [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
        )

        return {
            "total_tasks": total_tasks,
            "queued": queued_count,
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "current_task": self.current_task.id if self.current_task else None,
            "is_processing": self.is_running,
        }


# 全局任务管理器实例
task_manager = TaskManager()
