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
class ValidationTaskConfig:
    """验证任务配置"""

    task_code: str
    dataset_name: str
    model_path: str
    conf: float = 0.25  # 置信度阈值
    iou: float = 0.5  # IoU阈值
    imgsz: int = 640  # 图像尺寸
    batch: int = 32  # 批大小
    device: str = "auto"
    workers: int = 8
    save_txt: bool = True  # 保存预测结果文本
    save_conf: bool = True  # 保存置信度
    save_crop: bool = False  # 保存裁剪图片
    verbose: bool = True  # 详细输出


@dataclass
class TaskProgress:
    """任务进度信息"""

    current_epoch: int = 0
    total_epochs: int = 0

    @property
    def progress_percent(self) -> float:
        """进度百分比"""
        if self.total_epochs <= 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100


class TaskType(Enum):
    """任务类型枚举"""

    TRAINING = "training"
    VALIDATION = "validation"


@dataclass
class Task:
    """训练任务"""

    id: str
    name: str
    task_type: TaskType
    config: TrainingTaskConfig | ValidationTaskConfig
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


# 移除输出解析器以简化过程指标跟踪


class TaskManager:
    """任务管理器"""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_queue = PriorityQueue()
        self.current_task: Optional[Task] = None
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        # 不再解析训练输出的过程指标

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

    def create_training_task(
        self,
        name: str,
        config: TrainingTaskConfig,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """创建训练任务"""
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            name=name,
            task_type=TaskType.TRAINING,
            config=config,
            priority=priority,
        )

        self.tasks[task_id] = task
        task.status = TaskStatus.QUEUED
        self.task_queue.put(task)

        logger.info(f"创建训练任务: {name} (ID: {task_id})")
        self._notify_status_change(task)

        return task_id

    def create_validation_task(
        self,
        name: str,
        config: ValidationTaskConfig,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """创建验证任务"""
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            name=name,
            task_type=TaskType.VALIDATION,
            config=config,
            priority=priority,
        )

        self.tasks[task_id] = task
        task.status = TaskStatus.QUEUED
        self.task_queue.put(task)

        logger.info(f"创建验证任务: {name} (ID: {task_id})")
        self._notify_status_change(task)

        return task_id

    def create_task(
        self,
        name: str,
        config: TrainingTaskConfig,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """创建新任务（向后兼容，默认为训练任务）"""
        return self.create_training_task(name, config, priority)

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

        # 根据任务类型设置不同的进度参数
        if task.task_type == TaskType.TRAINING and isinstance(
            task.config, TrainingTaskConfig
        ):
            task.progress.total_epochs = task.config.epochs
        elif task.task_type == TaskType.VALIDATION:
            task.progress.total_epochs = 1

        logger.info(f"开始执行任务: {task.name} (ID: {task.id})")
        self._notify_status_change(task)

        try:
            # 根据任务类型调用不同的执行函数
            if task.task_type == TaskType.TRAINING:
                self._run_training(task)
            elif task.task_type == TaskType.VALIDATION:
                self._run_validation(task)
            else:
                raise ValueError(f"未支持的任务类型: {task.task_type}")

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
        """运行训练（集成真实的YOLO训练逻辑）"""
        if not isinstance(task.config, TrainingTaskConfig):
            raise ValueError("训练任务需要TrainingTaskConfig配置")

        config = task.config

        try:
            # 导入真实训练函数
            from .training_manager import start_training as real_start_training

            # 调用真实的训练启动函数
            success, message = real_start_training(
                task_code=config.task_code,
                dataset_name=config.dataset_name,
                model_path=config.model_path,
                epochs=config.epochs,
                lr0=config.lr0,
                imgsz=config.imgsz,
                batch=config.batch,
                device=config.device,
                degrees=config.degrees,
                translate=config.translate,
                scale=config.scale,
                shear=config.shear,
                fliplr=config.fliplr,
                flipud=config.flipud,
                mosaic=config.mosaic,
                mixup=config.mixup,
                optimizer=config.optimizer,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                workers=config.workers,
            )

            if not success:
                raise Exception(message)

            # 获取训练状态并持续监控
            from .training_manager import training_state

            # 等待训练真正开始
            timeout = 30  # 30秒超时
            start_time = time.time()
            while (
                not training_state.is_running and (time.time() - start_time) < timeout
            ):
                time.sleep(0.5)

            if not training_state.is_running:
                raise Exception("训练启动超时")

            # 监控训练进程
            while (
                training_state.is_running
                and self.is_running
                and task.status == TaskStatus.RUNNING
            ):
                # 更新任务进度
                task.progress.current_epoch = training_state.current_epoch
                task.progress.total_epochs = training_state.total_epochs

                # 获取最新日志
                if training_state.log_lines:
                    # 只添加新的日志行
                    new_logs = training_state.log_lines[len(task.logs) :]
                    task.logs.extend(new_logs)

                # 通知进度更新
                self._notify_progress_update(task)

                # 检查是否应该停止
                if not self.is_running or task.status != TaskStatus.RUNNING:
                    # 停止真实训练
                    from .training_manager import stop_training

                    stop_training()
                    break

                time.sleep(2)  # 每2秒检查一次

            # 训练完成处理
            if training_state.is_running:
                # 等待训练自然完成
                while training_state.is_running:
                    time.sleep(1)

            # 最后更新一次状态
            task.progress.current_epoch = training_state.current_epoch
            task.progress.total_epochs = training_state.total_epochs
            if training_state.log_lines:
                task.logs = training_state.log_lines.copy()

        except Exception as e:
            logger.error(f"训练执行失败: {e}")
            task.logs.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 训练失败: {e}"
            )
            raise

    def _run_validation(self, task: Task):
        """运行验证任务"""
        if not isinstance(task.config, ValidationTaskConfig):
            raise ValueError("验证任务需要ValidationTaskConfig配置")

        config = task.config

        try:
            # 导入验证函数和状态
            from .validation_manager import start_validation, validation_state

            task.logs.append(f"开始验证任务: {task.name}")
            task.logs.append(f"模型: {config.model_path}")
            task.logs.append(f"数据集: {config.dataset_name}")
            task.logs.append(f"置信度阈值: {config.conf}")
            task.logs.append(f"IoU阈值: {config.iou}")

            # 调用验证函数
            success, message, results = start_validation(
                task_code=config.task_code,
                dataset_name=config.dataset_name,
                model_path=config.model_path,
                conf=config.conf,
                iou=config.iou,
                imgsz=config.imgsz,
                batch=config.batch,
                device=config.device,
                workers=config.workers,
                save_txt=config.save_txt,
                save_conf=config.save_conf,
                save_crop=config.save_crop,
                verbose=config.verbose,
            )

            if not success:
                raise Exception(message)

            # 等待验证开始
            timeout = 10  # 10秒超时
            start_time = time.time()
            while (
                not validation_state.is_running and (time.time() - start_time) < timeout
            ):
                time.sleep(0.1)

            # 监控验证进程
            while (
                validation_state.is_running
                and self.is_running
                and task.status == TaskStatus.RUNNING
            ):
                # 更新任务进度
                task.progress.current_epoch = validation_state.current_epoch
                task.progress.total_epochs = validation_state.total_epochs

                # 获取验证状态的日志
                if validation_state.log_lines:
                    # 同步验证状态的日志到任务日志
                    task.logs = validation_state.log_lines.copy()

                # 通知进度更新
                self._notify_progress_update(task)

                # 限制日志数量
                if len(task.logs) > 1000:
                    task.logs = task.logs[-500:]

                time.sleep(1)

            # 最终同步一次日志
            if validation_state.log_lines:
                task.logs = validation_state.log_lines.copy()

            # 检查最终状态
            if validation_state.completed_successfully:
                task.logs.append("验证成功完成")

                # 保留结果到日志，不再回填过程指标

                # 记录验证结果
                if validation_state.results:
                    task.logs.append("=== 验证结果 ===")
                    for key, value in validation_state.results.items():
                        task.logs.append(f"{key}: {value}")
            else:
                error_msg = validation_state.error_message or "验证异常结束"
                raise Exception(error_msg)

            task.logs.append("验证任务完成")
            self._notify_progress_update(task)

        except Exception as e:
            error_msg = f"验证失败: {str(e)}"
            task.logs.append(error_msg)
            task.error_message = error_msg
            logger.error(f"任务 {task.id} 验证失败: {e}")
            raise

    def _stop_current_task(self):
        """停止当前任务"""
        if self.current_task:
            # 针对不同任务类型执行真实停止逻辑
            try:
                if self.current_task.task_type == TaskType.TRAINING:
                    from .training_manager import stop_training as _stop_train

                    _stop_train()
                elif self.current_task.task_type == TaskType.VALIDATION:
                    from .validation_manager import stop_validation as _stop_val
                    from .validation_manager import validation_state

                    _stop_val()
            except Exception as _e:
                logger.error(f"停止底层任务进程失败: {_e}")

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
