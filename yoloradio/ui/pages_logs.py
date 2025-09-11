"""日志页面 - 展示训练结果和日志"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import yaml

from ..core.paths import PROJECT_DIR


def create_logs_tab() -> None:
    """创建日志标签页"""
    gr.Markdown("## 训练日志与结果\n查看训练运行结果、图表和日志文件。")

    # 默认runs目录路径
    runs_dir = PROJECT_DIR / "runs"

    # 状态变量
    refresh_trigger = gr.State(0)
    selected_run_path = gr.State("")

    with gr.Row():
        # 第一栏：runs目录结构 (动态渲染)
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 📁 训练运行目录")

            refresh_btn = gr.Button("🔄 刷新目录", variant="secondary")

            # 使用@gr.render装饰器动态创建目录结构
            @gr.render(inputs=[refresh_trigger])
            def render_runs_structure(trigger):
                """动态渲染runs目录结构"""
                runs_path = Path(runs_dir)

                if not runs_path.exists():
                    gr.Markdown("📂 runs目录不存在")
                    return

                # 遍历runs目录下的主要类别
                categories = [d for d in runs_path.iterdir() if d.is_dir()]

                if not categories:
                    gr.Markdown("📂 runs目录为空")
                    return

                for category_dir in sorted(categories):
                    # 为每个类别创建标题
                    gr.Markdown(f"#### 📁 {category_dir.name}")

                    # 获取类别下的运行目录
                    runs_in_category = [d for d in category_dir.iterdir() if d.is_dir()]

                    if runs_in_category:
                        # 为每个运行创建按钮
                        for run_dir in sorted(runs_in_category):
                            relative_path = str(run_dir.relative_to(runs_path))

                            # 创建选择按钮
                            btn = gr.Button(
                                f"📊 {run_dir.name}",
                                variant="secondary",
                                elem_classes=["run-selector-btn"],
                            )

                            # 绑定点击事件
                            btn.click(
                                lambda path=relative_path: path,
                                outputs=[selected_run_path],
                            )
                    else:
                        gr.Markdown("　　*空目录*", elem_classes=["text-muted"])

        # 第二、三、四栏合并：内容展示区域 (动态渲染)
        with gr.Column(scale=3, min_width=750):
            gr.Markdown("### 📊 日志内容")

            # 使用@gr.render装饰器动态创建内容展示
            @gr.render(inputs=[selected_run_path])
            def render_run_content(run_path):
                """动态渲染选中运行的内容"""
                if not run_path:
                    gr.Markdown("**当前展示**: 请从左侧选择一个训练运行")
                    gr.Markdown(
                        """
                    <div style='text-align: center; padding: 50px; color: #666;'>
                        <h3>📂 请选择要查看的训练运行</h3>
                        <p>从左侧目录中点击训练运行来查看详细内容</p>
                    </div>
                    """
                    )
                    return

                # 显示当前路径
                gr.Markdown(f"**当前展示**: 📁 `{run_path}`")

                try:
                    full_path = runs_dir / run_path
                    if not full_path.exists():
                        gr.Markdown(f"❌ 路径不存在: {run_path}")
                        return

                    # 扫描文件
                    images = []
                    csv_files = []
                    yaml_files = []
                    other_files = []

                    for item in full_path.rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(full_path)

                            if item.suffix.lower() in [
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".gif",
                                ".bmp",
                                ".webp",
                            ]:
                                images.append((str(relative_path), str(item)))
                            elif item.suffix.lower() == ".csv":
                                csv_files.append((str(relative_path), str(item)))
                            elif item.suffix.lower() in [".yaml", ".yml"]:
                                yaml_files.append((str(relative_path), str(item)))
                            else:
                                other_files.append((str(relative_path), str(item)))

                    # 展示图片
                    if images:
                        gr.Markdown("### 🖼️ 图片文件")

                        for img_name, img_path in sorted(images):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    try:
                                        gr.Image(
                                            value=img_path,
                                            label=img_name,
                                            show_label=True,
                                            container=True,
                                            show_download_button=True,
                                        )
                                    except Exception as e:
                                        gr.Markdown(f"❌ 无法加载图片 {img_name}: {e}")

                                with gr.Column(scale=1):
                                    gr.Markdown(f"**文件名**: `{img_name}`")
                                    try:
                                        file_size = os.path.getsize(img_path)
                                        gr.Markdown(f"**大小**: {file_size:,} 字节")
                                    except:
                                        gr.Markdown("**大小**: 未知")

                    # 展示CSV文件
                    if csv_files:
                        gr.Markdown("### 📊 CSV数据文件")

                        for csv_name, csv_path in sorted(csv_files):
                            gr.Markdown(f"#### 📄 {csv_name}")
                            try:
                                df = pd.read_csv(csv_path)
                                gr.Dataframe(
                                    value=df,
                                    label=f"数据表: {csv_name}",
                                    interactive=False,
                                    wrap=True,
                                )
                            except Exception as e:
                                gr.Markdown(f"❌ 无法读取CSV文件: {e}")

                    # 展示YAML文件
                    if yaml_files:
                        gr.Markdown("### ⚙️ YAML配置文件")

                        for yaml_name, yaml_path in sorted(yaml_files):
                            gr.Markdown(f"#### 📄 {yaml_name}")
                            try:
                                with open(yaml_path, "r", encoding="utf-8") as f:
                                    yaml_content = f.read()

                                gr.Code(
                                    value=yaml_content,
                                    language="yaml",
                                    label=f"配置内容: {yaml_name}",
                                    interactive=False,
                                )
                            except Exception as e:
                                gr.Markdown(f"❌ 无法读取YAML文件: {e}")

                    # 展示其他文件
                    if other_files:
                        gr.Markdown("### 📄 其他文件")

                        file_list = []
                        for file_name, file_path in sorted(other_files):
                            try:
                                file_size = os.path.getsize(file_path)
                                file_list.append([file_name, f"{file_size:,} 字节"])
                            except:
                                file_list.append([file_name, "未知大小"])

                        if file_list:
                            gr.Dataframe(
                                value=file_list,
                                headers=["文件名", "大小"],
                                label="文件列表",
                                interactive=False,
                            )

                    # 如果没有找到任何文件
                    if not any([images, csv_files, yaml_files, other_files]):
                        gr.Markdown("📂 该目录下没有找到任何文件")

                except Exception as e:
                    gr.Markdown(f"❌ 加载内容失败: {str(e)}")

    # 事件处理
    def trigger_refresh(current_trigger):
        """刷新目录触发器"""
        return current_trigger + 1

    # 绑定刷新按钮
    refresh_btn.click(
        fn=trigger_refresh, inputs=[refresh_trigger], outputs=[refresh_trigger]
    )
