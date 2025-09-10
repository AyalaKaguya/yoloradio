#!/usr/bin/env python3
"""YoloRadio 开发快速设置脚本"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并显示结果"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """主函数"""
    print("🚀 YoloRadio 开发环境设置\n")

    # 检查 uv 是否安装
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv 未安装，请先安装 uv: https://docs.astral.sh/uv/")
        return 1

    # 同步依赖
    if not run_command(["uv", "sync"], "安装基础依赖"):
        return 1

    # 安装开发依赖
    if not run_command(["uv", "sync", "--extra", "dev"], "安装开发依赖"):
        return 1

    # 测试应用
    if not run_command(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from main import create_app; app = create_app(); print('应用创建测试通过')",
        ],
        "测试应用创建",
    ):
        return 1

    print(
        f"""
🎉 设置完成! 

可用命令:
  运行应用:     uv run python main.py
  使用 CLI:     uv run yoloradio
  运行测试:     uv run pytest
  代码格式化:   uv run black .
  代码检查:     uv run flake8 yoloradio/ main.py
  
在 Windows 上也可以使用:
  dev.bat run      # 运行应用
  dev.bat test     # 运行测试
  dev.bat format   # 代码格式化
  dev.bat lint     # 代码检查

VS Code 集成:
  - 已配置 Python 解释器路径
  - 已设置格式化和检查工具
  - 可使用 Ctrl+Shift+P > Tasks 运行任务
  - 按 F5 启动调试模式
"""
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
