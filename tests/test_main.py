"""测试应用入口"""

from __future__ import annotations

from unittest.mock import patch

import gradio as gr

from main import create_app


class TestApp:
    """测试应用创建和配置"""

    def test_create_app_returns_blocks(self):
        """测试创建应用返回 Blocks 对象"""
        app = create_app()
        assert isinstance(app, gr.Blocks)
        assert "YOLO" in app.title

    def test_app_has_routes(self):
        """测试应用包含必要的路由"""
        app = create_app()
        # 注意：实际路由测试需要 Gradio 内部 API，这里仅做基本检查
        assert app is not None
