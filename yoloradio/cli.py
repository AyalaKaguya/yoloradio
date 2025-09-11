"""YoloRadio CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .__version__ import get_version


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YoloRadio - YOLO 可视化训练平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务器主机地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口 (默认: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公共链接 (Gradio share)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"yoloradio {get_version()}",
    )

    args = parser.parse_args()

    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path

        # Add project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from yoloradio.app import create_app

        app = create_app()

        print(f"🚀 启动 YoloRadio 在 http://{args.host}:{args.port}")
        if args.share:
            print("📡 创建公共分享链接...")

        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            show_error=True,
        )

    except KeyboardInterrupt:
        print("\n👋 再见!")
        return 0
    except Exception as e:
        print(f"❌ 启动失败: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
