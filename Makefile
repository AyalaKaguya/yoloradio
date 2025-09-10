# Makefile for YoloRadio development

.PHONY: help install dev clean test lint format check run build docs

help: ## 显示帮助信息
	@echo "YoloRadio 开发工具"
	@echo ""
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装项目依赖
	uv sync

dev: ## 安装开发依赖
	uv sync --extra dev

clean: ## 清理缓存和构建文件
	rm -rf __pycache__ .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test: ## 运行测试
	uv run pytest

lint: ## 代码检查
	uv run flake8 yoloradio/ main.py
	uv run mypy yoloradio/ main.py

format: ## 代码格式化
	uv run black yoloradio/ main.py
	uv run isort yoloradio/ main.py

check: lint test ## 完整检查 (lint + test)

run: ## 运行应用
	uv run python main.py

run-cli: ## 使用 CLI 运行应用
	uv run yoloradio

run-dev: ## 开发模式运行 (带调试)
	uv run yoloradio --debug

build: ## 构建分发包
	uv build

docs: ## 生成文档 (占位)
	@echo "文档生成功能待实现"

# 快捷方式
fmt: format
t: test
l: lint
r: run
