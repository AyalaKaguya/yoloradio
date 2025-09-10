# 批处理脚本等价的开发工具 (Windows)

@echo off
setlocal enabledelayedexpansion

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="dev" goto dev
if "%1"=="clean" goto clean
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="check" goto check
if "%1"=="run" goto run
if "%1"=="run-cli" goto run-cli
if "%1"=="run-dev" goto run-dev
if "%1"=="build" goto build

:help
echo YoloRadio 开发工具 (Windows)
echo.
echo 可用命令:
echo   install     安装项目依赖
echo   dev         安装开发依赖
echo   clean       清理缓存和构建文件
echo   test        运行测试
echo   lint        代码检查
echo   format      代码格式化
echo   check       完整检查 (lint + test)
echo   run         运行应用
echo   run-cli     使用 CLI 运行应用
echo   run-dev     开发模式运行 (带调试)
echo   build       构建分发包
goto end

:install
uv sync
goto end

:dev
uv sync --extra dev
goto end

:clean
if exist __pycache__ rmdir /s /q __pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del .coverage
if exist htmlcov rmdir /s /q htmlcov
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del "%%f"
goto end

:test
uv run pytest
goto end

:lint
uv run flake8 yoloradio\ main.py
uv run mypy yoloradio\ main.py
goto end

:format
uv run black yoloradio\ main.py
uv run isort yoloradio\ main.py
goto end

:check
call :lint
call :test
goto end

:run
uv run python main.py
goto end

:run-cli
uv run yoloradio
goto end

:run-dev
uv run yoloradio --debug
goto end

:build
uv build
goto end

:end
endlocal
