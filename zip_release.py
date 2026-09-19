#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旧版归档工具函数；v1 命令入口已停用，仅支持原始码发行

功能：
1. 打包 CapsWriter-Offline（服务端+客户端）
3. 智能排除模型文件（.onnx, .dll, .json 等），但保留说明文档
"""

import os
import math
import subprocess
from pathlib import Path
from datetime import datetime

ZIP_RELEASE_TIMEOUT_ENV = "CAPSWRITER_ZIP_RELEASE_TIMEOUT"
DEFAULT_ZIP_RELEASE_TIMEOUT_SECONDS = 900
REQUIRED_EMPTY_RELEASE_DIRECTORIES = ("models", "logs")


def _format_seconds(value):
    return f"{value:g}"


def _positive_float_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def zip_release_timeout_seconds():
    return _positive_float_env(
        ZIP_RELEASE_TIMEOUT_ENV,
        DEFAULT_ZIP_RELEASE_TIMEOUT_SECONDS,
    )


def find_7zip():
    """查找 7zip 可执行文件"""
    possible_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]

    # 从 PATH 环境变量查找
    for path in os.environ.get("PATH", "").split(os.pathsep):
        possible_paths.append(os.path.join(path, "7z.exe"))

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def should_include_file(file_path, is_client_only=False):
    """
    判断文件是否应该被打包

    打包规则：
    - 所有文件，除了 models/模型名/子目录/... 的内容
    - models/模型名/文件 会被打包（层级深度 == 2）
    - models/模型名/子目录/文件  不会被打包（层级深度 >= 3）
    - 如果是【仅客户端】打包：
        - 排除 core 目录下的所有 .dll 文件（客户端不需要本地识别引擎）
    """
    path = Path(file_path)
    parts = path.parts

    # 1. 客户端特殊排除逻辑
    if is_client_only:
        # 排除 core 中的 dll 文件
        if 'core' in parts and path.suffix.lower() == '.dll':
            return False

    # 2. 检查是否在 models 目录下
    if 'models' not in parts:
        return True  # 非 models 目录，全部打包

    # 找到 models 在路径中的位置
    try:
        models_index = parts.index('models')
    except ValueError:
        return True

    # 排除 models 目录下的所有 .zip 文件（原始压缩包不打包）
    if 'models' in parts and path.suffix.lower() == '.zip' or  path.suffix.lower() == '.cfg':
        return False

    # models/模型名/子目录/... 的深度 >= 3 不打包
    depth = len(parts) - models_index

    if depth >= 4:  # models/模型名/子目录/文件 或更深
        return False
    else:  # models/模型名/文件 或更浅
        return True


def create_file_list(dist_folder, output_file='file_list.txt', is_client_only=False):
    """
    创建要打包的文件列表

    7zip 使用 @参数从文件读取列表
    每行一个文件路径（相对于 dist 父目录）
    """
    files = []

    # 遍历 dist 目录，收集所有要打包的文件
    dist_path = Path(dist_folder)
    if not dist_path.exists():
        return files, None

    # The production artifact contract requires these root-level directories
    # even when they contain no files.  Add only genuinely empty, real
    # directories: giving 7-Zip a non-empty directory path would recursively
    # bypass the per-file model exclusions below.
    for directory_name in REQUIRED_EMPTY_RELEASE_DIRECTORIES:
        directory = dist_path / directory_name
        is_junction = getattr(directory, "is_junction", None)
        if (
            directory.is_dir()
            and not directory.is_symlink()
            and not (is_junction and is_junction())
            and not any(directory.iterdir())
        ):
            files.append(os.path.relpath(directory, dist_path.parent))

    for root, dirs, filenames in os.walk(dist_path):
        # 排除不需要打包的文件夹
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.vscode', '.git')]

        for filename in filenames:
            file_path = os.path.join(root, filename)
            if should_include_file(file_path, is_client_only):
                # 计算相对于 dist 父目录的路径
                rel_path = os.path.relpath(file_path, dist_path.parent)
                files.append(rel_path)

    if not files:
        return files, None

    # 写入文件列表
    list_file = Path(output_file)
    list_file.write_text('\n'.join(files), encoding='utf-8')

    return files, list_file


def package_with_7zip(source_dir, output_zip, file_list_file):
    """使用 7zip 打包目录"""

    timeout = zip_release_timeout_seconds()
    seven_zip = find_7zip()
    if not seven_zip:
        raise FileNotFoundError(
            "找不到 7zip。请确认已安装 7-Zip。\n"
            "下载地址: https://www.7-zip.org/"
        )

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")

    # 确保输出目录存在
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 文件列表处理
    # 文件列表在当前工作目录，需要从 dist 目录访问
    dist_dir = source_path.parent
    list_file_abs = Path(file_list_file).absolute()
    list_file_rel_to_dist = os.path.relpath(list_file_abs, dist_dir)

    # 构建 7zip 命令
    # 使用 -tzip 创建 ZIP 格式（兼容性好）
    # 使用 -mx9 最大压缩
    # 使用 @file_list.txt 从文件读取要打包的文件列表
    cmd = [
        seven_zip,
        'a',                      # 添加到压缩包
        '-tzip',                  # ZIP 格式
        '-mx9',                   # 最大压缩级别
        str(output_path.absolute()),  # 输出文件（绝对路径）
        f'@{list_file_rel_to_dist}',  # 从文件读取列表（相对于 dist 目录）
    ]

    # 读取文件列表统计信息
    with open(file_list_file, 'r', encoding='utf-8') as f:
        files_count = len(f.readlines())

    print(f"\n正在打包: {source_path.name}")
    print(f"输出文件: {output_zip}")
    print(f"打包文件数: {files_count}")
    print(f"工作目录: {dist_dir.absolute()}")

    # 执行压缩（从 dist 目录运行）
    result = subprocess.run(
        cmd,
        cwd=str(dist_dir),  # 从 dist 目录运行
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=timeout,
    )

    if result.returncode != 0:
        print(f"\n错误: 7zip 执行失败")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)

    print("\n✅ 打包成功！")

    # 显示压缩包信息
    try:
        info_result = subprocess.run(
            [seven_zip, 'l', str(output_path.absolute())],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"\n警告: 读取压缩包信息超时（{_format_seconds(timeout)}s）")
        return

    if info_result.returncode == 0:
        # 解析文件数量和大小
        lines = info_result.stdout.split('\n')
        for line in lines:
            if 'files' in line.lower() or '文件夹' in line or '文件' in line:
                print(f"\n压缩包信息: {line.strip()}")
                break


def main():
    """v1 仅发行原始码，不重新归档旧 PyInstaller 产物。"""
    raise SystemExit(
        "v1 is source-only on Python 3.10-3.12; binary release packaging is retired. "
        "See docs/en/maintenance.md."
    )


if __name__ == '__main__':
    main()
