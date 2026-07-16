# -*- mode: python ; coding: utf-8 -*-
"""
现代化 PyInstaller 打包配置
适配 PyInstaller 6.0+ 版本
"""

from importlib.util import find_spec
from os.path import basename
from pathlib import Path
from platform import system
from shutil import copy2, copytree
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# ==================== 打包配置选项 ====================

# 是否收集 CUDA provider
# - True: 包含 onnxruntime_providers_cuda.dll，支持 GPU 加速（需要在用户机器安装 CUDA 和 CUDNN）
# - False: 不包含 CUDA provider，只使用 CPU 模式（打包体积更小，兼容性更好）
INCLUDE_CUDA_PROVIDER = False

# ====================================================

if system() != 'Windows':
    raise RuntimeError('build.spec supports only Windows production artifacts')
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('build.spec production artifacts require CPython 3.12')
if sys.maxsize <= 2**32:
    raise RuntimeError('build.spec production artifacts require 64-bit Python')


# 初始化空列表
binaries = []
datas = []
hiddenimports = [
    'artifact_self_check',
    'websockets',
    'websockets.client',
    'websockets.server',
    'rich',
    'rich.console',
    'rich.markdown',
    'rich._unicode_data.unicode17-0-0',
    'keyboard',
    'pynput',
    'pyclip',
    'numpy',
    'numba',
    'soundfile',
    'sounddevice',
    'pypinyin',
    'rapidfuzz',
    'watchdog',
    'typer',
    'colorama',
    'srt',
    'sherpa_onnx',
    'onnxruntime',
    'gguf',
    'PIL',
    'PIL.Image',
    'pystray',
    'openai',
    'ollama',
    'httpx',
    'markdown',
    'tkhtmlview',
]


def require_importable(package, import_name=None):
    selected_name = import_name or package
    try:
        available = find_spec(selected_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError) as error:
        raise RuntimeError(
            f'Cannot inspect Windows build dependency {package} ({selected_name})'
        ) from error
    if not available:
        raise RuntimeError(
            f'Missing Windows build dependency {package} (import {selected_name})'
        )


def without_cuda_provider(entries):
    filtered = []
    for src, dest in entries:
        if 'providers_cuda' in basename(src).lower():
            print(f'[INFO] Excluding CUDA provider: {ascii(basename(src))}')
            continue
        filtered.append((src, dest))
    return filtered

# Sherpa/Pillow carry native/data files that static analysis cannot infer.
# These are production requirements: a missing package or failed collection
# must stop the build instead of producing a subtly incomplete artifact.
require_importable('sherpa-onnx', 'sherpa_onnx')
sherpa_datas, sherpa_binaries, sherpa_hiddenimports = collect_all('sherpa_onnx')
if not INCLUDE_CUDA_PROVIDER:
    sherpa_datas = without_cuda_provider(sherpa_datas)
    sherpa_binaries = without_cuda_provider(sherpa_binaries)
datas += sherpa_datas
binaries += sherpa_binaries
hiddenimports += sherpa_hiddenimports

require_importable('Pillow', 'PIL')
pillow_datas, pillow_binaries, pillow_hiddenimports = collect_all('PIL')
datas += pillow_datas
binaries += pillow_binaries
hiddenimports += pillow_hiddenimports

# Fork server runtime. The Windows distribution uses the universal entrypoint:
# it selects the exact upstream server lifecycle by default and the fork server
# only when the optional OpenAI-compatible API is enabled. Collect the complete
# plugin-driven package trees because FastAPI/Pydantic/Uvicorn contain dynamic
# imports that static analysis does not reliably discover across supported
# Python versions.
server_hiddenimports = list(hiddenimports)
for package in (
    'fork_server',
    'fastapi',
    'starlette',
    'uvicorn',
    'pydantic',
    'pydantic_core',
    'multipart',
    'python_multipart',
):
    require_importable(package)
    try:
        if package == 'fork_server':
            collected = collect_submodules(
                package,
                filter=lambda name: '.tests' not in name,
            )
        else:
            collected = collect_submodules(package)
    except Exception as error:
        raise RuntimeError(f'Cannot collect HTTP API package {package}') from error
    server_hiddenimports += collected

# # 对所有模块用 .py 源码而非 .pyc（猴子补丁 _get_module_collection_mode）
# import PyInstaller.building.build_main as _bm
# _bm._get_module_collection_mode = lambda md, n, na=False: _bm._ModuleCollectionMode.PY

a_1 = Analysis(
    ['start_server_universal.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=server_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['build_hook.py'],
    excludes=['IPython',
              'PySide6', 'PySide2', 'PyQt5',
              'matplotlib', 'wx',
              'funasr', 'torch',
              ],
    noarchive=True,
)

# 过滤掉从二进制依赖分析中收集的 DLL
# 这些 DLL 是 PyInstaller 在分析 DLL 依赖时自动收集的
# 我们排除从系统 CUDA 安装目录收集的 DLL（它们应该运行时从系统加载）
filtered_binaries = []
for name, src, type in a_1.binaries:
    src_lower = src.lower() if isinstance(src, str) else ''
    is_system_cuda_dll = (
        '\\nvidia gpu computing toolkit\\cuda\\' in src_lower or
        '\\nvidia\\cudnn\\' in src_lower or
        ('\\cuda\\v' in src_lower and '\\bin\\' in src_lower)
    )
    is_unwanted_onnx_dll = (
        'onnxruntime_providers_cuda.dll' in name.lower() 
    )

    if not is_system_cuda_dll and not is_unwanted_onnx_dll:
        filtered_binaries.append((name, src, type))
    else:
        reason = "system CUDA DLL" if is_system_cuda_dll else "redundant ONNX DLL"
        print(
            f"[INFO] Excluding {reason}: {ascii(name)} "
            f"(collected from {ascii(src)})"
        )
a_1.binaries = filtered_binaries

a_2 = Analysis(
    ['start_client.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['build_hook.py'],
    excludes=['IPython',
              'PySide6', 'PySide2', 'PyQt5',
              'matplotlib', 'wx',
              ],
    noarchive=True,
)

# 客户端也过滤从系统 CUDA 目录收集的 DLL（保持一致性）
filtered_binaries = []
for name, src, type in a_2.binaries:
    src_lower = src.lower() if isinstance(src, str) else ''
    is_system_cuda_dll = (
        '\\nvidia gpu computing toolkit\\cuda\\' in src_lower or
        '\\nvidia\\cudnn\\' in src_lower or
        ('\\cuda\\v' in src_lower and '\\bin\\' in src_lower)
    )
    is_unwanted_onnx_dll = (
        'onnxruntime_providers_cuda.dll' in name.lower() or
        'directml.dll' in name.lower()
    )

    if not is_system_cuda_dll and not is_unwanted_onnx_dll:
        filtered_binaries.append((name, src, type))
    else:
        reason = "system CUDA DLL" if is_system_cuda_dll else "redundant ONNX DLL"
        print(
            f"[INFO] Excluding {reason}: {ascii(name)} "
            f"(collected from {ascii(src)})"
        )
a_2.binaries = filtered_binaries


# 排除不要打包的模块（这些将作为源文件复制）
private_module = ['core', 'config_client', 'config_server', 'LLM', ]

for which in (a_1, a_2):
    filtered = []
    for name, src, type in which.pure:
        if not any(name == m or name.startswith(m + '.') for m in private_module):
            filtered.append((name, src, type))
    which.pure = filtered

# noarchive 会将私有模块也编译成 .pyc 放进 datas，排除掉以保持源码运行
for which in (a_1, a_2):
    filtered = []
    for name, src, type in which.datas:
        is_private = any(
            name.startswith(m + '/') or name.startswith(m + '\\') or name in (m + '.py', m + '.pyc')
            for m in private_module
        )
        if not is_private:
            filtered.append((name, src, type))
    which.datas = filtered


pyz_1 = PYZ(a_1.pure)
pyz_2 = PYZ(a_2.pure)


exe_1 = EXE(
    pyz_1,
    a_1.scripts,
    [],
    exclude_binaries=True,
    name='start_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\\\icon.ico'],
    # 所有第三方依赖放入 internal 目录
    contents_directory='internal',
)
exe_2 = EXE(
    pyz_2,
    a_2.scripts,
    [],
    exclude_binaries=True,
    name='start_client',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\\\icon.ico'],
    # 所有第三方依赖放入 internal 目录
    contents_directory='internal',
)

coll = COLLECT(
    exe_1,
    a_1.binaries,
    a_1.datas,

    exe_2,
    a_2.binaries,
    a_2.datas,

    strip=False,
    upx=True,
    upx_exclude=[],
    name='CapsWriter-Offline',
)


# Assemble a genuinely portable distribution.  Immutable product trees are
# copied into dist; mutable/model data starts empty.  Never follow source-tree
# symlinks or junctions and never smuggle local caches, secrets, archives, model
# blobs, logs, or non-Windows native libraries into a release.
source_root = Path(SPECPATH).resolve()
dest_root = Path(DISTPATH) / basename(coll.name)
required_files = (
    'config_client.py',
    'config_server.py',
    'hot.txt',
    'hot-server.txt',
    'hot-rule.txt',
    'readme.md',
    'README.en.md',
    'LICENSE',
)
required_folders = ('core', 'LLM', 'assets', 'docs')
mutable_folders = ('models', 'logs')

ignored_directory_names = {
    '__pycache__',
    '.ipynb_checkpoints',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.git',
    'node_modules',
    'log',
    'logs',
    'models',
}
ignored_suffixes = (
    '.pyc', '.pyo', '.bak', '.orig', '.log',
    '.zip', '.7z', '.rar', '.tar', '.tar.gz', '.tgz', '.gz', '.bz2', '.xz',
    '.whl',
    '.onnx', '.ort', '.tflite', '.gguf', '.bin', '.model', '.engine',
    '.ckpt', '.pt', '.pth', '.safetensors',
    '.key', '.pem', '.p12', '.pfx',
)


def is_link_or_junction(path):
    candidate = Path(path)
    if candidate.is_symlink():
        return True
    is_junction = getattr(candidate, 'is_junction', None)
    return bool(is_junction and is_junction())


def portable_copy_ignore(directory, names):
    ignored = []
    for name in names:
        candidate = Path(directory) / name
        lower_name = name.lower()
        if (
            lower_name in ignored_directory_names
            or lower_name.startswith('.env')
            or lower_name.endswith(ignored_suffixes)
            or '.so.' in lower_name
            or lower_name.endswith(('.so', '.dylib'))
            or is_link_or_junction(candidate)
        ):
            ignored.append(name)
    return ignored


for relative in required_files:
    source = source_root / relative
    if not source.is_file() or is_link_or_junction(source):
        raise RuntimeError(f'Missing required Windows artifact file: {relative}')
    copy2(source, dest_root / relative)

for relative in required_folders:
    source = source_root / relative
    if not source.is_dir() or is_link_or_junction(source):
        raise RuntimeError(f'Missing required Windows artifact directory: {relative}')
    copytree(
        source,
        dest_root / relative,
        copy_function=copy2,
        dirs_exist_ok=True,
        ignore=portable_copy_ignore,
    )

for relative in mutable_folders:
    destination = dest_root / relative
    if destination.exists():
        if not destination.is_dir() or is_link_or_junction(destination):
            raise RuntimeError(f'Windows artifact mutable path is not a real directory: {relative}')
        if any(destination.iterdir()):
            raise RuntimeError(f'Windows artifact mutable path must be empty: {relative}')
    else:
        destination.mkdir()
