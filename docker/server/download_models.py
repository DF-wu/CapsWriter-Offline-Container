import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import ModelPaths, ServerConfig

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
DOWNLOAD_BLOCK_SIZE = 1024 * 1024


class Asset:
    def __init__(
        self,
        name: str,
        url: str,
        sha256: str,
        target_dir: Path,
        required_files: List[Path],
    ):
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.target_dir = target_dir
        self.required_files = required_files

    @property
    def archive_path(self) -> Path:
        return Path("models") / ".downloads" / self.name

    def is_ready(self) -> bool:
        return all(path.exists() for path in self.required_files)


# 本 fork 只支援兩個 ASR 模型: fun_asr_nano (低延遲) 與 qwen_asr (高精度)。
# 兩者皆為 GGUF 後端, 自帶標點 (EngineCapabilities.PUNC), 因此不需額外的 punct 模型。
# 上游 release page: https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models
SUPPORTED_MODELS = ("fun_asr_nano", "qwen_asr")

ASSETS = {
    "fun_asr_nano": [
        Asset(
            # upstream Fun-ASR-Nano-GGUF.zip (fp16 ONNX + q5_k GGUF, 2026-05-08 release)
            name="Fun-ASR-Nano-GGUF.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Fun-ASR-Nano-GGUF.zip",
            sha256="26a557923aedc44f1a3033d0a9b9c7b13cbb551f57fb9fd4b15a67bb4b57f998",
            target_dir=Path("models") / "Fun-ASR-Nano",
            required_files=[
                ModelPaths.fun_asr_nano_gguf_encoder_adaptor,
                ModelPaths.fun_asr_nano_gguf_ctc,
                ModelPaths.fun_asr_nano_gguf_llm_decode,
                ModelPaths.fun_asr_nano_gguf_token,
            ],
        )
    ],
    "qwen_asr": [
        Asset(
            # upstream Qwen3-ASR-1.7B-q5_k.zip (2026-05-08 release).
            # zip 內含 Qwen3-ASR-1.7B/ 前綴目錄, 解壓位置與 target_dir 自然對齊。
            # 若需更小體積, 改用 q4_k 版 (URL 結尾 -q4_k, sha 不同), 但 q4_k zip
            # 內沒前綴目錄, target_dir 需相應改為 Qwen3-ASR/Qwen3-ASR-1.7B。
            name="Qwen3-ASR-1.7B-q5_k.zip",
            url="https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip",
            sha256="f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f",
            target_dir=Path("models") / "Qwen3-ASR",
            required_files=[
                ModelPaths.qwen3_asr_gguf_encoder_frontend,
                ModelPaths.qwen3_asr_gguf_encoder_backend,
                ModelPaths.qwen3_asr_gguf_llm_decode,
            ],
        )
    ],
}

LLAMA_CPP_ASSETS = {
    "cpu": Asset(
        name="llama-b7798-bin-ubuntu-x64.tar.gz",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-ubuntu-x64.tar.gz",
        sha256="13e0fbbb2e1a3379ec5ed84ee9fe230f7d0c1dedbfc1b769424f0adc37f616b8",
        # 此欄位對 llama assets 是 sentinel; 實際解壓目錄請看 LLAMA_TARGET_DIRS
        target_dir=Path("core") / "server" / "engines" / "qwen_asr_gguf" / "inference" / "bin",
        required_files=[],
    ),
    "vulkan": Asset(
        name="llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="5a4ee2db7e6f9d5c0a04741f79437b433e63862d72a962466b9f28f09f7ffff9",
        # 此欄位對 llama assets 是 sentinel; 實際解壓目錄請看 LLAMA_TARGET_DIRS
        target_dir=Path("core") / "server" / "engines" / "qwen_asr_gguf" / "inference" / "bin",
        required_files=[],
    ),
}

LLAMA_TARGET_DIRS = [
    Path("core") / "server" / "engines" / "qwen_asr_gguf" / "inference" / "bin",
    Path("core") / "server" / "engines" / "fun_asr_gguf" / "inference" / "bin",
    Path("core") / "server" / "engines" / "force_aligner_gguf" / "inference" / "bin",
]

LLAMA_REQUIRED_CPU_LIBRARIES = [
    # ctypes loads the unversioned names directly.
    "libggml.so",
    "libggml-base.so",
    "libllama.so",
    "libggml-cpu.so",
    # llama.cpp Linux builds also link against these SONAME files at runtime.
    "libggml.so.0",
    "libggml-base.so.0",
]
LLAMA_REQUIRED_VULKAN_LIBRARIES = [
    "libggml-vulkan.so",
]


def _download_timeout_seconds() -> float:
    value = os.getenv("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT", "").strip()
    if not value:
        return DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT must be a number") from exc
    if timeout <= 0:
        raise ValueError("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT must be > 0")
    return timeout


def _download(url: str, destination: Path, *, timeout: float | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(f"{destination.name}.part")
    if partial_path.exists():
        partial_path.unlink()

    def _report(downloaded: int, total_size: int) -> None:
        if total_size <= 0:
            return
        percent = min(downloaded, total_size) * 100 / total_size
        print(f"\r下载中 {destination.name}: {percent:5.1f}%", end="", flush=True)

    effective_timeout = _download_timeout_seconds() if timeout is None else timeout
    try:
        with urllib.request.urlopen(url, timeout=effective_timeout) as response:
            total_header = response.headers.get("Content-Length", "")
            total_size = int(total_header) if total_header.isdigit() else 0
            downloaded = 0
            with partial_path.open("wb") as output:
                while True:
                    chunk = response.read(DOWNLOAD_BLOCK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    _report(downloaded, total_size)
        partial_path.replace(destination)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise
    print()


def _extract(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(target_dir)


def _llama_binaries_ready() -> bool:
    required_names = list(LLAMA_REQUIRED_CPU_LIBRARIES)
    preferred_backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    if preferred_backend == "vulkan":
        required_names.extend(LLAMA_REQUIRED_VULKAN_LIBRARIES)

    for target_dir in LLAMA_TARGET_DIRS:
        for required_name in required_names:
            if not (target_dir / required_name).exists():
                return False
    return True


def _extract_llama_binaries(archive: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with tarfile.open(archive, "r:gz") as tar_file:
            tar_file.extractall(temp_dir)

        shared_libraries = [
            file_path
            for file_path in temp_dir.rglob("*")
            if file_path.is_file() and ".so" in file_path.name
        ]

        if not shared_libraries:
            raise FileNotFoundError("未在 llama.cpp 压缩包中找到 Linux 共享库")

        for target_dir in LLAMA_TARGET_DIRS:
            target_dir.mkdir(parents=True, exist_ok=True)
            for library_path in shared_libraries:
                shutil.copy2(library_path, target_dir / library_path.name)

            cpu_backend = target_dir / "libggml-cpu.so"
            if not cpu_backend.exists():
                preferred_backend = None
                for candidate_name in [
                    "libggml-cpu-x64.so",
                    "libggml-cpu-haswell.so",
                    "libggml-cpu-sse42.so",
                ]:
                    candidate_path = target_dir / candidate_name
                    if candidate_path.exists():
                        preferred_backend = candidate_path
                        break

                if preferred_backend is not None:
                    shutil.copy2(preferred_backend, cpu_backend)


def _verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _prepare_llama_binaries() -> int:
    backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    if backend not in LLAMA_CPP_ASSETS:
        print(f"不支持的 llama.cpp backend: {backend}", file=sys.stderr)
        return 1

    if _llama_binaries_ready():
        print(f"llama.cpp Linux 共享库已就绪，backend={backend}，跳过下载")
        return 0

    asset = LLAMA_CPP_ASSETS[backend]
    archive_path = asset.archive_path
    if archive_path.exists():
        print(f"发现已下载 llama.cpp 压缩包，直接解压: {archive_path}")
    else:
        print(f"开始下载 {asset.name}")
        try:
            _download(asset.url, archive_path)
        except Exception as error:
            print(f"下载 llama.cpp 压缩包失败: {asset.name}: {error}", file=sys.stderr)
            return 1

    if not _verify_sha256(archive_path, asset.sha256):
        print(f"llama.cpp 压缩包校验失败: {asset.name}", file=sys.stderr)
        return 1

    try:
        _extract_llama_binaries(archive_path)
    except Exception as error:
        print(f"解压 llama.cpp Linux 共享库失败: {error}", file=sys.stderr)
        return 1

    if not _llama_binaries_ready():
        print("llama.cpp Linux 共享库解压后仍不完整", file=sys.stderr)
        return 1

    print(f"llama.cpp Linux 共享库已准备完成，backend={backend}")
    return 0


def _resolve_model_type() -> str:
    """env > ServerConfig class default.

    download_models.py 在 entrypoint.sh 內早於 start_server_docker.py 跑,
    那時 fork_server.env_config 還沒 setattr 過 ServerConfig, 所以需要
    在這獨立讀一次 env。"""
    return (
        os.environ.get("CAPSWRITER_MODEL_TYPE", ServerConfig.model_type)
        .strip()
        .lower()
    )


def main() -> int:
    model_type = _resolve_model_type()
    assets = ASSETS.get(model_type)
    if not assets:
        print(
            f"本 fork 只支援這幾個 ASR 模型: {SUPPORTED_MODELS}; 但 "
            f"CAPSWRITER_MODEL_TYPE={model_type!r}",
            file=sys.stderr,
        )
        print(
            "上游 release 仍有其他模型 (sensevoice / paraformer / 等), 但本 "
            "fork 容器化路徑不自動下載這些, 也未經測試; 如需使用請改用上游裸機部署。",
            file=sys.stderr,
        )
        return 1

    for asset in assets:
        if asset.is_ready():
            print(f"模型已就绪，跳过下载: {asset.name}")
            continue

        if asset.archive_path.exists():
            print(f"发现已下载压缩包，直接解压: {asset.archive_path}")
        else:
            print(f"开始下载 {asset.name}")
            try:
                _download(asset.url, asset.archive_path)
            except Exception as error:
                print(f"模型压缩包下载失败: {asset.name}: {error}", file=sys.stderr)
                return 1

        if not _verify_sha256(asset.archive_path, asset.sha256):
            print(f"模型压缩包校验失败: {asset.name}", file=sys.stderr)
            return 1

        print(f"开始解压 {asset.name} -> {asset.target_dir}")
        _extract(asset.archive_path, asset.target_dir)

        if not asset.is_ready():
            print(f"解压完成后仍缺少模型文件: {asset.name}", file=sys.stderr)
            return 1

        print(f"模型已准备完成: {asset.name}")

    if model_type in {"qwen_asr", "fun_asr_nano"}:
        result = _prepare_llama_binaries()
        if result != 0:
            return result

    if os.getenv("CAPSWRITER_REMOVE_MODEL_ARCHIVES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        downloads_dir = Path("models") / ".downloads"
        if downloads_dir.exists():
            shutil.rmtree(downloads_dir)
            print("已清理模型压缩包缓存")

    return 0


if __name__ == "__main__":
    sys.exit(main())
