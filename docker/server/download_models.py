import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, List

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import ModelPaths, ServerConfig

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
MODEL_BOOTSTRAP_LOCK_TIMEOUT_ENV = "CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT"
DEFAULT_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS = 1800.0
MAX_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS = 86400.0
MODEL_BOOTSTRAP_LOCK_NAME = ".capswriter-bootstrap.lock"
MODEL_BOOTSTRAP_LOCK_POLL_SECONDS = 0.1
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
MAX_LLAMA_ARCHIVE_MEMBERS = 2048
MAX_LLAMA_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024
MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MODEL_ARCHIVE_MEMBERS = 4096
MAX_MODEL_ARCHIVE_MEMBER_BYTES = 4 * 1024 * 1024 * 1024
MAX_MODEL_ARCHIVE_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024
MODEL_READY_MARKER = ".capswriter-model-ready.json"
MODEL_READY_MARKER_SCHEMA = 2
MAX_MODEL_READY_MARKER_BYTES = 64 * 1024
LLAMA_READY_MARKER = ".capswriter-llama-ready.json"
LLAMA_READY_MARKER_SCHEMA = 2
MAX_LLAMA_READY_MARKER_BYTES = 256 * 1024
_MISSING_MARKER = object()


def _secure_open_flags(*, directory: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("secure asset access requires O_NOFOLLOW")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            raise RuntimeError("secure asset access requires O_DIRECTORY")
        flags |= directory_flag
    return flags


def _directory_parts(path: Path) -> tuple[str, ...]:
    parts = []
    for part in path.parts:
        if part in {path.anchor, "", "."}:
            continue
        if part == "..":
            raise ValueError(f"路径不得包含父目录跳转: {path}")
        parts.append(part)
    return tuple(parts)


def _open_directory_nofollow(path: Path, *, create: bool = False) -> int:
    """Open every directory component without ever following a symlink."""

    flags = _secure_open_flags(directory=True)
    descriptor = os.open(path.anchor or ".", flags)
    try:
        for part in _directory_parts(path):
            if create:
                try:
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _relative_file_parts(path: Path) -> tuple[tuple[str, ...], str]:
    if path.is_absolute():
        raise ValueError(f"相对文件路径不得为绝对路径: {path}")
    parts = _directory_parts(path)
    if not parts:
        raise ValueError(f"文件路径为空: {path}")
    return parts[:-1], parts[-1]


def _open_regular_file_at(root_descriptor: int, relative_path: Path) -> int:
    """Open a regular file below one trusted directory descriptor."""

    directory_parts, filename = _relative_file_parts(relative_path)
    directory_flags = _secure_open_flags(directory=True)
    descriptor = os.dup(root_descriptor)
    try:
        for part in directory_parts:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_flags = _secure_open_flags() | getattr(os, "O_NONBLOCK", 0)
        file_descriptor = os.open(filename, file_flags, dir_fd=descriptor)
    finally:
        os.close(descriptor)

    try:
        file_stat = os.fstat(file_descriptor)
    except Exception:
        os.close(file_descriptor)
        raise
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_descriptor)
        raise ValueError(f"资产不是普通文件: {relative_path}")
    return file_descriptor


def _stable_stat_signature(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _regular_file_manifest_from_descriptor(
    descriptor: int,
) -> dict[str, int | str] | None:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        return None
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, DOWNLOAD_BLOCK_SIZE, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if _stable_stat_signature(before) != _stable_stat_signature(after):
        return None
    if offset != before.st_size:
        return None
    return {"size": before.st_size, "sha256": digest.hexdigest()}


def _regular_file_manifest_at(
    root_descriptor: int,
    relative_path: Path,
) -> dict[str, int | str] | None:
    try:
        descriptor = _open_regular_file_at(root_descriptor, relative_path)
    except (OSError, ValueError):
        return None
    try:
        return _regular_file_manifest_from_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _read_bounded_json_marker_at(
    root_descriptor: int,
    marker_name: str,
    max_bytes: int,
) -> object:
    """Read bounded JSON from one stable, non-symlinked marker inode."""

    try:
        descriptor = _open_regular_file_at(root_descriptor, Path(marker_name))
    except FileNotFoundError:
        return _MISSING_MARKER
    except (OSError, ValueError):
        return None
    try:
        try:
            before = os.fstat(descriptor)
            if (
                before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > max_bytes
            ):
                return None
            chunks = []
            remaining = max_bytes + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(DOWNLOAD_BLOCK_SIZE, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload_bytes = b"".join(chunks)
            after = os.fstat(descriptor)
        except OSError:
            return None
        if _stable_stat_signature(before) != _stable_stat_signature(after):
            return None
        if len(payload_bytes) != before.st_size or len(payload_bytes) > max_bytes:
            return None
        try:
            return json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
    finally:
        os.close(descriptor)


class Asset:
    def __init__(
        self,
        name: str,
        url: str,
        sha256: str,
        target_dir: Path,
        required_files: List[Path],
        required_sizes: List[int] | None = None,
        required_sha256: List[str] | None = None,
        archive_size: int | None = None,
    ):
        self.name = name
        self.url = url
        self.sha256 = sha256
        self.target_dir = target_dir
        self.required_files = required_files
        if required_sizes is not None and len(required_sizes) != len(required_files):
            raise ValueError("required_sizes must match required_files")
        self.required_sizes = required_sizes
        if required_sha256 is not None and len(required_sha256) != len(required_files):
            raise ValueError("required_sha256 must match required_files")
        if required_sha256 is not None:
            normalized_hashes = [digest.strip().lower() for digest in required_sha256]
            if any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in normalized_hashes
            ):
                raise ValueError("required_sha256 entries must be 64 lowercase hex digits")
            self.required_sha256 = normalized_hashes
        else:
            self.required_sha256 = None
        if archive_size is not None and archive_size <= 0:
            raise ValueError("archive_size must be > 0")
        self.archive_size = archive_size

    @property
    def archive_path(self) -> Path:
        return Path("models") / ".downloads" / self.name

    @property
    def ready_marker_path(self) -> Path:
        return self.target_dir / MODEL_READY_MARKER

    def required_relative_paths(self) -> List[Path]:
        relative_paths = []
        for required_path in self.required_files:
            try:
                relative_path = required_path.relative_to(self.target_dir)
            except ValueError as exc:
                raise ValueError(
                    f"模型必需文件不在目标目录中: {required_path}"
                ) from exc
            if not relative_path.parts or ".." in relative_path.parts:
                raise ValueError(f"模型必需文件路径不安全: {required_path}")
            relative_paths.append(relative_path)
        return relative_paths

    def artifact_manifest(
        self,
        root: Path | None = None,
    ) -> dict[str, dict[str, int | str]] | None:
        artifact_root = self.target_dir if root is None else root
        try:
            root_descriptor = _open_directory_nofollow(artifact_root)
        except (OSError, ValueError):
            return None
        try:
            return self._artifact_manifest_at(root_descriptor)
        finally:
            os.close(root_descriptor)

    def _artifact_manifest_at(
        self,
        root_descriptor: int,
    ) -> dict[str, dict[str, int | str]] | None:
        manifest: dict[str, dict[str, int | str]] = {}
        for index, relative_path in enumerate(self.required_relative_paths()):
            file_manifest = _regular_file_manifest_at(root_descriptor, relative_path)
            if file_manifest is None:
                return None
            if (
                self.required_sizes is not None
                and file_manifest["size"] != self.required_sizes[index]
            ):
                return None
            if (
                self.required_sha256 is not None
                and file_manifest["sha256"] != self.required_sha256[index]
            ):
                return None
            manifest[relative_path.as_posix()] = file_manifest
        return manifest

    def marker_payload(
        self,
        manifest: dict[str, dict[str, int | str]],
    ) -> dict[str, object]:
        return {
            "schema": MODEL_READY_MARKER_SCHEMA,
            "archive": self.name,
            "archive_sha256": self.sha256,
            "artifacts": manifest,
        }

    def is_ready_at(self, root: Path, *, allow_legacy: bool) -> bool:
        try:
            root_descriptor = _open_directory_nofollow(root)
        except (OSError, ValueError):
            return False
        try:
            manifest = self._artifact_manifest_at(root_descriptor)
            if manifest is None:
                return False
            payload = _read_bounded_json_marker_at(
                root_descriptor,
                MODEL_READY_MARKER,
                MAX_MODEL_READY_MARKER_BYTES,
            )
        finally:
            os.close(root_descriptor)

        if payload is _MISSING_MARKER:
            # Existing installations predate the marker. Preserve compatibility
            # while still rejecting missing, empty, non-regular, and symlinked
            # artifacts. Every installation created by this version is marked.
            return (
                allow_legacy
                and self.required_sizes is not None
                and self.required_sha256 is not None
            )
        return payload == self.marker_payload(manifest)

    def is_ready(self) -> bool:
        return self.is_ready_at(self.target_dir, allow_legacy=True)


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
            required_sizes=[464131509, 78185816, 444414752, 1000331],
            required_sha256=[
                "ce21db255c94a25d740485c0144efc7eec496c7845ecf02df5cb3afc4c974d27",
                "433390b8bc7b1607c09e15ef898e387770c028684a186cb80eb6a9e3a6f70f40",
                "95f1ac301ff554de352fb272a442575f177c5f83ae376a9417924ece0443f4ba",
                "1c3670431c8b35557515148ad476fc0d61e33f7deb04964bdb56e35ae12e05d1",
            ],
            archive_size=834231292,
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
            required_sizes=[24063978, 611276553, 1471800896],
            required_sha256=[
                "fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd",
                "4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3",
                "89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5",
            ],
            archive_size=1951570656,
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
        archive_size=23594688,
    ),
    "vulkan": Asset(
        name="llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        url="https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-ubuntu-vulkan-x64.tar.gz",
        sha256="5a4ee2db7e6f9d5c0a04741f79437b433e63862d72a962466b9f28f09f7ffff9",
        # 此欄位對 llama assets 是 sentinel; 實際解壓目錄請看 LLAMA_TARGET_DIRS
        target_dir=Path("core") / "server" / "engines" / "qwen_asr_gguf" / "inference" / "bin",
        required_files=[],
        archive_size=40413592,
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

# Post-extraction manifests for the official ggml-org/llama.cpp b7798 Linux
# archives. Derived on 2026-07-17 by downloading each GitHub release asset,
# verifying the archive size/SHA-256 above, resolving every in-archive SONAME
# link to regular bytes exactly as _extract_llama_binaries does, and hashing the
# final flat target directory (including the synthesized libggml-cpu.so=x64).
_LLAMA_B7798_CPU_MANIFEST: dict[str, dict[str, int | str]] = {
    "libggml-base.so": {"size": 738176, "sha256": "f4a7bc5a7dbe1a25dc9e846776c67a2f51ac8e7588ff5355432200baa9ff840e"},
    "libggml-base.so.0": {"size": 738176, "sha256": "f4a7bc5a7dbe1a25dc9e846776c67a2f51ac8e7588ff5355432200baa9ff840e"},
    "libggml-base.so.0.9.5": {"size": 738176, "sha256": "f4a7bc5a7dbe1a25dc9e846776c67a2f51ac8e7588ff5355432200baa9ff840e"},
    "libggml-cpu-alderlake.so": {"size": 975752, "sha256": "c6db8e77eb4cc454f57494100d36fe501195faccd3c8d461c54d3194f636abd9"},
    "libggml-cpu-cannonlake.so": {"size": 1104352, "sha256": "0263f11307d37385061fafa3fec1a86101cddd35ec2418678e2c84c3623a0722"},
    "libggml-cpu-cascadelake.so": {"size": 1104352, "sha256": "27d695b280e9e8413a7d12a53efe7c23657a33045bba31d89d3d7401e7457410"},
    "libggml-cpu-cooperlake.so": {"size": 1104424, "sha256": "563f3814f70d4ded67ee8d987c01a57aca2a269f1202163e883cfe271d4af340"},
    "libggml-cpu-haswell.so": {"size": 975752, "sha256": "01df77d9e28f33eb2f1188c9859f98e13026cc82891415b1b7797846fcd6ec97"},
    "libggml-cpu-icelake.so": {"size": 1104352, "sha256": "aa4063fcaefef6b01b22fc9e3cf5dd84d5de493a8b4c1bd02f8ad6e1b1085013"},
    "libggml-cpu-ivybridge.so": {"size": 929280, "sha256": "d40b33608be0f8214432d19ae535cc4cf25816fb3ba4192fe5f3c01832f24a4f"},
    "libggml-cpu-piledriver.so": {"size": 925184, "sha256": "578e75efa05c937912250cf2a220f2fb63ba54390d6969e88120001220bf139a"},
    "libggml-cpu-sandybridge.so": {"size": 915872, "sha256": "86e643556939db7e2e25f42d86dc7496021e8fb99f720ce124a55d97f60a4e9a"},
    "libggml-cpu-sapphirerapids.so": {"size": 1371680, "sha256": "d72f844f9b67141491a91a92df7c8b99938c1612a7d35624c5716be5e3195e0f"},
    "libggml-cpu-skylakex.so": {"size": 1104352, "sha256": "03de97d65f981390a9900ab22f193b79279d034fdade51944c2b5b4dec3b298e"},
    "libggml-cpu-sse42.so": {"size": 720808, "sha256": "247db23db17967ff6e1dce408f5641541caf75b804588f7ab190d920b6525776"},
    "libggml-cpu-x64.so": {"size": 716848, "sha256": "b4719931bcfa58fbb0a478da82a809a8c37e80553aea7572ec5fdeff9a1d3ad8"},
    "libggml-cpu-zen4.so": {"size": 1104424, "sha256": "7bb6b2b89cc21be27a8d038f3987c9868238698ed23262ba01c96c1e261918b2"},
    "libggml-cpu.so": {"size": 716848, "sha256": "b4719931bcfa58fbb0a478da82a809a8c37e80553aea7572ec5fdeff9a1d3ad8"},
    "libggml-rpc.so": {"size": 131392, "sha256": "1aa499332c04e3d43a1d6489d387a17aaf2898ff53708fc9a25c5ba54729f72e"},
    "libggml.so": {"size": 54800, "sha256": "29b2c60c1d8dc5582fbbfbbdb15dd5982b38f49c60fc4c22bdb61f5222b9ae2e"},
    "libggml.so.0": {"size": 54800, "sha256": "29b2c60c1d8dc5582fbbfbbdb15dd5982b38f49c60fc4c22bdb61f5222b9ae2e"},
    "libggml.so.0.9.5": {"size": 54800, "sha256": "29b2c60c1d8dc5582fbbfbbdb15dd5982b38f49c60fc4c22bdb61f5222b9ae2e"},
    "libllama.so": {"size": 3095928, "sha256": "bbe4fd0f99a5062bac8e585192047f82399988e9e24d993a19fdd113454828ec"},
    "libllama.so.0": {"size": 3095928, "sha256": "bbe4fd0f99a5062bac8e585192047f82399988e9e24d993a19fdd113454828ec"},
    "libllama.so.0.0.7798": {"size": 3095928, "sha256": "bbe4fd0f99a5062bac8e585192047f82399988e9e24d993a19fdd113454828ec"},
    "libmtmd.so": {"size": 914680, "sha256": "e0840a54d15f1ea19e66a311ab76a04478fb31fe0562553b54839864114bf703"},
    "libmtmd.so.0": {"size": 914680, "sha256": "e0840a54d15f1ea19e66a311ab76a04478fb31fe0562553b54839864114bf703"},
    "libmtmd.so.0.0.7798": {"size": 914680, "sha256": "e0840a54d15f1ea19e66a311ab76a04478fb31fe0562553b54839864114bf703"},
}
LLAMA_CPP_OFFICIAL_MANIFESTS: dict[
    str,
    dict[str, dict[str, int | str]],
] = {
    "cpu": dict(_LLAMA_B7798_CPU_MANIFEST),
    "vulkan": {
        **_LLAMA_B7798_CPU_MANIFEST,
        "libggml-vulkan.so": {
            "size": 56414152,
            "sha256": "0826ad3bfa43c30209cf155edb226dab776d98befce72a7df89f4055b61a541d",
        },
    },
}


def _download_timeout_seconds() -> float:
    value = os.getenv("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT", "").strip()
    if not value:
        return DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT must be > 0")
    if timeout <= 0:
        raise ValueError("CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT must be > 0")
    return timeout


def _bootstrap_lock_timeout_seconds() -> float:
    value = os.getenv(MODEL_BOOTSTRAP_LOCK_TIMEOUT_ENV, "").strip()
    if not value:
        return DEFAULT_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{MODEL_BOOTSTRAP_LOCK_TIMEOUT_ENV} must be a number"
        ) from exc
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or timeout > MAX_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"{MODEL_BOOTSTRAP_LOCK_TIMEOUT_ENV} must be > 0 and <= "
            f"{MAX_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS:g}"
        )
    return timeout


def _open_secure_bootstrap_lock(models_root: Path) -> int:
    """Open one same-filesystem lock without following a planted link."""

    try:
        root_stat = models_root.lstat()
    except FileNotFoundError:
        models_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        root_stat = models_root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"模型根目录必须是真实目录: {models_root}")

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("secure model bootstrap locking requires O_NOFOLLOW")
    flags = os.O_CREAT | os.O_RDWR | nofollow
    flags |= getattr(os, "O_CLOEXEC", 0)
    lock_path = models_root / MODEL_BOOTSTRAP_LOCK_NAME
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        handle_stat = os.fstat(descriptor)
        path_stat = lock_path.lstat()
        if (
            not stat.S_ISREG(handle_stat.st_mode)
            or handle_stat.st_nlink != 1
            or handle_stat.st_dev != path_stat.st_dev
            or handle_stat.st_ino != path_stat.st_ino
        ):
            raise ValueError(f"模型 bootstrap lock 不是安全的普通文件: {lock_path}")
        effective_uid = getattr(os, "geteuid", lambda: 0)()
        if effective_uid != 0 and handle_stat.st_uid != effective_uid:
            raise PermissionError(
                f"模型 bootstrap lock 不属于当前用户: {lock_path}"
            )
        if handle_stat.st_mode & 0o022:
            raise PermissionError(
                f"模型 bootstrap lock 不得允许 group/other 写入: {lock_path}"
            )
        os.fchmod(descriptor, 0o600)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _try_lock_bootstrap_file(descriptor: int) -> bool:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise


def _unlock_bootstrap_file(descriptor: int) -> None:
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _model_bootstrap_lock(
    timeout_seconds: float,
    *,
    models_root: Path = Path("models"),
) -> Iterator[None]:
    """Serialize mutations across containers sharing one model volume."""

    descriptor = _open_secure_bootstrap_lock(models_root)
    acquired = False
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if _try_lock_bootstrap_file(descriptor):
                acquired = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "等待模型 bootstrap lock 超时 "
                    f"({timeout_seconds:g}s): "
                    f"{models_root / MODEL_BOOTSTRAP_LOCK_NAME}"
                )
            time.sleep(min(MODEL_BOOTSTRAP_LOCK_POLL_SECONDS, remaining))
        yield
    finally:
        try:
            if acquired:
                _unlock_bootstrap_file(descriptor)
        finally:
            os.close(descriptor)


def _download(
    url: str,
    destination: Path,
    *,
    timeout: float | None = None,
    max_bytes: int | None = None,
    expected_bytes: int | None = None,
) -> None:
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError(f"下载目标文件名不安全: {destination}")
    parent_descriptor = _open_directory_nofollow(destination.parent, create=True)
    partial_name = f"{destination.name}.part"
    try:
        try:
            # unlinkat removes the directory entry itself and never follows a
            # stale or dangling partial symlink.
            os.unlink(partial_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        except IsADirectoryError as exc:
            raise ValueError(f"下载暂存路径不得为目录: {partial_name}") from exc

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
                if max_bytes is not None and total_size > max_bytes:
                    raise ValueError(
                        f"下载大小超过上限: {total_size} > {max_bytes} bytes"
                    )
                if expected_bytes is not None and total_size not in {0, expected_bytes}:
                    raise ValueError(
                        f"下载大小与固定资产不符: {total_size} != {expected_bytes} bytes"
                    )
                output_flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | _secure_open_flags()
                )
                output_descriptor = os.open(
                    partial_name,
                    output_flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                downloaded = 0
                with os.fdopen(output_descriptor, "wb") as output:
                    while True:
                        chunk = response.read(DOWNLOAD_BLOCK_SIZE)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if max_bytes is not None and downloaded > max_bytes:
                            raise ValueError(
                                "下载大小超过上限: "
                                f"{downloaded} > {max_bytes} bytes"
                            )
                        output.write(chunk)
                        _report(downloaded, total_size)
                if expected_bytes is not None and downloaded != expected_bytes:
                    raise ValueError(
                        "下载大小与固定资产不符: "
                        f"{downloaded} != {expected_bytes} bytes"
                    )
            os.replace(
                partial_name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except Exception:
            try:
                os.unlink(partial_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            raise
    finally:
        os.close(parent_descriptor)
    print()


def _safe_archive_path(root: Path, member_name: str) -> Path:
    if not member_name.strip():
        raise ValueError("压缩包成员路径不安全: empty name")
    if "\\" in member_name:
        raise ValueError(f"压缩包成员路径不安全: {member_name!r}")
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
        raise ValueError(f"压缩包成员路径不安全: {member_name!r}")
    target = (root / Path(*member_path.parts)).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"压缩包成员路径越界: {member_name!r}")
    return target


def _extract(archive: Path | BinaryIO, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        if len(members) > MAX_MODEL_ARCHIVE_MEMBERS:
            raise ValueError(
                "模型压缩包成员过多: "
                f"{len(members)} > {MAX_MODEL_ARCHIVE_MEMBERS}"
            )

        members_by_path: dict[tuple[str, ...], zipfile.ZipInfo] = {}
        extracted_bytes = 0
        for member in members:
            member_parts = _archive_member_parts(target_dir, member.filename)
            if not member_parts:
                if member.is_dir():
                    continue
                raise ValueError(f"压缩包成员路径不安全: {member.filename!r}")
            if member_parts in members_by_path:
                raise ValueError(f"压缩包成员路径重复: {member.filename!r}")
            if member.flag_bits & 0x1:
                raise ValueError(f"不支持加密的模型压缩包成员: {member.filename!r}")

            unix_mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode)
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"压缩包成员类型不安全: {member.filename!r}")
            if member.is_dir():
                members_by_path[member_parts] = member
                continue
            if (
                member.file_size < 0
                or member.file_size > MAX_MODEL_ARCHIVE_MEMBER_BYTES
            ):
                raise ValueError(
                    f"模型压缩包成员过大: {member.filename!r} "
                    f"({member.file_size} bytes)"
                )
            extracted_bytes += member.file_size
            if extracted_bytes > MAX_MODEL_ARCHIVE_EXTRACTED_BYTES:
                raise ValueError(
                    "模型压缩包解压大小超过上限: "
                    f"{extracted_bytes} > {MAX_MODEL_ARCHIVE_EXTRACTED_BYTES}"
                )
            members_by_path[member_parts] = member

        # Manual extraction into a private staging directory avoids ZipFile's
        # platform-dependent handling of special members and never writes into
        # the live model directory.
        for member_parts, member in members_by_path.items():
            destination = target_dir.joinpath(*member_parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member, "r") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=DOWNLOAD_BLOCK_SIZE)
            if destination.stat().st_size != member.file_size:
                raise ValueError(f"模型压缩包成员大小不符: {member.filename!r}")


def _path_lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _model_backup_path(target_dir: Path) -> Path:
    return target_dir.with_name(f".{target_dir.name}.capswriter-backup")


def _recover_model_backup(asset: Asset) -> None:
    backup_path = _model_backup_path(asset.target_dir)
    if not _path_lexists(backup_path):
        return
    if _path_lexists(asset.target_dir):
        _remove_path(backup_path)
        return
    if asset.is_ready_at(backup_path, allow_legacy=True):
        os.replace(backup_path, asset.target_dir)
        print(f"已恢复上次中断的模型目录: {asset.target_dir}")
        return
    _remove_path(backup_path)


def _write_model_ready_marker(
    asset: Asset,
    root: Path,
    manifest: dict[str, dict[str, int | str]],
) -> None:
    marker_path = root / MODEL_READY_MARKER
    payload = json.dumps(
        asset.marker_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    marker_path.write_text(f"{payload}\n", encoding="utf-8")


def _promote_model_directory(staging_dir: Path, target_dir: Path) -> None:
    backup_path = _model_backup_path(target_dir)
    if _path_lexists(backup_path):
        _remove_path(backup_path)

    had_target = _path_lexists(target_dir)
    if had_target:
        os.replace(target_dir, backup_path)
    try:
        os.replace(staging_dir, target_dir)
    except Exception:
        if had_target and _path_lexists(backup_path) and not _path_lexists(target_dir):
            os.replace(backup_path, target_dir)
        raise
    if _path_lexists(backup_path):
        _remove_path(backup_path)


def _install_model_asset(asset: Asset, archive: Path | BinaryIO) -> None:
    target_parent = asset.target_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{asset.target_dir.name}.capswriter-staging-",
            dir=target_parent,
        )
    )
    try:
        _extract(archive, staging_dir)
        manifest = asset.artifact_manifest(staging_dir)
        if manifest is None:
            raise FileNotFoundError(f"解压完成后仍缺少或损坏模型文件: {asset.name}")
        _write_model_ready_marker(asset, staging_dir, manifest)
        if not asset.is_ready_at(staging_dir, allow_legacy=False):
            raise ValueError(f"模型完成标记验证失败: {asset.name}")
        _promote_model_directory(staging_dir, asset.target_dir)
    finally:
        if _path_lexists(staging_dir):
            _remove_path(staging_dir)


def _open_archive_descriptor(archive_path: Path) -> int | None:
    """Open an archive beneath nofollow-traversed parents, or return missing."""

    try:
        parent_descriptor = _open_directory_nofollow(archive_path.parent)
    except FileNotFoundError:
        return None
    try:
        try:
            entry_stat = os.stat(
                archive_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISDIR(entry_stat.st_mode):
            raise ValueError(f"缓存压缩包不是普通文件: {archive_path}")
        if not stat.S_ISREG(entry_stat.st_mode):
            os.unlink(archive_path.name, dir_fd=parent_descriptor)
            return None
        return _open_regular_file_at(parent_descriptor, Path(archive_path.name))
    finally:
        os.close(parent_descriptor)


def _archive_descriptor_matches(asset: Asset, descriptor: int) -> bool:
    manifest = _regular_file_manifest_from_descriptor(descriptor)
    if manifest is None:
        return False
    return (
        (asset.archive_size is None or manifest["size"] == asset.archive_size)
        and manifest["sha256"] == asset.sha256
    )


def _unlink_archive_descriptor_path(
    archive_path: Path,
    descriptor: int,
) -> None:
    """Unlink only if the pathname still names the descriptor's inode."""

    parent_descriptor = _open_directory_nofollow(archive_path.parent)
    try:
        try:
            path_stat = os.stat(
                archive_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        handle_stat = os.fstat(descriptor)
        if (
            path_stat.st_dev != handle_stat.st_dev
            or path_stat.st_ino != handle_stat.st_ino
        ):
            raise RuntimeError(f"压缩包路径在校验期间被替换: {archive_path}")
        os.unlink(archive_path.name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


@contextmanager
def _ensure_verified_archive(
    asset: Asset,
) -> Iterator[tuple[str, BinaryIO]]:
    """Yield source and the same verified archive inode used for extraction."""

    archive_path = asset.archive_path
    descriptor = _open_archive_descriptor(archive_path)
    if descriptor is not None:
        if _archive_descriptor_matches(asset, descriptor):
            with os.fdopen(descriptor, "rb") as archive_file:
                archive_file.seek(0)
                yield "cached", archive_file
            return
        try:
            _unlink_archive_descriptor_path(archive_path, descriptor)
        finally:
            os.close(descriptor)
        print(f"缓存压缩包校验失败，已删除并重新下载: {archive_path}")

    print(f"开始下载 {asset.name}")
    _download(
        asset.url,
        archive_path,
        max_bytes=asset.archive_size,
        expected_bytes=asset.archive_size,
    )
    descriptor = _open_archive_descriptor(archive_path)
    if descriptor is None:
        raise OSError(f"下载后找不到压缩包: {archive_path}")
    if not _archive_descriptor_matches(asset, descriptor):
        try:
            _unlink_archive_descriptor_path(archive_path, descriptor)
        finally:
            os.close(descriptor)
        raise ValueError(f"下载的压缩包校验失败，已删除: {asset.name}")
    with os.fdopen(descriptor, "rb") as archive_file:
        archive_file.seek(0)
        yield "downloaded", archive_file


def _llama_required_names(backend: str) -> List[str]:
    required_names = list(LLAMA_REQUIRED_CPU_LIBRARIES)
    if backend == "vulkan":
        required_names.extend(LLAMA_REQUIRED_VULKAN_LIBRARIES)
    return required_names


def _llama_library_manifest(
    root: Path,
) -> dict[str, dict[str, int | str]] | None:
    try:
        root_descriptor = _open_directory_nofollow(root)
    except (OSError, ValueError):
        return None
    try:
        return _llama_library_manifest_at(root_descriptor)
    finally:
        os.close(root_descriptor)


def _llama_library_manifest_at(
    root_descriptor: int,
) -> dict[str, dict[str, int | str]] | None:
    try:
        entry_names = sorted(os.listdir(root_descriptor))
    except OSError:
        return None

    manifest: dict[str, dict[str, int | str]] = {}
    for entry_name in entry_names:
        if ".so" not in entry_name:
            continue
        library_manifest = _regular_file_manifest_at(
            root_descriptor,
            Path(entry_name),
        )
        if library_manifest is None:
            return None
        manifest[entry_name] = library_manifest
    return manifest or None


def _llama_marker_payload(
    asset: Asset,
    backend: str,
    manifest: dict[str, dict[str, int | str]],
) -> dict[str, object]:
    return {
        "schema": LLAMA_READY_MARKER_SCHEMA,
        "backend": backend,
        "archive": asset.name,
        "archive_sha256": asset.sha256,
        "libraries": manifest,
    }


def _write_llama_ready_marker(
    asset: Asset,
    backend: str,
    root: Path,
    manifest: dict[str, dict[str, int | str]],
) -> None:
    payload = json.dumps(
        _llama_marker_payload(asset, backend, manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    (root / LLAMA_READY_MARKER).write_text(f"{payload}\n", encoding="utf-8")


def _llama_target_ready(target_dir: Path, asset: Asset, backend: str) -> bool:
    expected_manifest = LLAMA_CPP_OFFICIAL_MANIFESTS.get(backend)
    if expected_manifest is None:
        return False
    try:
        root_descriptor = _open_directory_nofollow(target_dir)
    except (OSError, ValueError):
        return False
    try:
        manifest = _llama_library_manifest_at(root_descriptor)
        if manifest != expected_manifest:
            return False
        if any(name not in manifest for name in _llama_required_names(backend)):
            return False
        payload = _read_bounded_json_marker_at(
            root_descriptor,
            LLAMA_READY_MARKER,
            MAX_LLAMA_READY_MARKER_BYTES,
        )
    finally:
        os.close(root_descriptor)

    if payload is _MISSING_MARKER:
        # Unlike model artifacts, runtime libraries are small enough to rebuild;
        # never trust a legacy directory that lacks an authenticated-source marker.
        return False
    return payload == _llama_marker_payload(asset, backend, expected_manifest)


def _llama_backup_path(target_dir: Path) -> Path:
    return target_dir.with_name(f".{target_dir.name}.capswriter-llama-backup")


def _recover_llama_backup(target_dir: Path) -> None:
    backup_path = _llama_backup_path(target_dir)
    if not _path_lexists(backup_path):
        return
    if _path_lexists(target_dir):
        _remove_path(backup_path)
    else:
        os.replace(backup_path, target_dir)


def _llama_binaries_ready() -> bool:
    backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    asset = LLAMA_CPP_ASSETS.get(backend)
    if asset is None:
        return False

    for target_dir in LLAMA_TARGET_DIRS:
        if not _llama_target_ready(target_dir, asset, backend):
            return False
    return bool(LLAMA_TARGET_DIRS)


def _archive_member_parts(root: Path, member_name: str) -> tuple[str, ...]:
    """Return a validated, normalized archive path without filesystem links."""

    _safe_archive_path(root, member_name)
    return PurePosixPath(member_name).parts


def _archive_link_target_parts(
    member: tarfile.TarInfo,
    member_parts: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve a tar link lexically and reject targets outside the archive."""

    link_name = member.linkname
    if not link_name.strip() or "\\" in link_name:
        raise ValueError(f"压缩包链接目标不安全: {member.name!r} -> {link_name!r}")

    link_path = PurePosixPath(link_name)
    if link_path.is_absolute():
        raise ValueError(f"压缩包链接目标不安全: {member.name!r} -> {link_name!r}")

    # POSIX symlink targets are relative to the link's parent. Tar hard-link
    # targets are archive-root relative.
    parts = list(member_parts[:-1] if member.issym() else ())
    for part in link_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(
                    f"压缩包链接目标越界: {member.name!r} -> {link_name!r}"
                )
            parts.pop()
            continue
        parts.append(part)

    if not parts:
        raise ValueError(f"压缩包链接目标不安全: {member.name!r} -> {link_name!r}")
    return tuple(parts)


def _copy_regular_file(source: Path, destination: Path) -> None:
    """Atomically replace destination without following a pre-existing link."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=DOWNLOAD_BLOCK_SIZE)
            os.fchmod(output.fileno(), 0o755)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _populate_llama_target(
    target_dir: Path,
    shared_libraries: List[Path],
) -> None:
    for library_path in shared_libraries:
        _copy_regular_file(library_path, target_dir / library_path.name)

    cpu_backend = target_dir / "libggml-cpu.so"
    if cpu_backend.exists():
        return
    for candidate_name in [
        "libggml-cpu-x64.so",
        "libggml-cpu-haswell.so",
        "libggml-cpu-sse42.so",
    ]:
        candidate_path = target_dir / candidate_name
        if candidate_path.exists():
            _copy_regular_file(candidate_path, cpu_backend)
            return


def _promote_llama_staging_dirs(staging_dirs: dict[Path, Path]) -> None:
    backups: dict[Path, Path] = {}
    promoted_targets: List[Path] = []
    try:
        for target_dir in staging_dirs:
            backup_path = _llama_backup_path(target_dir)
            if _path_lexists(backup_path):
                _remove_path(backup_path)
            if _path_lexists(target_dir):
                os.replace(target_dir, backup_path)
                backups[target_dir] = backup_path

        for target_dir, staging_dir in staging_dirs.items():
            os.replace(staging_dir, target_dir)
            promoted_targets.append(target_dir)
    except Exception:
        for target_dir in reversed(promoted_targets):
            if _path_lexists(target_dir):
                _remove_path(target_dir)
        for target_dir, backup_path in backups.items():
            if _path_lexists(backup_path) and not _path_lexists(target_dir):
                os.replace(backup_path, target_dir)
        raise

    for backup_path in backups.values():
        if _path_lexists(backup_path):
            _remove_path(backup_path)


def _extract_llama_binaries(
    archive: Path | BinaryIO,
    *,
    asset: Asset | None = None,
    backend: str | None = None,
) -> None:
    selected_backend = (
        os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
        if backend is None
        else backend
    )
    selected_asset = LLAMA_CPP_ASSETS.get(selected_backend) if asset is None else asset
    if selected_asset is None:
        raise ValueError(f"不支持的 llama.cpp backend: {selected_backend}")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        if isinstance(archive, (str, os.PathLike)):
            tar_context = tarfile.open(archive, "r:gz")
        else:
            archive.seek(0)
            tar_context = tarfile.open(fileobj=archive, mode="r:gz")
        with tar_context as tar_file:
            members = tar_file.getmembers()
            if len(members) > MAX_LLAMA_ARCHIVE_MEMBERS:
                raise ValueError(
                    "llama.cpp 压缩包成员过多: "
                    f"{len(members)} > {MAX_LLAMA_ARCHIVE_MEMBERS}"
                )

            members_by_path: dict[tuple[str, ...], tarfile.TarInfo] = {}
            regular_file_bytes = 0
            for member in members:
                member_parts = _archive_member_parts(temp_dir, member.name)
                if not member_parts:
                    if member.isdir():
                        continue
                    raise ValueError(f"压缩包成员路径不安全: {member.name!r}")
                if member_parts in members_by_path:
                    raise ValueError(f"压缩包成员路径重复: {member.name!r}")
                if not (
                    member.isfile()
                    or member.isdir()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError(f"压缩包成员类型不安全: {member.name!r}")
                if member.isfile():
                    if member.size < 0 or member.size > MAX_LLAMA_ARCHIVE_MEMBER_BYTES:
                        raise ValueError(
                            f"压缩包成员过大: {member.name!r} ({member.size} bytes)"
                        )
                    regular_file_bytes += member.size
                    if regular_file_bytes > MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES:
                        raise ValueError(
                            "llama.cpp 压缩包解压大小超过上限: "
                            f"{regular_file_bytes} > {MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES}"
                        )
                members_by_path[member_parts] = member

            resolved_links: dict[tuple[str, ...], tuple[str, ...]] = {}

            def resolve_regular_target(
                member_parts: tuple[str, ...],
                chain: frozenset[tuple[str, ...]] = frozenset(),
            ) -> tuple[str, ...]:
                if member_parts in resolved_links:
                    return resolved_links[member_parts]
                if member_parts in chain:
                    raise ValueError(f"压缩包链接形成循环: {'/'.join(member_parts)!r}")

                target_member = members_by_path.get(member_parts)
                if target_member is None:
                    raise ValueError(
                        f"压缩包链接目标不存在: {'/'.join(member_parts)!r}"
                    )
                if target_member.isfile():
                    return member_parts
                if not (target_member.issym() or target_member.islnk()):
                    raise ValueError(
                        f"压缩包链接目标不是普通文件: {'/'.join(member_parts)!r}"
                    )

                target_parts = _archive_link_target_parts(target_member, member_parts)
                resolved = resolve_regular_target(target_parts, chain | {member_parts})
                resolved_links[member_parts] = resolved
                return resolved

            materialized_bytes = regular_file_bytes
            for member_parts, member in members_by_path.items():
                if not (member.issym() or member.islnk()):
                    continue
                target_parts = _archive_link_target_parts(member, member_parts)
                resolved_parts = resolve_regular_target(target_parts, {member_parts})
                resolved_links[member_parts] = resolved_parts
                materialized_bytes += members_by_path[resolved_parts].size
                if materialized_bytes > MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES:
                    raise ValueError(
                        "llama.cpp 压缩包解压大小超过上限: "
                        f"{materialized_bytes} > {MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES}"
                    )

            # Extract regular files ourselves into a fresh 0700 directory.
            # Links are never created on disk; validated aliases are copied as
            # regular files after their complete chain resolves to an archive
            # regular file. This prevents tar extraction from following a host
            # filesystem symlink while retaining release archive SONAME aliases.
            for member_parts, member in members_by_path.items():
                destination = temp_dir.joinpath(*member_parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    raise ValueError(f"无法读取压缩包成员: {member.name!r}")
                with extracted, destination.open("xb") as output:
                    shutil.copyfileobj(extracted, output, length=DOWNLOAD_BLOCK_SIZE)
                if destination.stat().st_size != member.size:
                    raise ValueError(f"压缩包成员大小不符: {member.name!r}")

            for member_parts, resolved_parts in resolved_links.items():
                source = temp_dir.joinpath(*resolved_parts)
                destination = temp_dir.joinpath(*member_parts)
                if not source.is_file() or source.is_symlink():
                    raise ValueError(
                        f"压缩包链接目标不是普通文件: {'/'.join(resolved_parts)!r}"
                    )
                _copy_regular_file(source, destination)

        shared_libraries = [
            file_path
            for file_path in temp_dir.rglob("*")
            if file_path.is_file() and ".so" in file_path.name
        ]

        if not shared_libraries:
            raise FileNotFoundError("未在 llama.cpp 压缩包中找到 Linux 共享库")

        staging_dirs: dict[Path, Path] = {}
        try:
            for target_dir in LLAMA_TARGET_DIRS:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                _recover_llama_backup(target_dir)
                staging_dir = Path(
                    tempfile.mkdtemp(
                        prefix=f".{target_dir.name}.capswriter-llama-staging-",
                        dir=target_dir.parent,
                    )
                )
                staging_dirs[target_dir] = staging_dir
                _populate_llama_target(staging_dir, shared_libraries)
                manifest = _llama_library_manifest(staging_dir)
                if manifest is None:
                    raise FileNotFoundError(
                        f"llama.cpp 共享库 staging 目录为空: {target_dir}"
                    )
                _write_llama_ready_marker(
                    selected_asset,
                    selected_backend,
                    staging_dir,
                    manifest,
                )
                if not _llama_target_ready(
                    staging_dir,
                    selected_asset,
                    selected_backend,
                ):
                    raise ValueError(f"llama.cpp staging 标记验证失败: {target_dir}")

            _promote_llama_staging_dirs(staging_dirs)
        finally:
            for staging_dir in staging_dirs.values():
                if _path_lexists(staging_dir):
                    _remove_path(staging_dir)


def _prepare_llama_binaries() -> int:
    backend = os.getenv("CAPSWRITER_LLAMA_BACKEND", "cpu").strip().lower()
    if backend not in LLAMA_CPP_ASSETS:
        print(f"不支持的 llama.cpp backend: {backend}", file=sys.stderr)
        return 1

    for target_dir in LLAMA_TARGET_DIRS:
        _recover_llama_backup(target_dir)
    if _llama_binaries_ready():
        print(f"llama.cpp Linux 共享库已就绪，backend={backend}，跳过下载")
        return 0

    asset = LLAMA_CPP_ASSETS[backend]
    try:
        with _ensure_verified_archive(asset) as (archive_source, archive_file):
            if archive_source == "cached":
                print(
                    "发现已验证 llama.cpp 压缩包，直接解压: "
                    f"{asset.archive_path}"
                )
            try:
                _extract_llama_binaries(
                    archive_file,
                    asset=asset,
                    backend=backend,
                )
            except Exception as error:
                print(f"解压 llama.cpp Linux 共享库失败: {error}", file=sys.stderr)
                return 1
    except Exception as error:
        print(f"下载 llama.cpp 压缩包失败: {asset.name}: {error}", file=sys.stderr)
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


def _runtime_assets_ready(model_type: str, assets: List[Asset]) -> bool:
    if not all(asset.is_ready() for asset in assets):
        return False
    if model_type in {"qwen_asr", "fun_asr_nano"}:
        return _llama_binaries_ready()
    return True


def _remove_model_archives_requested() -> bool:
    return os.getenv("CAPSWRITER_REMOVE_MODEL_ARCHIVES", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _remove_model_archive_cache() -> None:
    downloads_dir = Path("models") / ".downloads"
    if _path_lexists(downloads_dir):
        _remove_path(downloads_dir)
        print("已清理模型压缩包缓存")


def _prepare_runtime_assets_locked(model_type: str, assets: List[Asset]) -> int:
    """Run every mutating recovery/download/promotion while holding the lock."""

    for asset in assets:
        _recover_model_backup(asset)
        if asset.is_ready():
            print(f"模型已就绪，跳过下载: {asset.name}")
            continue

        try:
            with _ensure_verified_archive(asset) as (archive_source, archive_file):
                if archive_source == "cached":
                    print(f"发现已验证压缩包，直接解压: {asset.archive_path}")

                print(f"开始解压 {asset.name} -> {asset.target_dir}")
                try:
                    _install_model_asset(asset, archive_file)
                except Exception as error:
                    print(
                        f"模型解压或安装失败: {asset.name}: {error}",
                        file=sys.stderr,
                    )
                    return 1
        except Exception as error:
            print(f"模型压缩包下载失败: {asset.name}: {error}", file=sys.stderr)
            return 1

        if not asset.is_ready():
            print(f"解压完成后仍缺少或损坏模型文件: {asset.name}", file=sys.stderr)
            return 1

        print(f"模型已准备完成: {asset.name}")

    if model_type in {"qwen_asr", "fun_asr_nano"}:
        result = _prepare_llama_binaries()
        if result != 0:
            return result

    if _remove_model_archives_requested():
        _remove_model_archive_cache()

    return 0


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

    try:
        lock_timeout = _bootstrap_lock_timeout_seconds()
    except ValueError as error:
        print(f"模型 bootstrap lock 配置无效: {error}", file=sys.stderr)
        return 1

    remove_archives = _remove_model_archives_requested()
    # A fully warm runtime performs no filesystem mutation and never creates a
    # lock file. This keeps repeated probes and read-only warm deployments viable.
    if not remove_archives and _runtime_assets_ready(model_type, assets):
        print(f"模型与 llama.cpp runtime 已就绪，跳过 bootstrap: {model_type}")
        return 0

    try:
        with _model_bootstrap_lock(lock_timeout):
            # Another container may have completed bootstrap while this process
            # waited. Re-read every marker under the lock before mutating state.
            if _runtime_assets_ready(model_type, assets):
                print(f"模型与 llama.cpp runtime 已由其他进程准备完成: {model_type}")
                if remove_archives:
                    _remove_model_archive_cache()
                return 0
            return _prepare_runtime_assets_locked(model_type, assets)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"无法取得安全的模型 bootstrap lock: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
