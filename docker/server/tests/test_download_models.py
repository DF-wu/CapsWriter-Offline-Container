# coding: utf-8

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import download_models  # noqa: E402


class FakeDownloadResponse:
    def __init__(self, *chunks: bytes, fail_after_chunks: int | None = None):
        self._chunks = list(chunks)
        self._index = 0
        self._fail_after_chunks = fail_after_chunks
        self.headers = {"Content-Length": str(sum(len(chunk) for chunk in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def read(self, _size: int) -> bytes:
        if self._fail_after_chunks is not None and self._index >= self._fail_after_chunks:
            raise TimeoutError("timed out")
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class DownloadModelsTest(unittest.TestCase):
    @staticmethod
    def _manifest_for_contents(
        contents: dict[str, bytes],
    ) -> dict[str, dict[str, int | str]]:
        return {
            name: {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(contents.items())
        }

    def _write_libraries(self, target_dir: Path, names: list[str]) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (target_dir / name).write_bytes(f"library:{name}".encode())

    def _write_llama_marker(self, target_dir: Path, backend: str) -> None:
        manifest = download_models._llama_library_manifest(target_dir)
        self.assertIsNotNone(manifest)
        download_models._write_llama_ready_marker(
            download_models.LLAMA_CPP_ASSETS[backend],
            backend,
            target_dir,
            manifest or {},
        )

    def test_resolve_model_type_normalizes_env_value(self) -> None:
        with patch.dict(
            os.environ,
            {"CAPSWRITER_MODEL_TYPE": " Fun_ASR_Nano "},
        ):
            self.assertEqual(download_models._resolve_model_type(), "fun_asr_nano")

    def test_unsupported_model_error_reports_resolved_env_value(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {"CAPSWRITER_MODEL_TYPE": "sensevoice"}):
            with redirect_stderr(stderr):
                code = download_models.main()

        self.assertEqual(code, 1)
        message = stderr.getvalue()
        self.assertIn("CAPSWRITER_MODEL_TYPE='sensevoice'", message)
        self.assertIn("qwen_asr", message)
        self.assertIn("fun_asr_nano", message)

    def test_download_timeout_defaults_to_bounded_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_models._download_timeout_seconds(),
                download_models.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
            )

    def test_download_timeout_rejects_invalid_values(self) -> None:
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT": value}):
                    with self.assertRaisesRegex(ValueError, "must be > 0"):
                        download_models._download_timeout_seconds()

    def test_bootstrap_lock_timeout_is_bounded_and_validated(self) -> None:
        env_name = download_models.MODEL_BOOTSTRAP_LOCK_TIMEOUT_ENV
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_models._bootstrap_lock_timeout_seconds(),
                download_models.DEFAULT_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS,
            )
        with patch.dict(os.environ, {env_name: "12.5"}, clear=True):
            self.assertEqual(
                download_models._bootstrap_lock_timeout_seconds(),
                12.5,
            )
        for value in ("0", "nan", "inf", "86401"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {env_name: value},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "must be > 0"):
                    download_models._bootstrap_lock_timeout_seconds()

    @unittest.skipUnless(
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "symlink"),
        "secure advisory lock requires POSIX O_NOFOLLOW",
    )
    def test_bootstrap_lock_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            models_root.mkdir()
            victim = root / "victim"
            victim.write_text("unchanged", encoding="utf-8")
            (models_root / download_models.MODEL_BOOTSTRAP_LOCK_NAME).symlink_to(
                victim
            )

            with self.assertRaises(OSError):
                with download_models._model_bootstrap_lock(
                    1.0,
                    models_root=models_root,
                ):
                    self.fail("symlinked lock must never be acquired")

            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged")

    @unittest.skipUnless(
        hasattr(os, "O_NOFOLLOW"),
        "secure advisory lock requires POSIX O_NOFOLLOW",
    )
    def test_bootstrap_lock_wait_has_one_absolute_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp) / "models"
            with (
                patch.object(
                    download_models,
                    "_try_lock_bootstrap_file",
                    return_value=False,
                ),
                patch.object(
                    download_models.time,
                    "monotonic",
                    side_effect=[10.0, 12.0],
                ),
                self.assertRaisesRegex(TimeoutError, "bootstrap lock"),
            ):
                with download_models._model_bootstrap_lock(
                    1.0,
                    models_root=models_root,
                ):
                    self.fail("contended lock must time out")

    def test_warm_main_is_read_only_and_does_not_create_lock(self) -> None:
        asset = MagicMock()
        asset.is_ready.return_value = True
        with (
            patch.object(download_models, "ASSETS", {"qwen_asr": [asset]}),
            patch.object(download_models, "_llama_binaries_ready", return_value=True),
            patch.object(download_models, "_model_bootstrap_lock") as lock,
            patch.dict(
                os.environ,
                {
                    "CAPSWRITER_MODEL_TYPE": "qwen_asr",
                    "CAPSWRITER_LLAMA_BACKEND": "cpu",
                },
                clear=True,
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = download_models.main()

        self.assertEqual(code, 0)
        lock.assert_not_called()

    def test_main_rechecks_warm_state_after_acquiring_lock(self) -> None:
        asset = MagicMock()
        asset.is_ready.side_effect = [False, True]
        with (
            patch.object(download_models, "ASSETS", {"qwen_asr": [asset]}),
            patch.object(download_models, "_llama_binaries_ready", return_value=True),
            patch.object(
                download_models,
                "_model_bootstrap_lock",
                return_value=nullcontext(),
            ) as lock,
            patch.object(download_models, "_prepare_runtime_assets_locked") as prepare,
            patch.dict(
                os.environ,
                {
                    "CAPSWRITER_MODEL_TYPE": "qwen_asr",
                    "CAPSWRITER_LLAMA_BACKEND": "cpu",
                },
                clear=True,
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = download_models.main()

        self.assertEqual(code, 0)
        lock.assert_called_once_with(
            download_models.DEFAULT_MODEL_BOOTSTRAP_LOCK_TIMEOUT_SECONDS
        )
        prepare.assert_not_called()

    def test_download_streams_with_configured_timeout_and_atomic_replace(self) -> None:
        observed = {}

        def fake_urlopen(url, *, timeout):
            observed["url"] = url
            observed["timeout"] = timeout
            return FakeDownloadResponse(b"abc", b"def")

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "models" / ".downloads" / "asset.zip"
            with (
                patch.object(download_models.urllib.request, "urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {"CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT": "7.5"}),
                redirect_stdout(io.StringIO()),
            ):
                download_models._download("https://example.test/asset.zip", destination)

            self.assertEqual(observed, {"url": "https://example.test/asset.zip", "timeout": 7.5})
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(destination.with_name("asset.zip.part").exists())

    def test_download_enforces_exact_streamed_size_and_cleans_partial(self) -> None:
        def fake_urlopen(_url, *, timeout):
            self.assertEqual(timeout, download_models.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
            response = FakeDownloadResponse(b"abc", b"def")
            response.headers = {}
            return response

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "asset.zip"
            with (
                patch.object(
                    download_models.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "下载大小超过上限"),
            ):
                download_models._download(
                    "https://example.test/asset.zip",
                    destination,
                    max_bytes=5,
                    expected_bytes=5,
                )

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("asset.zip.part").exists())

    def test_download_removes_partial_file_on_failure(self) -> None:
        def fake_urlopen(_url, *, timeout):
            self.assertEqual(timeout, download_models.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
            return FakeDownloadResponse(b"partial", fail_after_chunks=1)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "asset.zip"
            with (
                patch.object(download_models.urllib.request, "urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {}, clear=True),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(TimeoutError, "timed out"),
            ):
                download_models._download("https://example.test/asset.zip", destination)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("asset.zip.part").exists())

    @unittest.skipUnless(
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "symlink"),
        "secure download staging requires POSIX O_NOFOLLOW",
    )
    def test_download_never_follows_partial_or_parent_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "downloads"
            downloads.mkdir()
            destination = downloads / "asset.zip"
            outside_file = root / "outside-created"
            destination.with_name("asset.zip.part").symlink_to(outside_file)

            with (
                patch.object(
                    download_models.urllib.request,
                    "urlopen",
                    return_value=FakeDownloadResponse(b"verified"),
                ),
                redirect_stdout(io.StringIO()),
            ):
                download_models._download(
                    "https://example.test/asset.zip",
                    destination,
                )

            self.assertFalse(outside_file.exists())
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"verified")

            models = root / "models"
            outside_directory = root / "outside-directory"
            models.mkdir()
            outside_directory.mkdir()
            (models / ".downloads").symlink_to(
                outside_directory,
                target_is_directory=True,
            )
            urlopen = MagicMock(return_value=FakeDownloadResponse(b"never"))
            with (
                patch.object(download_models.urllib.request, "urlopen", urlopen),
                self.assertRaises(OSError),
            ):
                download_models._download(
                    "https://example.test/asset.zip",
                    models / ".downloads" / "asset.zip",
                )

            urlopen.assert_not_called()
            self.assertEqual(list(outside_directory.iterdir()), [])

    def test_extract_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            target_dir = root / "target"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../escaped.bin", b"bad")

            with self.assertRaisesRegex(ValueError, "不安全"):
                download_models._extract(archive, target_dir)

            self.assertFalse((root / "escaped.bin").exists())

    def test_asset_readiness_rejects_empty_symlink_and_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            target_dir.mkdir()
            artifact = target_dir / "artifact.bin"
            asset = download_models.Asset(
                name="model.zip",
                url="https://example.test/model.zip",
                sha256="a" * 64,
                target_dir=target_dir,
                required_files=[artifact],
                required_sizes=[len(b"model")],
                required_sha256=[hashlib.sha256(b"model").hexdigest()],
            )

            artifact.write_bytes(b"")
            self.assertFalse(asset.is_ready())

            artifact.unlink()
            outside = root / "outside.bin"
            outside.write_bytes(b"model")
            artifact.symlink_to(outside)
            self.assertFalse(asset.is_ready())

            artifact.unlink()
            artifact.write_bytes(b"x")
            self.assertFalse(asset.is_ready(), "legacy artifacts require pinned sizes")

            artifact.write_bytes(b"model")
            self.assertTrue(asset.is_ready(), "legacy models require pinned size and digest")

            artifact.write_bytes(b"modem")
            self.assertFalse(asset.is_ready(), "same-size model corruption must be rejected")

            artifact.write_bytes(b"model")

            manifest = asset.artifact_manifest()
            self.assertIsNotNone(manifest)
            download_models._write_model_ready_marker(asset, target_dir, manifest or {})
            self.assertTrue(asset.is_ready())

            artifact.write_bytes(b"changed-size")
            self.assertFalse(asset.is_ready())

    @unittest.skipUnless(
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "symlink"),
        "secure artifact traversal requires POSIX O_NOFOLLOW",
    )
    def test_asset_readiness_rejects_symlinked_directory_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            outside_dir = root / "outside"
            target_dir.mkdir()
            outside_dir.mkdir()
            payload = b"official-model"
            (outside_dir / "artifact.bin").write_bytes(payload)
            (target_dir / "nested").symlink_to(
                outside_dir,
                target_is_directory=True,
            )
            artifact = target_dir / "nested" / "artifact.bin"
            asset = download_models.Asset(
                name="model.zip",
                url="https://example.test/model.zip",
                sha256="a" * 64,
                target_dir=target_dir,
                required_files=[artifact],
                required_sizes=[len(payload)],
                required_sha256=[hashlib.sha256(payload).hexdigest()],
            )

            self.assertFalse(asset.is_ready())

    def test_marker_swap_is_rejected_after_one_bounded_inode_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            target_dir.mkdir()
            artifact = target_dir / "artifact.bin"
            artifact.write_bytes(b"model")
            asset = download_models.Asset(
                name="model.zip",
                url="https://example.test/model.zip",
                sha256="a" * 64,
                target_dir=target_dir,
                required_files=[artifact],
                required_sizes=[len(b"model")],
                required_sha256=[hashlib.sha256(b"model").hexdigest()],
            )
            manifest = asset.artifact_manifest()
            self.assertIsNotNone(manifest)
            download_models._write_model_ready_marker(
                asset,
                target_dir,
                manifest or {},
            )
            replacement = target_dir / "replacement-marker"
            replacement.write_bytes(
                b"x" * (download_models.MAX_MODEL_READY_MARKER_BYTES + 1)
            )
            marker_path = target_dir / download_models.MODEL_READY_MARKER
            real_read = os.read
            read_sizes: list[int] = []
            swapped = False

            def swap_then_read(descriptor: int, size: int) -> bytes:
                nonlocal swapped
                read_sizes.append(size)
                if not swapped:
                    swapped = True
                    os.replace(replacement, marker_path)
                return real_read(descriptor, size)

            with patch.object(download_models.os, "read", side_effect=swap_then_read):
                self.assertFalse(asset.is_ready())

            self.assertTrue(swapped)
            self.assertTrue(read_sizes)
            self.assertLessEqual(
                max(read_sizes),
                download_models.MAX_MODEL_READY_MARKER_BYTES + 1,
            )
            self.assertGreater(
                marker_path.stat().st_size,
                download_models.MAX_MODEL_READY_MARKER_BYTES,
            )

    def test_extract_rejects_zip_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            target_dir = root / "target"
            with zipfile.ZipFile(archive, "w") as zip_file:
                info = zipfile.ZipInfo("model/link.bin")
                info.create_system = 3
                info.external_attr = (download_models.stat.S_IFLNK | 0o777) << 16
                zip_file.writestr(info, "../../outside.bin")

            with self.assertRaisesRegex(ValueError, "成员类型不安全"):
                download_models._extract(archive, target_dir)

            self.assertFalse((root / "outside.bin").exists())

    def test_extract_model_archive_enforces_member_and_size_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("one.bin", b"a")
                zip_file.writestr("two.bin", b"b")

            with (
                patch.object(download_models, "MAX_MODEL_ARCHIVE_MEMBERS", 1),
                self.assertRaisesRegex(ValueError, "成员过多"),
            ):
                download_models._extract(archive, root / "member-target")

            with (
                patch.object(download_models, "MAX_MODEL_ARCHIVE_MEMBERS", 2),
                patch.object(download_models, "MAX_MODEL_ARCHIVE_EXTRACTED_BYTES", 1),
                self.assertRaisesRegex(ValueError, "解压大小超过上限"),
            ):
                download_models._extract(archive, root / "size-target")

    def test_model_install_stages_validates_marks_and_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            target_dir.mkdir()
            artifact = target_dir / "artifact.bin"
            artifact.write_bytes(b"old-model")
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("artifact.bin", b"new-model")
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            asset = download_models.Asset(
                name=archive.as_posix(),
                url="https://example.test/model.zip",
                sha256=archive_sha,
                target_dir=target_dir,
                required_files=[artifact],
            )

            download_models._install_model_asset(asset, archive)

            self.assertEqual(artifact.read_bytes(), b"new-model")
            self.assertTrue(asset.ready_marker_path.is_file())
            self.assertTrue(asset.is_ready())
            self.assertFalse(download_models._path_lexists(
                download_models._model_backup_path(target_dir)
            ))
            self.assertEqual(list(root.glob(".model.capswriter-staging-*")), [])

    def test_failed_staged_model_install_preserves_live_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            target_dir.mkdir()
            artifact = target_dir / "artifact.bin"
            artifact.write_bytes(b"live-model")
            archive = root / "model.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("wrong-name.bin", b"replacement")
            asset = download_models.Asset(
                name=archive.as_posix(),
                url="https://example.test/model.zip",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                target_dir=target_dir,
                required_files=[artifact],
            )

            with self.assertRaisesRegex(FileNotFoundError, "缺少或损坏"):
                download_models._install_model_asset(asset, archive)

            self.assertEqual(artifact.read_bytes(), b"live-model")
            self.assertFalse(download_models._path_lexists(
                download_models._model_backup_path(target_dir)
            ))
            self.assertEqual(list(root.glob(".model.capswriter-staging-*")), [])

    def test_promotion_failure_rolls_back_live_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "model"
            staging_dir = root / ".model.stage"
            target_dir.mkdir()
            staging_dir.mkdir()
            (target_dir / "artifact.bin").write_bytes(b"live")
            (staging_dir / "artifact.bin").write_bytes(b"new")
            real_replace = os.replace

            def fail_staging_promotion(source, destination):
                if Path(source) == staging_dir and Path(destination) == target_dir:
                    raise OSError("simulated promotion failure")
                return real_replace(source, destination)

            with (
                patch.object(
                    download_models.os,
                    "replace",
                    side_effect=fail_staging_promotion,
                ),
                self.assertRaisesRegex(OSError, "simulated promotion failure"),
            ):
                download_models._promote_model_directory(staging_dir, target_dir)

            self.assertEqual((target_dir / "artifact.bin").read_bytes(), b"live")
            self.assertTrue(staging_dir.is_dir())
            self.assertFalse(download_models._path_lexists(
                download_models._model_backup_path(target_dir)
            ))

    def test_invalid_cached_archive_is_removed_and_redownloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            archive.write_bytes(b"corrupt-cache")
            verified_bytes = b"verified-download"
            asset = download_models.Asset(
                name=archive.as_posix(),
                url="https://example.test/model.zip",
                sha256=hashlib.sha256(verified_bytes).hexdigest(),
                target_dir=root / "model",
                required_files=[],
            )

            def fake_download(
                _url,
                destination,
                *,
                timeout=None,
                max_bytes=None,
                expected_bytes=None,
            ):
                self.assertIsNone(timeout)
                self.assertIsNone(max_bytes)
                self.assertIsNone(expected_bytes)
                self.assertFalse(destination.exists())
                destination.write_bytes(verified_bytes)

            with (
                patch.object(download_models, "_download", side_effect=fake_download),
                redirect_stdout(io.StringIO()),
            ):
                with download_models._ensure_verified_archive(asset) as (
                    source,
                    archive_file,
                ):
                    self.assertEqual(archive_file.read(), verified_bytes)

            self.assertEqual(source, "downloaded")
            self.assertEqual(archive.read_bytes(), verified_bytes)

    def test_invalid_downloaded_archive_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            asset = download_models.Asset(
                name=archive.as_posix(),
                url="https://example.test/model.zip",
                sha256=hashlib.sha256(b"expected").hexdigest(),
                target_dir=root / "model",
                required_files=[],
            )

            def fake_download(
                _url,
                destination,
                *,
                timeout=None,
                max_bytes=None,
                expected_bytes=None,
            ):
                self.assertIsNone(timeout)
                self.assertIsNone(max_bytes)
                self.assertIsNone(expected_bytes)
                destination.write_bytes(b"corrupt-download")

            with (
                patch.object(download_models, "_download", side_effect=fake_download),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "校验失败，已删除"),
            ):
                with download_models._ensure_verified_archive(asset):
                    self.fail("invalid archive must not be yielded")

            self.assertFalse(archive.exists())

    def test_verified_archive_inode_remains_stable_through_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            replacement = root / "replacement.zip"
            verified_payload = b"verified-model"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("artifact.bin", verified_payload)
            with zipfile.ZipFile(replacement, "w") as zip_file:
                zip_file.writestr("artifact.bin", b"unverified-replacement")

            archive_bytes = archive.read_bytes()
            target_dir = root / "model"
            artifact = target_dir / "artifact.bin"
            asset = download_models.Asset(
                name=archive.as_posix(),
                url="https://example.test/model.zip",
                sha256=hashlib.sha256(archive_bytes).hexdigest(),
                target_dir=target_dir,
                required_files=[artifact],
                required_sizes=[len(verified_payload)],
                required_sha256=[hashlib.sha256(verified_payload).hexdigest()],
                archive_size=len(archive_bytes),
            )

            with download_models._ensure_verified_archive(asset) as (
                source,
                archive_file,
            ):
                self.assertEqual(source, "cached")
                os.replace(replacement, archive)
                download_models._install_model_asset(asset, archive_file)

            self.assertEqual(artifact.read_bytes(), verified_payload)
            with zipfile.ZipFile(archive) as zip_file:
                self.assertEqual(
                    zip_file.read("artifact.bin"),
                    b"unverified-replacement",
                )

    def test_extract_llama_rejects_tar_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "llama.tar.gz"
            data = b"bad"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("../escaped.so")
                info.size = len(data)
                tar_file.addfile(info, io.BytesIO(data))

            with self.assertRaisesRegex(ValueError, "不安全"):
                download_models._extract_llama_binaries(archive)

            self.assertFalse((root / "escaped.so").exists())

    def test_extract_llama_materializes_safe_internal_link_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "llama.tar.gz"
            target_dir = root / "target"
            ggml_data = b"ggml-library"
            llama_data = b"llama-library"
            library_contents = {
                "libggml.so.0.9.5": ggml_data,
                "libggml.so.0": ggml_data,
                "libggml.so": ggml_data,
                "libllama.so.0.0.7798": llama_data,
                "libllama.so.0": llama_data,
                "libllama.so": llama_data,
                "libggml-base.so": b"library:libggml-base.so",
                "libggml-base.so.0": b"library:libggml-base.so.0",
                "libggml-cpu.so": b"library:libggml-cpu.so",
            }
            with tarfile.open(archive, "w:gz") as tar_file:
                ggml = tarfile.TarInfo("llama/libggml.so.0.9.5")
                ggml.size = len(ggml_data)
                tar_file.addfile(ggml, io.BytesIO(ggml_data))

                ggml_soname = tarfile.TarInfo("llama/libggml.so.0")
                ggml_soname.type = tarfile.SYMTYPE
                ggml_soname.linkname = "libggml.so.0.9.5"
                tar_file.addfile(ggml_soname)

                ggml_unversioned = tarfile.TarInfo("llama/libggml.so")
                ggml_unversioned.type = tarfile.SYMTYPE
                ggml_unversioned.linkname = "libggml.so.0"
                tar_file.addfile(ggml_unversioned)

                llama = tarfile.TarInfo("llama/libllama.so.0.0.7798")
                llama.size = len(llama_data)
                tar_file.addfile(llama, io.BytesIO(llama_data))

                llama_soname = tarfile.TarInfo("llama/libllama.so.0")
                llama_soname.type = tarfile.LNKTYPE
                llama_soname.linkname = "llama/libllama.so.0.0.7798"
                tar_file.addfile(llama_soname)

                llama_unversioned = tarfile.TarInfo("llama/libllama.so")
                llama_unversioned.type = tarfile.SYMTYPE
                llama_unversioned.linkname = "libllama.so.0"
                tar_file.addfile(llama_unversioned)

                for name in (
                    "libggml-base.so",
                    "libggml-base.so.0",
                    "libggml-cpu.so",
                ):
                    data = library_contents[name]
                    library = tarfile.TarInfo(f"llama/{name}")
                    library.size = len(data)
                    tar_file.addfile(library, io.BytesIO(data))

            expected_manifest = self._manifest_for_contents(library_contents)
            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"cpu": expected_manifest},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                download_models._extract_llama_binaries(archive)
                self.assertTrue(download_models._llama_binaries_ready())

            self.assertTrue((target_dir / download_models.LLAMA_READY_MARKER).is_file())
            for name in ("libggml.so.0.9.5", "libggml.so.0", "libggml.so"):
                materialized = target_dir / name
                self.assertEqual(materialized.read_bytes(), ggml_data)
                self.assertFalse(materialized.is_symlink())
            for name in ("libllama.so.0.0.7798", "libllama.so.0", "libllama.so"):
                materialized = target_dir / name
                self.assertEqual(materialized.read_bytes(), llama_data)
                self.assertFalse(materialized.is_symlink())

    def test_extract_llama_rejects_absolute_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "llama.tar.gz"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("libllama.so")
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar_file.addfile(info)

            with self.assertRaisesRegex(ValueError, "链接目标不安全"):
                download_models._extract_llama_binaries(archive)

    def test_extract_llama_rejects_symlink_traversal_and_escaping_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "llama.tar.gz"
            with tarfile.open(archive, "w:gz") as tar_file:
                first = tarfile.TarInfo("llama/libllama.so")
                first.type = tarfile.SYMTYPE
                first.linkname = "libllama.so.0"
                tar_file.addfile(first)

                escape = tarfile.TarInfo("llama/libllama.so.0")
                escape.type = tarfile.SYMTYPE
                escape.linkname = "../../etc/passwd"
                tar_file.addfile(escape)

            with self.assertRaisesRegex(ValueError, "链接目标越界"):
                download_models._extract_llama_binaries(archive)

            self.assertFalse((root / "etc" / "passwd").exists())

    def test_extract_llama_rejects_escaping_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "llama.tar.gz"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("llama/libllama.so")
                info.type = tarfile.LNKTYPE
                info.linkname = "../outside.so"
                tar_file.addfile(info)

            with self.assertRaisesRegex(ValueError, "链接目标越界"):
                download_models._extract_llama_binaries(archive)

    def test_extract_llama_rejects_special_members(self) -> None:
        for member_type in (tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            with self.subTest(member_type=member_type):
                with tempfile.TemporaryDirectory() as tmp:
                    archive = Path(tmp) / "llama.tar.gz"
                    with tarfile.open(archive, "w:gz") as tar_file:
                        info = tarfile.TarInfo("llama/unsafe")
                        info.type = member_type
                        tar_file.addfile(info)

                    with self.assertRaisesRegex(ValueError, "成员类型不安全"):
                        download_models._extract_llama_binaries(archive)

    def test_extract_llama_enforces_member_and_materialized_size_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "llama.tar.gz"
            data = b"abcd"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("llama/libllama.so")
                info.size = len(data)
                tar_file.addfile(info, io.BytesIO(data))

                alias = tarfile.TarInfo("llama/libllama.so.0")
                alias.type = tarfile.SYMTYPE
                alias.linkname = "libllama.so"
                tar_file.addfile(alias)

            with (
                patch.object(download_models, "MAX_LLAMA_ARCHIVE_MEMBERS", 1),
                self.assertRaisesRegex(ValueError, "成员过多"),
            ):
                download_models._extract_llama_binaries(archive)

            with (
                patch.object(download_models, "MAX_LLAMA_ARCHIVE_MEMBERS", 2),
                patch.object(download_models, "MAX_LLAMA_ARCHIVE_EXTRACTED_BYTES", 7),
                self.assertRaisesRegex(ValueError, "解压大小超过上限"),
            ):
                download_models._extract_llama_binaries(archive)

    def test_llama_download_failure_returns_clean_error(self) -> None:
        fake_asset = download_models.Asset(
            name="unit-test-llama.tar.gz",
            url="https://example.test/llama.tar.gz",
            sha256="",
            target_dir=Path("unused"),
            required_files=[],
        )
        stderr = io.StringIO()
        with (
            patch.object(download_models, "LLAMA_CPP_ASSETS", {"cpu": fake_asset}),
            patch.object(download_models, "_llama_binaries_ready", return_value=False),
            patch.object(download_models, "_download", side_effect=TimeoutError("timed out")),
            patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            code = download_models._prepare_llama_binaries()

        self.assertEqual(code, 1)
        self.assertIn("下载 llama.cpp 压缩包失败: unit-test-llama.tar.gz: timed out", stderr.getvalue())

    def test_official_llama_manifests_match_b7798_layout(self) -> None:
        cpu = download_models.LLAMA_CPP_OFFICIAL_MANIFESTS["cpu"]
        vulkan = download_models.LLAMA_CPP_OFFICIAL_MANIFESTS["vulkan"]

        self.assertEqual(len(cpu), 28)
        self.assertEqual(len(vulkan), 29)
        self.assertEqual(
            set(vulkan) - set(cpu),
            {"libggml-vulkan.so"},
        )
        self.assertEqual(
            cpu["libggml-cpu.so"],
            cpu["libggml-cpu-x64.so"],
        )
        self.assertEqual(
            vulkan["libggml-vulkan.so"],
            {
                "size": 56414152,
                "sha256": (
                    "0826ad3bfa43c30209cf155edb226dab776d98befce72a7df89f4055b61a541d"
                ),
            },
        )
        for backend, manifest in (("cpu", cpu), ("vulkan", vulkan)):
            with self.subTest(backend=backend):
                self.assertTrue(
                    set(download_models._llama_required_names(backend))
                    <= set(manifest)
                )

    def test_llama_binaries_ready_rejects_unversioned_only_libraries(self) -> None:
        previous_required = [
            "libggml.so",
            "libggml-base.so",
            "libllama.so",
            "libggml-cpu.so",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, previous_required)
            self._write_llama_marker(target_dir, "cpu")
            expected_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(expected_manifest)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"cpu": expected_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertFalse(download_models._llama_binaries_ready())

    def test_llama_binaries_ready_rejects_missing_marker_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, download_models.LLAMA_REQUIRED_CPU_LIBRARIES)
            expected_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(expected_manifest)
            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"cpu": expected_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertFalse(download_models._llama_binaries_ready())
                self._write_llama_marker(target_dir, "cpu")
                self.assertTrue(download_models._llama_binaries_ready())
                original = (target_dir / "libllama.so").read_bytes()
                (target_dir / "libllama.so").write_bytes(
                    bytes([original[0] ^ 1]) + original[1:]
                )
                self.assertFalse(
                    download_models._llama_binaries_ready(),
                    "same-size llama corruption must invalidate the marker",
                )
                (target_dir / "libllama.so").write_bytes(original)
                self.assertTrue(download_models._llama_binaries_ready())
                (target_dir / "libllama.so").write_bytes(b"x")
                self.assertFalse(download_models._llama_binaries_ready())

    def test_llama_marker_cannot_self_attest_replaced_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(
                target_dir,
                download_models.LLAMA_REQUIRED_CPU_LIBRARIES,
            )
            expected_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(expected_manifest)
            self._write_llama_marker(target_dir, "cpu")

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"cpu": expected_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertTrue(download_models._llama_binaries_ready())
                (target_dir / "libllama.so").write_bytes(b"forged-native-library")
                # Even a freshly rewritten self-consistent marker must not turn
                # unpinned native bytes into an official runtime.
                self._write_llama_marker(target_dir, "cpu")
                self.assertFalse(download_models._llama_binaries_ready())

    def test_llama_multi_directory_promotion_rolls_back_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_one = root / "one" / "bin"
            target_two = root / "two" / "bin"
            stage_one = root / "one" / ".bin.stage"
            stage_two = root / "two" / ".bin.stage"
            for path, content in (
                (target_one, b"old-one"),
                (target_two, b"old-two"),
                (stage_one, b"new-one"),
                (stage_two, b"new-two"),
            ):
                path.mkdir(parents=True)
                (path / "value").write_bytes(content)

            real_replace = os.replace

            def fail_second_stage(source, destination):
                if Path(source) == stage_two and Path(destination) == target_two:
                    raise OSError("simulated llama promotion failure")
                return real_replace(source, destination)

            with (
                patch.object(
                    download_models.os,
                    "replace",
                    side_effect=fail_second_stage,
                ),
                self.assertRaisesRegex(OSError, "simulated llama promotion failure"),
            ):
                download_models._promote_llama_staging_dirs(
                    {target_one: stage_one, target_two: stage_two}
                )

            self.assertEqual((target_one / "value").read_bytes(), b"old-one")
            self.assertEqual((target_two / "value").read_bytes(), b"old-two")
            self.assertFalse(download_models._path_lexists(
                download_models._llama_backup_path(target_one)
            ))
            self.assertFalse(download_models._path_lexists(
                download_models._llama_backup_path(target_two)
            ))

    def test_llama_binaries_ready_accepts_runtime_linked_cpu_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, download_models.LLAMA_REQUIRED_CPU_LIBRARIES)
            self._write_llama_marker(target_dir, "cpu")
            expected_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(expected_manifest)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"cpu": expected_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertTrue(download_models._llama_binaries_ready())

    def test_llama_binaries_ready_requires_vulkan_backend_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, download_models.LLAMA_REQUIRED_CPU_LIBRARIES)
            self._write_llama_marker(target_dir, "vulkan")
            cpu_only_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(cpu_only_manifest)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"vulkan": cpu_only_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "vulkan"}),
            ):
                self.assertFalse(download_models._llama_binaries_ready())

            self._write_libraries(
                target_dir,
                download_models.LLAMA_REQUIRED_VULKAN_LIBRARIES,
            )
            self._write_llama_marker(target_dir, "vulkan")
            vulkan_manifest = download_models._llama_library_manifest(target_dir)
            self.assertIsNotNone(vulkan_manifest)
            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(
                    download_models.LLAMA_CPP_OFFICIAL_MANIFESTS,
                    {"vulkan": vulkan_manifest or {}},
                ),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "vulkan"}),
            ):
                self.assertTrue(download_models._llama_binaries_ready())


if __name__ == "__main__":
    unittest.main()
