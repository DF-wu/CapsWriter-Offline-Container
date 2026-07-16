# coding: utf-8

from __future__ import annotations

import io
import subprocess
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts import check_upstream_divergence as guard


class UpstreamDivergenceGuardTest(unittest.TestCase):
    def test_name_status_parser_handles_nul_unicode_rename_and_copy(self) -> None:
        output = "\0".join(
            (
                "M",
                "readme.md",
                "A",
                "fork_server/new.py",
                "D",
                "requirements-server.txt",
                "R100",
                "舊目錄/角色.py",
                "新目錄/角色.py",
                "C075",
                "LLM/大助理.py",
                "範本/大助理.py",
                "",
            )
        )

        with patch.object(guard, "git_output", return_value=output) as git_output:
            paths = guard.changed_paths("origin/master")

        git_output.assert_called_once_with(
            [
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                "--find-copies",
                "origin/master",
                "--",
            ],
            cwd=guard.ROOT,
        )
        self.assertEqual(
            paths,
            {
                "readme.md",
                "fork_server/new.py",
                "requirements-server.txt",
                "舊目錄/角色.py",
                "新目錄/角色.py",
                "LLM/大助理.py",
                "範本/大助理.py",
            },
        )

    def test_name_status_parser_rejects_incomplete_rename_record(self) -> None:
        output = "R100\0old/path.py\0"

        with patch.object(guard, "git_output", return_value=output):
            with self.assertRaisesRegex(RuntimeError, "malformed NUL-delimited"):
                guard.changed_paths("origin/master")

    def test_allowlist_covers_all_documented_unicode_paths(self) -> None:
        self.assertEqual(len(guard.ALLOWED_UPSTREAM_DIVERGENCE), 59)
        self.assertTrue(
            {
                "LLM/大助理.py",
                "core/client/audio/recorder.py",
                "core/client/connection/websocket_manager.py",
                "core/client/llm/llm_output_typing.py",
                "core/client/output/result_processor.py",
                "core/client/output/text_output.py",
                "core/server/app.py",
                "docs/显卡加速的若干问题.md",
                "docs/角色功能如何使用.md",
                "start_client.py",
            }
            <= guard.ALLOWED_UPSTREAM_DIVERGENCE
        )

    def test_upstream_tracked_changes_filter_new_fork_owned_paths(self) -> None:
        with (
            patch.object(
                guard,
                "changed_paths",
                return_value={
                    "readme.md",
                    "LLM/大助理.py",
                    "fork_server/new.py",
                },
            ),
            patch.object(
                guard,
                "path_exists_in_ref",
                side_effect=lambda _base, path, cwd=guard.ROOT: path
                in {"readme.md", "LLM/大助理.py"},
            ),
        ):
            paths = guard.upstream_tracked_changes("origin/master")

        self.assertEqual(paths, ["LLM/大助理.py", "readme.md"])

    def test_unexpected_changes_reports_only_undocumented_divergence(self) -> None:
        paths = [
            "README.en.md",
            "CLAUDE.md",
            "LLM/大助理.py",
            "build.spec",
            "core/client/audio/file_manager.py",
            "core/client/audio/recorder.py",
            "core/client/audio/stream.py",
            "core/client/connection/websocket_manager.py",
            "core/client/global_hotkey/global_hotkey.py",
            "core/client/hotword/hotword_standalone.ipynb",
            "core/client/hotword/hotword_standalone.py",
            "core/client/manager/file_runner.py",
            "core/client/manager/tray_manager.py",
            "core/client/shortcut/shortcut_manager.py",
            "core/client/state.py",
            "core/client/transcribe/file_transcriber.py",
            "core/client/transcribe/media_tool.py",
            "core/client/transcribe/srt_adjuster.py",
            "core/protocol.py",
            "core/server/app.py",
            "core/server/connection/ws_send.py",
            "core/server/engines/force_aligner_gguf/export/gguf/utility.py",
            "core/server/engines/force_aligner_gguf/inference/aligner.py",
            "core/server/engines/fun_asr_gguf/export/gguf/utility.py",
            "core/server/engines/fun_asr_gguf/inference/prompt_builder.py",
            "core/server/engines/qwen_asr_gguf/export/gguf/utility.py",
            "core/server/engines/qwen_asr_gguf/inference/aligner.py",
            "core/server/worker/gpu_boost.py",
            "core/server/worker/pipeline.py",
            "core/server/worker/process_manager.py",
            "core/server/worker/task_handler.py",
            "core/server/engines/force_aligner_gguf/inference/audio.py",
            "core/server/engines/fun_asr_gguf/inference/audio.py",
            "core/server/engines/qwen_asr_gguf/inference/audio.py",
            "core/server/engines/sensevoice_onnx/inference/audio.py",
            "core/server/schema.py",
            "core/server/state.py",
            "core/tools/window_detector.py",
            "core/ui/tray.py",
            "docs/text_merge_algorithm.md",
            "docs/显卡加速的若干问题.md",
            "docs/角色功能如何使用.md",
            "readme.md",
            "requirements-server.txt",
            "start_client.py",
            "zip_release.py",
        ]

        self.assertEqual(
            guard.unexpected_changes(paths, guard.ALLOWED_UPSTREAM_DIVERGENCE),
            ["README.en.md"],
        )

    def test_git_output_reports_timeout(self) -> None:
        with patch.object(
            guard.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["git", "diff"], timeout=15),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out after 15s"):
                guard.git_output(["diff"])

    def test_main_reports_git_runtime_error_without_traceback(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(guard, "ref_exists", side_effect=RuntimeError("git diff timed out")),
            patch.object(guard.sys, "argv", ["check_upstream_divergence.py"]),
            redirect_stderr(stderr),
        ):
            code = guard.main()

        self.assertEqual(code, 1)
        self.assertIn(
            "Upstream divergence guard failed: git diff timed out",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
