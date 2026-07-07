# coding: utf-8

from __future__ import annotations

import io
import subprocess
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts import check_upstream_divergence as guard


class UpstreamDivergenceGuardTest(unittest.TestCase):
    def test_name_status_parser_includes_rename_old_and_new_paths(self) -> None:
        output = "\n".join(
            [
                "M\treadme.md",
                "A\tfork_server/new.py",
                "D\trequirements-server.txt",
                "R100\told/path.py\tnew/path.py",
            ]
        )

        with patch.object(guard, "git_output", return_value=output):
            paths = guard.changed_paths("origin/master")

        self.assertEqual(
            paths,
            {
                "readme.md",
                "fork_server/new.py",
                "requirements-server.txt",
                "old/path.py",
                "new/path.py",
            },
        )

    def test_upstream_tracked_changes_filter_new_fork_owned_paths(self) -> None:
        with (
            patch.object(
                guard,
                "changed_paths",
                return_value={"readme.md", "fork_server/new.py"},
            ),
            patch.object(
                guard,
                "path_exists_in_ref",
                side_effect=lambda _base, path, cwd=guard.ROOT: path == "readme.md",
            ),
        ):
            paths = guard.upstream_tracked_changes("origin/master")

        self.assertEqual(paths, ["readme.md"])

    def test_unexpected_changes_reports_only_undocumented_divergence(self) -> None:
        paths = ["README.en.md", "readme.md", "requirements-server.txt"]

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
