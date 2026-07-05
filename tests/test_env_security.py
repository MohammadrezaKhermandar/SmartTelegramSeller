"""Security tests: .env must never be committed to git."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    )


def _in_git_repo() -> bool:
    if shutil.which("git") is None:
        return False
    return _git("rev-parse", "--is-inside-work-tree").returncode == 0


pytestmark = pytest.mark.skipif(
    not _in_git_repo(), reason="git not available or not a repository"
)


def test_env_is_ignored_by_git():
    result = _git("check-ignore", ".env")
    assert result.returncode == 0, ".env must be listed in .gitignore"


def test_env_is_not_tracked_by_git():
    result = _git("ls-files", ".env")
    assert result.stdout.strip() == "", ".env must never be committed"


def test_env_example_has_no_secret_values():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if "API_KEY" in line and "=" in line and not line.strip().startswith("#"):
            value = line.split("=", 1)[1].strip()
            assert value == "", f"Secret value found in .env.example: {line.split('=')[0]}"
