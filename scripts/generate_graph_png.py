#!/usr/bin/env python
"""Generate docs/graph.png from docs/graph.mmd."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MMD_PATH = ROOT / "docs" / "graph.mmd"
PNG_PATH = ROOT / "docs" / "graph.png"


def _try_mermaid_cli() -> bool:
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return False
    env = os.environ.copy()
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    result = subprocess.run(
        [mmdc, "-i", str(MMD_PATH), "-o", str(PNG_PATH)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return False
    return PNG_PATH.exists()


def _try_npx_mermaid() -> bool:
    env = os.environ.copy()
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    result = subprocess.run(
        [
            "npx",
            "--yes",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(MMD_PATH),
            "-o",
            str(PNG_PATH),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return False
    return PNG_PATH.exists()


def _try_mermaid_ink() -> bool:
    source = MMD_PATH.read_text(encoding="utf-8")
    encoded = base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}"
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, timeout=90)
    response.raise_for_status()
    PNG_PATH.write_bytes(response.content)
    return True


def main() -> int:
    if not MMD_PATH.exists():
        print(f"Missing {MMD_PATH}", file=sys.stderr)
        return 1

    for name, fn in (
        ("mermaid-cli", _try_mermaid_cli),
        ("npx @mermaid-js/mermaid-cli", _try_npx_mermaid),
        ("mermaid.ink", _try_mermaid_ink),
    ):
        try:
            if fn():
                print(f"Generated {PNG_PATH} via {name}")
                return 0
        except Exception as exc:
            print(f"{name} failed: {exc}", file=sys.stderr)

    print("Could not generate graph.png. Run manually:", file=sys.stderr)
    print("  npx @mermaid-js/mermaid-cli -i docs/graph.mmd -o docs/graph.png", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
