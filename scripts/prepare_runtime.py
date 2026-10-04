"""Explicit provisioning command, never invoked silently by a running task."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

WHEEL = "cq_ai_engine-0.14.6-py3-none-win_amd64.whl"
SHA256 = "d47d071820eac8aa40c1b9c05fee247a2a94bc6c76c78476892f5e6a38e8e193"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["gui", "train", "inference"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = root / ".runtimes" / args.role
    wheel = root / "vendor" / WHEEL
    if args.role == "inference":
        if not wheel.is_file():
            parser.error(f"请先准备已校验的 CQ_AI Wheel：{wheel}")
        with wheel.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                parser.error("CQ_AI Wheel 哈希不匹配")
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment), "UV_LINK_MODE": "copy"}
    offline = ["--offline"] if args.offline else []
    subprocess.run(
        ["uv", "sync", "--locked", "--python", "3.12", "--no-dev", "--extra", args.role, *offline],
        cwd=root,
        env=env,
        check=True,
    )
    if args.role == "inference":
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment / "Scripts/python.exe"),
                str(wheel),
                *offline,
            ],
            cwd=root,
            env=env,
            check=True,
        )
    print(f"Prepared {args.role}: {environment}")


if __name__ == "__main__":
    main()
