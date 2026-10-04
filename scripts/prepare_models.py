"""Explicit online setup. Workers never call this script automatically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"])
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = root / ".setup"
    config.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    from ultralytics import YOLO, __version__
    from ultralytics.utils.downloads import attempt_download_asset

    os.chdir(root)
    manifest_path = root / "cache-manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    prepared = {
        item["file"]: item
        for item in previous.get("models", [])
        if Path(item["file"]).name == item["file"] and (root / item["file"]).is_file()
    }
    for name in args.models:
        if not re.fullmatch(r"(?:yolov8|yolo11|yolo26)[nsmlx]\.pt", name):
            raise ValueError("模型名称必须为官方权重文件名")
        path = Path(attempt_download_asset(name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"模型下载未完成：{name}")
        model = YOLO(str(path), task="detect")
        classes = [model.names[index] for index in range(len(model.names))]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        prepared[name] = {
            "file": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "classes": classes,
            "source": "https://github.com/ultralytics/assets",
            "license": "AGPL-3.0",
        }
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "ultralytics": __version__,
        "models": list(prepared.values()),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
