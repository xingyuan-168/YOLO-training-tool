"""Explicit cross-family real export checks in the isolated training runtime."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from yolo_workbench.jobs import JobManager


@pytest.mark.skipif(
    os.environ.get("YOLO_RUN_TRAINING_INTEGRATION") != "1", reason="Explicit heavy runtime check"
)
def test_yolo11_and_yolo26_real_exports(tmp_path):
    python = Path(os.environ["YOLO_TRAIN_PYTHON"])
    app = Path(__file__).resolve().parents[1]
    models = tmp_path / "models"
    models.mkdir()
    build = """
import sys
from pathlib import Path
from yolo_workbench.worker import configure_offline, prepare_ultralytics
root = Path(sys.argv[1])
configure_offline(root)
prepare_ultralytics()
from ultralytics import YOLO
for family in ("yolo11", "yolo26"):
    model = YOLO(family + "n.yaml")
    model.save(root / (family + "n.pt"))
"""
    subprocess.run(
        [str(python), "-c", build, str(models)],
        check=True,
        capture_output=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "2"},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    with JobManager(tmp_path / "project", {"train": python}, app) as manager:
        for family, format_name, profile in (
            ("yolo11", "onnx", "cq"),
            ("yolo26", "onnx", "generic"),
            ("yolo26", "ncnn", "generic"),
        ):
            job = manager.start(
                "export",
                {
                    "model": str(models / f"{family}n.pt"),
                    "family": family,
                    "format": format_name,
                    "profile": profile,
                    "imgsz": 320,
                },
            )
            deadline = time.monotonic() + 240
            while manager.active_jobs() and time.monotonic() < deadline:
                manager.poll_events()
                time.sleep(0.1)
            if job.state != "succeeded":
                pytest.fail(
                    f"{family}/{format_name}: {job.error}\n"
                    f"{(job.run_dir / 'stderr.log').read_text(encoding='utf-8')[-12000:]}"
                )
            manifest = json.loads(Path(job.result["manifest"]).read_text(encoding="utf-8"))
            assert manifest["validation"]["passed"]
            assert manifest["family"] == family
            expected = (
                [[1, 84, 2100]]
                if family == "yolo11"
                else [[1, 300, 6]]
                if format_name == "onnx"
                else [[84, 2100]]
            )
            assert manifest["output_shapes"] == expected
