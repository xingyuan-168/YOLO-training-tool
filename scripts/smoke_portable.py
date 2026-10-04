"""Exercise a relocated portable GUI and its workers with Python networking denied."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def offline_workers(bundle, guard):
    """Install a temporary startup guard in the two task-owned portable Pythons.

    JobManager correctly discards the caller's PYTHONPATH, so placing the guard
    only there would not actually test the worker's offline behavior.
    """
    written = []
    try:
        for role in ("train", "inference"):
            path = bundle / "runtime" / role / "Lib" / "site-packages" / "sitecustomize.py"
            if path.exists():
                raise RuntimeError(f"不覆盖已有 Python 启动文件：{path}")
            path.write_text(
                "import socket,os\nfrom pathlib import Path\n"
                f"Path({str(guard)!r},str(os.getpid())+'.guard').write_text('active')\n"
                "def denied(*args, **kwargs):\n    raise OSError('Offline portable verification: network disabled')\n"
                "socket.socket.connect=denied\nsocket.socket.connect_ex=denied\nsocket.create_connection=denied\n",
                encoding="utf-8",
            )
            written.append(path)
        yield
    finally:
        for path in written:
            path.unlink()


def run(bundle: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    executable = bundle / "YOLOWorkbench.exe"
    train_python = bundle / "runtime" / "train" / "python.exe"
    results = {
        "bundle": str(bundle),
        "network_check": "Python socket.connect/create_connection disabled in worker processes",
        "steps": [],
    }
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    environment.update(
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
        QT_QPA_PLATFORM="offscreen",
        YOLO_WORKBENCH_HOME=str(destination / "userdata"),
    )
    with (
        tempfile.TemporaryDirectory(prefix="yolo-offline-", dir=destination) as guard,
        offline_workers(bundle, Path(guard)),
    ):
        guard_path = Path(guard)
        environment["PYTHONPATH"] = str(bundle / "app")
        project = destination / "中文 项目"
        setup = """import json,sys
from pathlib import Path
from yolo_workbench.dataset import DatasetService
sys.path.insert(0,sys.argv[1])
from create_demo_project import create_demo
root=Path(sys.argv[2])
create_demo(root,8)
with DatasetService(root) as data:
    snap=data.snapshot(data.split(),{'epochs':1})
    image=data.get_path(data.list_assets()[1]['id'])
print(json.dumps({'snapshot':str(snap),'image':str(image),'prefix':sys.prefix,'sys_path':sys.path}))
"""
        completed = subprocess.run(
            [str(train_python), "-c", setup, str(bundle / "scripts"), str(project)],
            cwd=bundle,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        fixture = json.loads(completed.stdout.strip().splitlines()[-1])
        results["python_probe"] = fixture

        def job(kind, parameters, runtime="train"):
            index = len(results["steps"])
            request_path, report_path = (
                destination / f"request-{index}.json",
                destination / f"result-{index}.json",
            )
            request_path.write_text(
                json.dumps({"kind": kind, "parameters": parameters, "runtime": runtime}), encoding="utf-8"
            )
            start = time.monotonic()
            process = subprocess.run(
                [
                    str(executable),
                    "--no-restore",
                    "--project",
                    str(project),
                    "--smoke-job",
                    str(request_path),
                    "--smoke-report",
                    str(report_path),
                ],
                cwd=bundle,
                env=environment,
                timeout=360,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if not report_path.is_file():
                raise RuntimeError(f"GUI 未生成任务报告，返回码 {process.returncode}")
            result = json.loads(report_path.read_text(encoding="utf-8"))
            worker = json.loads((Path(result["run_dir"]) / "job.json").read_text(encoding="utf-8"))
            result["offline_guard_active"] = (guard_path / f"{worker['pid']}.guard").is_file()
            diagnostics = (Path(result["run_dir"]) / "stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )
            result["missing_glyph_warning"] = "Glyph " in diagnostics and "missing from font" in diagnostics
            results["steps"].append(
                {"kind": kind, "runtime": runtime, "elapsed_seconds": time.monotonic() - start, **result}
            )
            if (
                result["state"] != "succeeded"
                or result["gui_imported_heavy"]
                or not result["offline_guard_active"]
                or result["missing_glyph_warning"]
            ):
                raise RuntimeError(json.dumps(result, ensure_ascii=False))
            return result["result"]

        try:
            trained = job(
                "train",
                {
                    "model": str(bundle / "models" / "yolov8n.pt"),
                    "snapshot": fixture["snapshot"],
                    "config": {
                        "family": "yolov8",
                        "scale": "n",
                        "imgsz": 64,
                        "epochs": 1,
                        "batch": 4,
                        "device": "cpu",
                        "workers": 0,
                        "patience": 100,
                    },
                    "expert_yaml": "amp: false\noptimizer: SGD\n",
                },
            )
            evaluated = job(
                "evaluate",
                {
                    "model": trained["best"],
                    "snapshot": fixture["snapshot"],
                    "imgsz": 64,
                    "device": "cpu",
                    "batch": 4,
                },
            )
            results["evaluation_source"] = evaluated.get("source")
            exported = job(
                "export",
                {
                    "model": trained["best"],
                    "family": "yolov8",
                    "profile": "cq",
                    "format": "onnx",
                    "imgsz": 320,
                    "device": "cpu",
                },
            )
            package = (
                exported.get("package_dir")
                or exported.get("output_dir")
                or exported.get("package")
                or exported.get("directory")
            )
            if not package:
                raise RuntimeError(f"导出结果缺少模型包路径：{exported}")
            job(
                "infer",
                {
                    "model": package,
                    "backend": "cq",
                    "device": "cpu",
                    "family": "yolov8",
                    "source": "image",
                    "source_path": fixture["image"],
                    "confidence": 0.5,
                    "iou": 0.45,
                },
                "inference",
            )
            results["state"] = "passed"
        except Exception as exc:
            results.update(state="failed", error=str(exc))
            raise
        finally:
            (destination / "portable-report.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle.resolve(), args.output.resolve()), ensure_ascii=False))
