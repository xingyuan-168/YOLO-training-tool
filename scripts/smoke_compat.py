"""Repeatable isolated CPU compatibility checks. All outputs stay outside input/."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".artifacts/compat"


def cq_probe():
    import numpy as np
    from ai_engine import AI_DEVICE_CPU, Engine, image_from_numpy
    from PIL import Image

    from yolo_workbench.inference import CqBackend

    model = ROOT / "input/模型样板/best.onnx"
    classes = (model.parent / "labels.txt").read_text(encoding="utf-8-sig").splitlines()
    rgb = np.array(Image.open(ROOT / "input/UI-1.png").convert("RGB"))
    image = np.ascontiguousarray(rgb[:, :, ::-1])
    with CqBackend(model, classes, ARTIFACTS / "cq", device=AI_DEVICE_CPU) as backend:
        adapted = backend.predict(image, 0.0)
        dll_path, config_path, manifest = backend.engine.dll_path, backend.config_path, backend.manifest
    # Engine.close releases process-global resources: compare sequential lifetimes.
    with Engine(dll_path=dll_path) as engine:
        with engine.yolo_model(320, AI_DEVICE_CPU, 0, 1) as direct:
            direct.load_model(model, config_path)
            expected = direct.infer(image_from_numpy(image), 0.0)
    if adapted["raw_detections"] != expected:
        raise AssertionError("Adapter and direct engine disagree")
    return {
        "version": version("cq-ai-engine"),
        "input_shape": manifest.input_shape,
        "output_shapes": manifest.output_shapes,
        "parity_exact": True,
        "detection_count": len(expected),
        "runtime": adapted["runtime"],
        "note": "UI screenshot at confidence 0.0; regression only, not business accuracy",
    }


def ncnn_probe(legacy=False):
    import ncnn
    import numpy as np

    from yolo_workbench.models import ModelManifest, validate_contract

    folder = ROOT / "input/模型样板" if legacy else ROOT / "input/模型样板/AScript_专用模型"
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_packing_layout = False
    net.opt.num_threads = 1
    assert net.load_param(str(folder / "best.ncnn.param")) == 0
    assert net.load_model(str(folder / "best.ncnn.bin")) == 0
    extractor = net.create_extractor()
    size = 320 if legacy else 640
    pixels = np.zeros((3, size, size), dtype=np.float32)
    tensor = ncnn.Mat(pixels)
    assert extractor.input("in0", tensor) == 0
    code, output = extractor.extract("out0")
    assert code == 0
    shape = list(np.asarray(output).shape)
    if legacy:
        shapes = [shape]
        for name in ("out1", "out2"):
            code, other = extractor.extract(name)
            assert code == 0
            shapes.append(list(np.asarray(other).shape))
        return {
            "ncnn_version": version("ncnn"),
            "output_shapes": shapes,
            "ascript_v8_compatible": False,
            "note": "Raw multi-head layout; not the decoded single-output AScript contract",
        }
    classes = (folder / "labels.txt").read_text(encoding="utf-8-sig").splitlines()
    validate_contract(
        ModelManifest("yolov8", "ncnn", classes, [1, 3, 640, 640], [shape], "probe"), "ascript_v8"
    )
    return {
        "ncnn_version": version("ncnn"),
        "output_shape": shape,
        "note": "CPU zero-input structural probe only; AScript Android acceptance remains pending",
    }


def generation_probe(family):
    import numpy as np
    import onnxruntime as ort
    from ultralytics import YOLO

    from yolo_workbench.models import inspect_onnx, validate_contract

    folder = ARTIFACTS / family
    folder.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"{family}n.yaml")
    model.save(folder / "architecture.pt")
    model = YOLO(folder / "architecture.pt")
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device="cpu", verbose=False)
    exported = model.export(
        format="onnx",
        imgsz=320,
        batch=1,
        opset=12,
        dynamic=False,
        half=False,
        simplify=False,
        nms=False,
        device="cpu",
    )
    manifest, opset = inspect_onnx(Path(exported), family, list(model.names.values()))
    if family != "yolo26":
        validate_contract(manifest, "cq", opset=opset)
    runtime = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    output = runtime.run(None, {runtime.get_inputs()[0].name: np.zeros((1, 3, 320, 320), dtype=np.float32)})
    return {
        "ultralytics": version("ultralytics"),
        "torch": version("torch"),
        "opset": opset,
        "output_shapes": [list(o.shape) for o in output],
        "note": "Nano architecture initialized without pretrained weights; forward and ONNX execution",
    }


def train_probe():
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    from yolo_workbench.dataset import DatasetService
    from yolo_workbench.storage import atomic_write

    project_dir = ARTIFACTS / f"cpu-project-{time.time_ns()}"
    fixture = ARTIFACTS / "fixture"
    fixture.mkdir(exist_ok=True)
    with DatasetService.create(project_dir, "CPU 训练冒烟", ["square"]) as project:
        for i in range(6):
            image = Image.new("RGB", (64, 64), (20 + i, 20, 20))
            ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill="red")
            path = fixture / f"{i}.png"
            image.save(path)
            project.import_image(path, labels="0 0.5 0.5 0.5 0.5")
        frozen = project.snapshot(project.split(grouped=False), {"epochs": 1})
    model = YOLO("yolov8n.yaml")

    def preserve(trainer):
        atomic_write(trainer.wdir / "resume.pt", trainer.last.read_bytes())

    model.add_callback("on_model_save", preserve)
    model.train(
        data=str(frozen / "data.yaml"),
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        plots=False,
        cache=False,
        project=str(project_dir / "runs"),
        name="smoke",
        exist_ok=False,
        verbose=False,
        val=True,
        pretrained=False,
    )
    import torch

    checkpoint = torch.load(
        project_dir / "runs/smoke/weights/resume.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer"] is not None and checkpoint["epoch"] == 0
    return {
        "epochs": 1,
        "device": "cpu",
        "resume_optimizer_preserved": True,
        "result_csv": (project_dir / "runs/smoke/results.csv").relative_to(ROOT).as_posix(),
        "note": "Synthetic six-image pipeline check; not an accuracy measurement",
    }


def main():
    parser = argparse.ArgumentParser()
    cases = ["cq", "ncnn", "legacy_ncnn", "yolov8", "yolo11", "yolo26", "train"]
    parser.add_argument("--case", choices=cases)
    parser.add_argument("--only", nargs="+", choices=cases)
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "ultralytics").mkdir(exist_ok=True)
    os.environ.update(
        YOLO_AUTOINSTALL="false",
        YOLO_OFFLINE="true",
        YOLO_CONFIG_DIR=str(ARTIFACTS / "ultralytics"),
        OMP_NUM_THREADS="2",
    )
    if args.case:
        functions = {
            "cq": cq_probe,
            "ncnn": ncnn_probe,
            "legacy_ncnn": lambda: ncnn_probe(True),
            "train": train_probe,
        }
        result = functions[args.case]() if args.case in functions else generation_probe(args.case)
        (ARTIFACTS / f"{args.case}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"PASS {args.case}", flush=True)
        return
    report_path = ARTIFACTS / "report.json"
    report = json.loads(report_path.read_text()) if args.only and report_path.exists() else {}
    for case in args.only or cases:
        python = ROOT / (
            ".runtimes/inference/Scripts/python.exe"
            if case in {"cq", "ncnn", "legacy_ncnn"}
            else ".runtimes/train/Scripts/python.exe"
        )
        print(f"Running {case} in isolated process...", flush=True)
        started = time.perf_counter()
        log = ARTIFACTS / f"{case}.log"
        with log.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(
                    [str(python), str(Path(__file__).resolve()), "--case", case],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1"},
                    timeout=240,
                )
                report[case] = {"returncode": result.returncode, "passed": result.returncode == 0}
            except subprocess.TimeoutExpired:
                report[case] = {"passed": False, "error": "timeout after 240 seconds"}
        report[case]["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        print(f"{case}: {report[case]}", flush=True)
        (ARTIFACTS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(item["passed"] for item in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
