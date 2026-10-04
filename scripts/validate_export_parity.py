"""Reproducible CPU FP32 export parity with non-vacuous detection acceptance.

Uses prepared local nano weights and bundled Ultralytics photos; never downloads.
The coordinator uses production export jobs; native comparisons run separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" if (ROOT / "src/yolo_workbench").is_dir() else ROOT / "app"
sys.path.insert(0, str(SOURCE))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    from yolo_workbench.storage import atomic_write, json_text

    atomic_write(Path(path), json_text(value))


def letterbox(image, size):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    new_width, new_height = round(width * gain), round(height * gain)
    left, top = round((size - new_width) / 2 - 0.1), round((size - new_height) / 2 - 0.1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    return tensor, {"gain": gain, "left": left, "top": top, "width": width, "height": height}


def iou_matrix(first, second):
    import numpy as np

    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    overlap = np.maximum(
        0,
        np.minimum(first[:, None, 2:], second[None, :, 2:])
        - np.maximum(first[:, None, :2], second[None, :, :2]),
    )
    intersection = overlap.prod(axis=2)
    first_area = np.maximum(0, first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = np.maximum(0, second[:, 2:] - second[:, :2]).prod(axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def decode(raw, family, transform, *, confidence=0.25, nms_iou=0.45, classes=80, max_det=300):
    import numpy as np

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite raw model output")
    if family == "yolo26":
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"Expected end-to-end [N,6], got {values.shape}")
        selected = values[values[:, 4] >= confidence].copy()
    else:
        if values.ndim != 2 or values.shape[0] != 4 + classes:
            raise ValueError(f"Expected decoded [4+C,N], got {values.shape}")
        values = values.T
        category = values[:, 4:].argmax(axis=1)
        scores = values[:, 4:].max(axis=1)
        indices = np.flatnonzero(scores >= confidence)
        box = values[indices, :4].copy()
        xyxy = np.column_stack((box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2))
        selected = np.column_stack((xyxy, scores[indices], category[indices]))
        order = np.argsort(-selected[:, 4], kind="stable")
        kept = []
        while len(order) and len(kept) < max_det:
            best, remainder = order[0], order[1:]
            kept.append(best)
            if not len(remainder):
                break
            overlaps = iou_matrix(selected[[best], :4], selected[remainder, :4])[0]
            suppress = (overlaps > nms_iou) & (selected[remainder, 5] == selected[best, 5])
            order = remainder[~suppress]
        selected = selected[kept]
    if selected.size:
        selected[:, [0, 2]] = (selected[:, [0, 2]] - transform["left"]) / transform["gain"]
        selected[:, [1, 3]] = (selected[:, [1, 3]] - transform["top"]) / transform["gain"]
        selected[:, [0, 2]] = selected[:, [0, 2]].clip(0, transform["width"])
        selected[:, [1, 3]] = selected[:, [1, 3]].clip(0, transform["height"])
    return selected.reshape(-1, 6)


def compare_detections(reference, candidate, *, min_iou=0.99, max_score_difference=0.001):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference, candidate = np.asarray(reference).reshape(-1, 6), np.asarray(candidate).reshape(-1, 6)
    result = {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched_count": 0,
        "min_matched_iou": None,
        "max_score_difference": None,
        "unmatched_reference": len(reference),
        "unmatched_candidate": len(candidate),
    }
    if not len(reference) and not len(candidate):
        return {**result, "status": "vacuous_no_detections", "passed": None}
    overlaps, differences = [], []
    common_classes = set(reference[:, 5].astype(int)) & set(candidate[:, 5].astype(int))
    for category in sorted(common_classes):
        first, second = reference[reference[:, 5] == category], candidate[candidate[:, 5] == category]
        matrix = iou_matrix(first[:, :4], second[:, :4])
        rows, columns = linear_sum_assignment(-matrix)
        overlaps.extend(matrix[rows, columns].tolist())
        differences.extend(np.abs(first[rows, 4] - second[columns, 4]).tolist())
    if overlaps:
        result.update(
            matched_count=len(overlaps),
            min_matched_iou=min(overlaps),
            max_score_difference=max(differences),
            unmatched_reference=len(reference) - len(overlaps),
            unmatched_candidate=len(candidate) - len(overlaps),
        )
    passed = (
        len(overlaps) == len(reference) == len(candidate)
        and bool(overlaps)
        and min(overlaps) >= min_iou
        and max(differences) <= max_score_difference
    )
    return {**result, "status": "passed" if passed else "failed", "passed": passed}


def run_case(spec):
    from yolo_workbench.worker import configure_offline, prepare_ultralytics

    configure_offline(Path(spec["case_dir"]))
    prepare_ultralytics()
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    torch.set_num_threads(2)
    model = YOLO(spec["model"], task="detect").model.cpu().float().eval()
    package = Path(spec["package"])
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    if sha256(spec["model"]) != manifest["source_sha256"]:
        raise ValueError("Reference PT no longer matches the exported package source")
    for name, expected in manifest["files"].items():
        if sha256(package / name) != expected:
            raise ValueError(f"Exported file changed after structural validation: {name}")
    if spec["format"] == "onnx":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        runtime = ort.InferenceSession(
            str(package / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

        def execute(pixels):
            return runtime.run(None, {runtime.get_inputs()[0].name: pixels})[0]
    else:
        import ncnn

        runtime = ncnn.Net()
        runtime.opt.use_vulkan_compute = False
        runtime.opt.use_packing_layout = False
        runtime.opt.use_fp16_packed = False
        runtime.opt.use_fp16_storage = False
        runtime.opt.use_fp16_arithmetic = False
        runtime.opt.use_bf16_storage = False
        runtime.opt.num_threads = 2
        if runtime.load_param(str(package / "model.ncnn.param")) or runtime.load_model(
            str(package / "model.ncnn.bin")
        ):
            raise RuntimeError("NCNN load failed")

        def execute(pixels):
            extractor = runtime.create_extractor()
            planar = np.ascontiguousarray(pixels[0])
            tensor = ncnn.Mat(planar)
            if extractor.input(manifest["input_name"], tensor):
                raise RuntimeError("NCNN input failed")
            status, output = extractor.extract(manifest["output_name"])
            if status:
                raise RuntimeError(f"NCNN extraction failed: {status}")
            return np.asarray(output).copy()

    results = []
    for source in [*spec["images"], "synthetic_ramp"]:
        if source == "synthetic_ramp":
            # Deterministic control; no model download or unknown generated asset.
            rgb = np.linspace(0, 255, 480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
            image_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        else:
            rgb = np.asarray(Image.open(source).convert("RGB"))
            image_hash = sha256(source)
        tensor, transform = letterbox(rgb, spec["imgsz"])
        with torch.inference_mode():
            raw_pt = model(torch.from_numpy(tensor))
        raw_pt = raw_pt[0] if isinstance(raw_pt, (tuple, list)) else raw_pt
        raw_pt = raw_pt.cpu().numpy()
        raw_candidate = execute(tensor)
        if not np.isfinite(raw_pt).all() or not np.isfinite(raw_candidate).all():
            raise ValueError("Non-finite outputs")
        first = decode(raw_pt, spec["family"], transform, classes=len(manifest["classes"]))
        second = decode(raw_candidate, spec["family"], transform, classes=len(manifest["classes"]))
        comparison = compare_detections(first, second)
        raw_diagnostics = {
            "reference_shape": list(raw_pt.shape),
            "candidate_shape": list(raw_candidate.shape),
        }
        if spec["family"] != "yolo26":
            delta = np.abs(raw_pt.squeeze(0).astype(np.float64) - raw_candidate.squeeze().astype(np.float64))
            raw_diagnostics.update(
                max_absolute_difference=float(delta.max()),
                mean_absolute_difference=float(delta.mean()),
                note="Coordinate and probability channels share this diagnostic; acceptance uses detections.",
            )
        else:
            raw_diagnostics["note"] = (
                "End-to-end topk row order is not a numerical invariant; detections are assigned by class/IoU."
            )
        results.append(
            {
                "image": Path(source).name,
                "image_sha256": image_hash,
                "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                "image_shape": list(rgb.shape),
                "synthetic": source == "synthetic_ramp",
                **comparison,
                "raw_output_diagnostics": raw_diagnostics,
            }
        )
    real_nonempty = [item for item in results if not item["synthetic"] and item["passed"] is True]
    passed = bool(real_nonempty) and not any(item["passed"] is False for item in results)
    return {
        "family": spec["family"],
        "format": spec["format"],
        "profile": manifest["profile"],
        "imgsz": spec["imgsz"],
        "model_sha256": sha256(spec["model"]),
        "export_files": manifest["files"],
        "status": "passed" if passed else "failed_or_inconclusive",
        "passed": passed,
        "real_nonempty_pass_count": len(real_nonempty),
        "vacuous_case_count": sum(item["status"] == "vacuous_no_detections" for item in results),
        "cases": results,
        "versions": {name: version(name) for name in ("torch", "ultralytics", "onnxruntime", "ncnn")},
    }


def self_test():
    import numpy as np

    box = np.array([[10, 10, 110, 110, 0.8, 0]])
    assert compare_detections(box, box)["passed"] is True
    assert compare_detections([], [])["passed"] is None
    assert compare_detections([], box)["passed"] is False
    changed = box.copy()
    changed[:, 5] = 1
    assert compare_detections(box, changed)["passed"] is False
    changed = box.copy()
    changed[:, 4] += 0.002
    assert compare_detections(box, changed)["passed"] is False
    changed = box.copy()
    changed[:, 0] += 2
    assert compare_detections(box, changed)["passed"] is False
    duplicate = np.vstack((box, box + np.array([200, 0, 200, 0, -0.1, 0])))
    assert compare_detections(duplicate, duplicate[::-1])["passed"] is True
    print("PASS seven acceptance checks (including non-vacuity and order independence)")


def coordinate(args):
    from yolo_workbench.jobs import JobManager

    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    runtime_root = (
        args.train_python.parent.parent
        if args.train_python.parent.name == "Scripts"
        else args.train_python.parent
    )
    assets = runtime_root / "Lib/site-packages/ultralytics/assets"
    images = [str(path.resolve()) for path in args.images or [assets / "bus.jpg", assets / "zidane.jpg"]]
    if any(not Path(path).is_file() for path in images):
        raise FileNotFoundError("Prepared validation images are missing")
    cases = []
    with JobManager(artifacts / "project", {"train": args.train_python}, ROOT) as manager:
        for family, format_name in (
            ("yolov8", "onnx"),
            ("yolo11", "onnx"),
            ("yolo26", "onnx"),
            ("yolov8", "ncnn"),
        ):
            model = (args.models / f"{family}n.pt").resolve()
            profile = "generic" if family == "yolo26" else "ascript_v8" if format_name == "ncnn" else "cq"
            job = manager.start(
                "export",
                {
                    "model": str(model),
                    "family": family,
                    "format": format_name,
                    "profile": profile,
                    "imgsz": 640,
                },
            )
            deadline = time.monotonic() + 240
            while manager.active_jobs() and time.monotonic() < deadline:
                manager.poll_events()
                time.sleep(0.1)
            if job.state != "succeeded":
                raise RuntimeError(f"Production export failed: {job.error}; {job.run_dir / 'stderr.log'}")
            directory = artifacts / f"{family}-{format_name}"
            directory.mkdir()
            spec = {
                "family": family,
                "format": format_name,
                "model": str(model),
                "package": job.result["package_dir"],
                "imgsz": 640,
                "images": images,
                "case_dir": str(directory),
            }
            spec_path = directory / "request.json"
            write_json(spec_path, spec)
            with (directory / "process.log").open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    [str(args.train_python), str(Path(__file__).resolve()), "--case-request", str(spec_path)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=directory,
                    timeout=120,
                    env={**os.environ, "PYTHONPATH": str(SOURCE), "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            result_path = directory / "result.json"
            if process.returncode or not result_path.is_file():
                raise RuntimeError(f"Native comparison failed; {directory / 'process.log'}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            cases.append(result)
            print(
                f"{family}/{format_name}: {result['status']}, nonempty real cases={result['real_nonempty_pass_count']}",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "passed": all(item["passed"] for item in cases),
        "scope": "CPU FP32 pretrained nano export numerical acceptance; not application accuracy or Android acceptance",
        "thresholds": {
            "minimum_matching_iou": 0.99,
            "maximum_score_difference": 0.001,
            "exact_detection_count_and_classes": True,
            "empty_both_is_pass": False,
        },
        "preprocessing": "PIL RGB decode; OpenCV INTER_LINEAR letterbox with 114 padding; NCHW float32 / 255; 640 square",
        "postprocessing": {
            "confidence": 0.25,
            "v8_v11": "shared class-aware NMS IoU .45, max_det300",
            "yolo26": "shared end-to-end xyxy/score/class decoding, no additional NMS",
            "matching": "per-class maximum-IoU linear assignment in original image coordinates",
        },
        "comparisons": cases,
    }
    write_json(args.report, report)
    print(f"Report: {args.report}", flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-python", type=Path, default=ROOT / ".runtimes/train/Scripts/python.exe")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / ".artifacts/export-parity" / str(time.time_ns())
    )
    parser.add_argument("--report", type=Path, default=ROOT / "docs/reports/export-parity.json")
    parser.add_argument("--case-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.case_request:
        spec = json.loads(args.case_request.read_text(encoding="utf-8"))
        try:
            result = run_case(spec)
            write_json(Path(spec["case_dir"]) / "result.json", result)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
    if not args.train_python.is_file() and (ROOT / "runtime/train/python.exe").is_file():
        args.train_python = ROOT / "runtime/train/python.exe"
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
