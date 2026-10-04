"""Job worker dispatch for deployment inference, capture, and real benchmarks."""

from __future__ import annotations

import json
import math
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from .frames import SharedFrameWriter
from .inference import create_backend, validate_thresholds
from .storage import atomic_write

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def save_image(path: Path, bgr_image):
    from io import BytesIO

    from PIL import Image

    output = BytesIO()
    Image.fromarray(bgr_image[:, :, ::-1]).save(output, format="PNG")
    atomic_write(path, output.getvalue())


def _source_frames(parameters, stopped, emit, *, repeat_image=False):
    source = parameters.get("source", "image")
    source_path = parameters.get("source_path") or parameters.get("path")
    if source in {"image", "folder", "video"}:
        if not source_path:
            raise ValueError("请选择图片、目录或视频路径")
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            raise ValueError("输入来源不存在")
    session = datetime.now(timezone.utc).strftime("source-%Y%m%dT%H%M%S.%fZ")
    if source in {"image", "folder"}:
        if source == "folder":
            paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
            if not paths:
                raise ValueError("所选目录没有支持的图片")
        else:
            paths = [path]
        sequence = 0
        while True:
            for image_path in paths:
                if stopped():
                    return
                started = time.perf_counter()
                image = read_image(image_path)
                sequence += 1
                yield (
                    image,
                    {
                        "kind": source,
                        "path": str(image_path),
                        "session": session,
                        "frame_sequence": sequence,
                        "source_started_monotonic": started,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            if not repeat_image:
                return
    elif source == "video":
        import cv2

        video = cv2.VideoCapture(str(path))
        try:
            if not video.isOpened():
                raise ValueError("无法打开视频")
            frame_number = 0
            while not stopped():
                started = time.perf_counter()
                ok, frame = video.read()
                if not ok:
                    return
                frame_number += 1
                yield (
                    frame,
                    {
                        "kind": "video",
                        "path": str(path),
                        "session": session,
                        "frame_sequence": frame_number,
                        "position_ms": video.get(cv2.CAP_PROP_POS_MSEC),
                        "source_started_monotonic": started,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        finally:
            video.release()
    elif source in {"window", "desktop"}:
        from .capture import CaptureSource

        timeout = float(parameters.get("source_timeout_seconds", 10))
        if not math.isfinite(timeout) or not 1 <= timeout <= 120:
            raise ValueError("来源等待超时应在 1–120 秒之间")
        with CaptureSource(
            source=source,
            hwnd=parameters.get("hwnd"),
            monitor_index=parameters.get("monitor_index", 1),
            client_only=parameters.get("client_only", True),
            max_fps=float(parameters.get("max_fps", 10)),
        ) as capture:
            last_status = None
            last_valid = time.perf_counter()
            while not stopped():
                started = time.perf_counter()
                item = capture.read(timeout=0.2)
                status = capture.status()
                if status["status"] != last_status:
                    emit("source_status", status)
                    last_status = status["status"]
                if status["status"] in {"closed", "unavailable"}:
                    raise RuntimeError("采集来源已关闭或不可用，请重新选择来源")
                if item is None:
                    if time.perf_counter() - last_valid >= timeout:
                        raise RuntimeError(f"采集来源持续无有效新帧（{last_status}），已停止保存")
                    time.sleep(0.02)
                    continue
                last_valid = time.perf_counter()
                frame, metadata = item
                metadata["source_started_monotonic"] = started
                metadata["dropped_frames"] = capture.queue.dropped
                yield frame, metadata
    else:
        raise ValueError("不支持的输入来源")


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def summarize_benchmark(inference_ms, end_to_end_ms, wall_seconds):
    if not inference_ms or len(inference_ms) != len(end_to_end_ms) or wall_seconds <= 0:
        raise ValueError("性能统计必须包含有效的实际测量样本")
    return {
        "iterations": len(inference_ms),
        "p50_ms": median(inference_ms),
        "p95_ms": _percentile(inference_ms, 0.95),
        "throughput_fps": len(inference_ms) / wall_seconds,
        "end_to_end_p50_ms": median(end_to_end_ms),
        "end_to_end_p95_ms": _percentile(end_to_end_ms, 0.95),
        "wall_seconds": wall_seconds,
    }


def _benchmark(parameters, run_dir, stopped, emit):
    warmup, iterations = int(parameters.get("warmup", 20)), int(parameters.get("iterations", 200))
    stability_seconds = float(parameters.get("stability_seconds", 0))
    if warmup < 20 or iterations < 200 or iterations > 100000 or not 0 <= stability_seconds <= 86400:
        raise ValueError("性能测试至少预热 20 次、测量 200 次；稳定性时长应在 0–86400 秒")
    with create_backend(parameters, run_dir / "backend") as backend:
        frames = _source_frames(parameters, stopped, emit, repeat_image=True)
        inference_ms, end_to_end_ms = [], []
        runtime = {}
        started = last_progress = None
        measured = 0
        try:
            for index, (image, source) in enumerate(frames):
                if stopped():
                    break
                result = backend.predict(
                    image, parameters.get("confidence", 0.5), parameters.get("iou", 0.45)
                )
                runtime = result["runtime"]
                now = time.perf_counter()
                if index >= warmup:
                    if started is None:
                        started = source["source_started_monotonic"]
                    measured += 1
                    inference_ms.append(result["elapsed_ms"])
                    end_to_end_ms.append((now - source["source_started_monotonic"]) * 1000)
                    if last_progress is None or now - last_progress > 1:
                        emit(
                            "progress",
                            {
                                "phase": "benchmark",
                                "completed": measured,
                                "total": iterations,
                                "elapsed_seconds": now - started,
                            },
                        )
                        last_progress = now
                    if measured >= iterations and now - started >= stability_seconds:
                        break
                elif index == 0 or index == warmup - 1:
                    emit("progress", {"phase": "warmup", "completed": index + 1, "total": warmup})
        finally:
            frames.close()
        if not inference_ms:
            if stopped():
                return {"cancelled": True, "iterations": 0, "runtime": runtime}
            raise ValueError("输入来源在预热期间结束，无法生成性能统计")
        if measured < iterations and not stopped():
            raise ValueError(f"输入来源不足 {iterations} 帧，无法完成规定的性能测量")
        metrics = summarize_benchmark(inference_ms, end_to_end_ms, time.perf_counter() - started)
        if metrics["wall_seconds"] < stability_seconds and not stopped():
            raise ValueError("输入来源提前结束，未达到规定的稳定性测量时长")
        metrics.update(
            {
                "kind": "benchmark",
                "warmup": warmup,
                "runtime": runtime,
                "model": backend.manifest.to_dict(),
                "parameters": parameters,
                "cancelled": stopped(),
                "completed": not stopped() and measured >= iterations,
                "metric_scope": "adapter elapsed includes preprocessing/inference/postprocessing; end-to-end includes source read; throughput includes source and loop overhead",
                "stability_seconds_requested": stability_seconds,
                "stability_completed": not stopped() and metrics["wall_seconds"] >= stability_seconds,
            }
        )
        output_path = run_dir / "benchmark.json"
        atomic_write(output_path, json.dumps(metrics, ensure_ascii=False, indent=2))
        metrics["output_path"] = str(output_path)
        emit("metrics", metrics)
        return metrics


def handle(request, emit):
    """Return one final result; outer worker owns terminal state/result events."""
    if request.get("protocol_version", 1) != 1:
        raise ValueError("不支持的推理 Worker 协议版本")
    kind = request["kind"]
    if kind not in {"infer", "infer_stream", "capture", "capture_stream", "benchmark"}:
        raise ValueError("未知推理/采集任务")
    parameters = dict(request.get("parameters", {}))
    run_dir = Path(request["run_dir"]).resolve()
    if any(part.casefold() == "input" for part in run_dir.parts):
        raise ValueError("input 目录只读，不能作为任务输出目录")
    run_dir.mkdir(parents=True, exist_ok=True)

    def stopped():
        return (run_dir / "stop.flag").exists()

    validate_thresholds(parameters.get("confidence", 0.5), parameters.get("iou", 0.45))
    if kind == "benchmark":
        return _benchmark(parameters, run_dir, stopped, emit)
    capture_only = kind.startswith("capture")
    streaming = kind.endswith("stream")
    max_fps = float(parameters.get("max_fps", 10))
    interval = float(parameters.get("interval_seconds", 5))
    max_frames = int(parameters.get("max_frames", 0))
    duration = float(parameters.get("duration_seconds", 0))
    if not math.isfinite(max_fps) or not 0 < max_fps <= 120 or not math.isfinite(interval) or interval < 0.1:
        raise ValueError("帧率需在 0–120，保存间隔至少 0.1 秒")
    if max_frames < 0 or not math.isfinite(duration) or duration < 0:
        raise ValueError("帧数和采集时长不能为负数")
    started = time.perf_counter()
    frame_count = saved_count = 0
    last_frame = last_result = None
    last_emit = last_saved = -math.inf
    results_path = run_dir / ("capture.jsonl" if capture_only else "detections.jsonl")
    with ExitStack() as stack:
        backend = (
            None if capture_only else stack.enter_context(create_backend(parameters, run_dir / "backend"))
        )
        if backend:
            emit("model", {"manifest": backend.manifest.to_dict(), "capabilities": backend.capabilities})
        writer = stack.enter_context(SharedFrameWriter())
        records = stack.enter_context(results_path.open("w", encoding="utf-8"))
        frames = _source_frames(parameters, stopped, emit)
        stack.callback(frames.close)
        for image, source in frames:
            if stopped():
                break
            result = (
                {
                    "detections": [],
                    "runtime": {"backend": source.get("capture_backend", source["kind"])},
                    "elapsed_ms": 0,
                }
                if backend is None
                else backend.predict(image, parameters.get("confidence", 0.5), parameters.get("iou", 0.45))
            )
            result.pop("raw_detections", None)
            if backend:
                result.update(
                    {
                        "model_sha256": backend.manifest.sha256,
                        "model_path": str(Path(parameters["model"]).resolve()),
                        "backend": parameters.get("backend", "cq"),
                        "family": backend.manifest.family,
                        "classes": backend.classes,
                        "confidence": parameters.get("confidence", 0.5),
                        "iou": parameters.get("iou", 0.45),
                    }
                )
            frame_count += 1
            result.update(
                {
                    "source": source,
                    "frame_number": frame_count,
                    "end_to_end_ms": (time.perf_counter() - source["source_started_monotonic"]) * 1000,
                }
            )
            last_frame, last_result = image, result
            now = time.perf_counter()
            if (
                capture_only
                and (not streaming or parameters.get("save_frames", False))
                and now - last_saved >= interval
            ):
                output_path = run_dir / f"capture-{saved_count + 1:06d}.png"
                save_image(output_path, image)
                result["output_path"] = str(output_path)
                atomic_write(
                    output_path.with_suffix(".json"), json.dumps(source, ensure_ascii=False, indent=2)
                )
                saved_count += 1
                last_saved = now
                emit(
                    "capture_saved",
                    {"output_path": str(output_path), "source": source, "saved_count": saved_count},
                )
            records.write(json.dumps(result, ensure_ascii=False) + "\n")
            records.flush()
            if now - last_emit >= 1 / max_fps or frame_count == 1:
                emit("frame", {**result, "frame": writer.write(image)})
                last_emit = now
            if capture_only and not streaming:
                break
            if parameters.get("source", "image") in {"window", "desktop"} and not streaming:
                break
            if max_frames and frame_count >= max_frames or duration and now - started >= duration:
                break
        final = {
            "kind": kind,
            "frame_count": frame_count,
            "saved_count": saved_count,
            "results_path": str(results_path),
            "cancelled": stopped(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if last_frame is not None:
            # File fallback makes one-shot and final display independent of shm lifetime.
            last_path = run_dir / "last-frame.png"
            save_image(last_path, last_frame)
            final.update(last_result)
            final["last_frame_path"] = str(last_path)
            if capture_only:
                final.setdefault("output_path", str(last_path))
            atomic_write(last_path.with_suffix(".json"), json.dumps(final, ensure_ascii=False, indent=2))
        elif not stopped():
            raise ValueError("没有收到有效图像，未保存任何截图")
        return final
