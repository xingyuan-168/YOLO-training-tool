"""Real bounded-stream stability check; never captures unrelated desktop content.

Run with the inference environment and the repository's src on PYTHONPATH.
Example: python scripts/soak_inference.py --repo-root . --seconds 1800
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from yolo_workbench.capture import LatestFrameQueue
from yolo_workbench.frames import SharedFrameReader, SharedFrameWriter
from yolo_workbench.inference import create_backend
from yolo_workbench.storage import atomic_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    report_path = (args.report or root / ".artifacts/inference-soak/report.json").resolve()
    if args.seconds <= 0 or any(part.casefold() == "input" for part in report_path.parts):
        raise ValueError("Duration must be positive and report must be outside read-only input")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model = (args.model or root / "input/模型样板/best.onnx").resolve()
    parameters = {"backend": "cq", "model": str(model), "device": "cpu"}
    process = psutil.Process()
    queue = LatestFrameQueue()
    stop = threading.Event()
    frames = []
    for height, width in ((320, 320), (240, 480)):
        yy, xx = np.indices((height, width))
        frames.append(np.stack(((xx % 256), (yy % 256), ((xx + yy) % 256)), axis=2).astype(np.uint8))
    report = {
        "schema_version": 1,
        "source": "owned_generated_image_stream",
        "requested_seconds": args.seconds,
        "state": "starting",
        "backend": "cq",
        "device": "cpu",
        "queue_capacity": 1,
        "frame_slots_per_mapping": 2,
        "max_mapping_count": 2,
        "business_accuracy_validated": False,
        "wgc_coverage": "separate owned-window test",
        "report_path": str(report_path),
    }
    producer = None
    started = None
    try:
        with (
            create_backend(parameters, report_path.parent / "backend") as backend,
            SharedFrameWriter() as writer,
            SharedFrameReader() as reader,
        ):
            for _ in range(20):
                backend.predict(frames[0])
            start_rss = peak_rss = process.memory_info().rss
            start_handles = process.num_handles() if hasattr(process, "num_handles") else 0
            peak_handles = start_handles
            count = missing = max_mappings = max_capacity = 0
            inference_sum = e2e_sum = 0.0
            started = time.perf_counter()
            last_report = last_sample = started

            def produce():
                sequence = 0
                while not stop.is_set():
                    pixels = frames[(sequence // 300) % len(frames)].copy()
                    pixels[:8, :8] = sequence % 256
                    queue.put((pixels, time.perf_counter()))
                    sequence += 1
                    stop.wait(1 / 30)

            producer = threading.Thread(target=produce, daemon=True)
            producer.start()
            while time.perf_counter() - started < args.seconds:
                if (report_path.parent / "stop.flag").exists():
                    report["state"] = "stopped"
                    break
                item = queue.get(0.25)
                if item is None:
                    missing += 1
                    continue
                image, received = item
                prediction = backend.predict(image, confidence=0.5)
                reference = writer.write(image)
                displayed = reader.read(reference)
                if displayed is None or displayed["data"] != image.tobytes():
                    raise AssertionError("Shared frame was lost or corrupted")
                count += 1
                inference_sum += prediction["elapsed_ms"]
                e2e_sum += (time.perf_counter() - received) * 1000
                max_mappings = max(
                    max_mappings, int(writer._memory is not None) + int(writer._retired is not None)
                )
                max_capacity = max(max_capacity, writer.capacity)
                now = time.perf_counter()
                if now - last_sample >= 1 or count == 1:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    if hasattr(process, "num_handles"):
                        peak_handles = max(peak_handles, process.num_handles())
                    last_sample = now
                if now - last_report >= 60 or count == 1:
                    report.update(
                        {
                            "state": "running",
                            "elapsed_seconds": now - started,
                            "frame_count": count,
                            "dropped_frames": queue.dropped,
                            "no_frame_count": missing,
                            "rss_start_bytes": start_rss,
                            "rss_peak_bytes": peak_rss,
                            "rss_current_bytes": process.memory_info().rss,
                            "handles_start": start_handles,
                            "handles_peak": peak_handles,
                            "mapping_count_observed": max_mappings,
                            "mapping_capacity_bytes": max_capacity,
                            "runtime": prediction["runtime"],
                        }
                    )
                    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2))
                    print(
                        json.dumps(
                            {
                                "elapsed_seconds": round(now - started),
                                "frame_count": count,
                                "rss_mib": round(process.memory_info().rss / 2**20, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    last_report = now
            elapsed = time.perf_counter() - started
            final_rss = process.memory_info().rss
            final_handles = process.num_handles() if hasattr(process, "num_handles") else 0
            completed = elapsed >= args.seconds
            checks = {
                "duration_completed": completed,
                "valid_frames": count > 0,
                "bounded_mappings": max_mappings <= 2,
                "rss_growth_below_64mib": final_rss - start_rss <= 64 * 2**20,
                "handle_growth_below_32": final_handles - start_handles <= 32,
            }
            report.update(
                {
                    "state": "passed" if all(checks.values()) else "stopped" if not completed else "failed",
                    "elapsed_seconds": elapsed,
                    "frame_count": count,
                    "dropped_frames": queue.dropped,
                    "no_frame_count": missing,
                    "rss_start_bytes": start_rss,
                    "rss_peak_bytes": peak_rss,
                    "rss_final_bytes": final_rss,
                    "handles_start": start_handles,
                    "handles_peak": peak_handles,
                    "handles_final": final_handles,
                    "mapping_count_observed": max_mappings,
                    "mapping_capacity_bytes": max_capacity,
                    "checks": checks,
                    "throughput_fps": count / elapsed,
                    "mean_adapter_ms": inference_sum / max(1, count),
                    "mean_source_to_display_ms": e2e_sum / max(1, count),
                }
            )
    except BaseException as error:
        report.update({"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        stop.set()
        queue.close()
        if producer:
            producer.join(timeout=2)
        if started:
            report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
