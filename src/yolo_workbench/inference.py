"""CQ_AI adapter. Instantiate only inside the dedicated inference environment."""

from __future__ import annotations

from importlib.metadata import distribution, version
from pathlib import Path

from .models import inspect_onnx, validate_contract
from .storage import atomic_write


class CqBackend:
    capabilities = {"confidence": True, "iou": False, "fixed_iou": 0.45, "single_session": True}

    def __init__(self, model_path: Path, classes: list[str], work_dir: Path, *, family="yolov8", device=2):
        if version("cq-ai-engine") != "0.14.6":
            raise RuntimeError("CQ_AI 运行环境必须固定为 0.14.6")
        if device not in {0, 1, 2, 3}:
            raise ValueError("未知推理设备")
        manifest, opset = inspect_onnx(model_path, family, classes)
        validate_contract(manifest, "cq", opset=opset)
        from ai_engine import Engine

        package = distribution("cq-ai-engine")
        dll = Path(package.locate_file("cq_ai_engine/_native/CQ_AI_x64.dll")).resolve()
        if not dll.is_file():
            raise RuntimeError("缺少配套 CQ_AI x64 DLL")
        work_dir.mkdir(parents=True, exist_ok=True)
        labels_path = (work_dir / "classes.txt").resolve()
        self.config_path = (work_dir / "cq_ai.ini").resolve()
        atomic_write(labels_path, "\n".join(classes) + "\n")
        atomic_write(self.config_path, f"yolo.labels_path={labels_path}\n")
        self.engine = Engine(dll_path=dll, auto_init=False)
        self.model = None
        try:
            self.model = self.engine.yolo_model(manifest.input_shape[2], device, 0, 1)
            self.model.load_model(model_path.resolve(), self.config_path)
        except BaseException:
            self.close()
            raise
        self.manifest = manifest

    def predict(self, bgr_image, confidence=0.5) -> dict:
        import math
        import time

        from ai_engine import image_from_numpy

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("置信度必须位于 0 到 1")
        if self.model is None:
            raise RuntimeError("模型已释放")
        started = time.perf_counter()
        detections = self.model.infer(image_from_numpy(bgr_image), confidence)
        # Preserve existing engine semantics; do not apply a second NMS.
        return {
            "detections": detections,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "engine_latency_us": self.model.last_latency_us(),
            "runtime": self.model.runtime_status(),
        }

    def close(self):
        try:
            if self.model is not None:
                self.model.release()
                self.model = None
        finally:
            self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
