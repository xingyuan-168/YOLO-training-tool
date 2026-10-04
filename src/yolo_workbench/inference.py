"""Deployment adapters. Native libraries are imported only in worker processes."""

from __future__ import annotations

import math
import time
from importlib.metadata import distribution, version
from pathlib import Path

from .models import (
    ModelManifest,
    _names,
    inspect_ncnn,
    inspect_onnx,
    model_classes,
    resolve_model,
    validate_contract,
)
from .storage import atomic_write


def validate_thresholds(confidence, iou=0.45):
    for name, value in (("置信度", confidence), ("IoU", iou)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name}必须位于 0 到 1")


def _image(bgr_image):
    import numpy as np

    if not isinstance(bgr_image, np.ndarray) or bgr_image.dtype != np.uint8:
        raise ValueError("输入必须为 uint8 BGR 图像")
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3 or min(bgr_image.shape[:2]) < 1:
        raise ValueError("输入必须为非空三通道 BGR 图像")
    return np.ascontiguousarray(bgr_image)


def letterbox(bgr_image, size: int):
    """Ultralytics fixed square RGB/CHW preprocessing and reversible geometry."""
    import cv2
    import numpy as np

    image = _image(bgr_image)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    left, top = round((size - resized_w) / 2 - 0.1), round((size - resized_h) / 2 - 0.1)
    right, bottom = size - resized_w - left, size - resized_h - top
    if (width, height) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, {"scale": scale, "left": left, "top": top, "width": width, "height": height}


def _detections(boxes, scores, ids, classes, geometry):
    import numpy as np

    boxes = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    boxes[:, [0, 2]] = np.clip(
        (boxes[:, [0, 2]] - geometry["left"]) / geometry["scale"], 0, geometry["width"]
    )
    boxes[:, [1, 3]] = np.clip(
        (boxes[:, [1, 3]] - geometry["top"]) / geometry["scale"], 0, geometry["height"]
    )
    result = []
    for box, score, class_id in zip(boxes, scores, ids, strict=True):
        class_id = int(class_id)
        if not 0 <= class_id < len(classes):
            raise ValueError("模型输出类别编号越界，类别映射与模型不兼容")
        if not np.isfinite(box).all() or not math.isfinite(float(score)):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result.append(
            {
                "xyxy": [float(v) for v in box],
                "class_id": class_id,
                "class_name": classes[class_id],
                "confidence": float(score),
            }
        )
    return result


def nms(boxes, scores, ids, iou=0.45, max_detections=300):
    """Class-aware NMS for generic decoded outputs; never used for CQ."""
    import numpy as np

    order = np.argsort(-scores, kind="stable")[:30000]
    kept = []
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    while order.size and len(kept) < max_detections:
        first, rest = int(order[0]), order[1:]
        kept.append(first)
        intersection = np.maximum(
            np.minimum(boxes[first, 2:], boxes[rest, 2:]) - np.maximum(boxes[first, :2], boxes[rest, :2]), 0
        ).prod(axis=1)
        overlap = intersection / np.maximum(areas[first] + areas[rest] - intersection, 1e-9)
        order = rest[(overlap <= iou) | (ids[rest] != ids[first])]
    return np.asarray(kept, dtype=np.int64)


def decode_output(output, classes, geometry, confidence=0.5, iou=0.45, *, end_to_end=False):
    """Decode [4+C,N] or an explicitly selected [N,6] end-to-end layout."""
    import numpy as np

    validate_thresholds(confidence, iou)
    array = np.asarray(output)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"不支持的检测输出维度：{array.shape}")
    if end_to_end:
        if array.shape[1] != 6:
            raise ValueError("端到端输出必须为 [N,6]")
        selected = array[np.isfinite(array).all(axis=1) & (array[:, 4] >= confidence)]
        if selected.size and not np.all(selected[:, 5] == np.floor(selected[:, 5])):
            raise ValueError("端到端类别编号必须为整数")
        return _detections(selected[:, :4], selected[:, 4], selected[:, 5], classes, geometry)
    if array.shape[0] != 4 + len(classes):
        raise ValueError(f"检测输出与冻结类别不匹配：需要 [{4 + len(classes)},N]，实际 {array.shape}")
    rows = array.T
    ids = np.argmax(rows[:, 4:], axis=1)
    scores = rows[np.arange(len(rows)), 4 + ids]
    valid = np.isfinite(rows).all(axis=1) & (scores >= confidence)
    rows, scores, ids = rows[valid], scores[valid], ids[valid]
    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, :2], boxes[:, 2:] = rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2
    kept = nms(boxes, scores, ids, iou)
    return _detections(boxes[kept], scores[kept], ids[kept], classes, geometry)


class _Backend:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CqBackend(_Backend):
    capabilities = {"confidence": True, "iou": False, "fixed_iou": 0.45, "single_session": True}
    _active = False

    def __init__(self, model_path: Path, classes: list[str], work_dir: Path, *, family="yolov8", device=2):
        self.engine = self.model = None
        self._owns_engine = False
        if CqBackend._active:
            raise RuntimeError("CQ_AI 同一工作进程只能拥有一个引擎会话")
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
        CqBackend._active = self._owns_engine = True
        try:
            self.model = self.engine.yolo_model(manifest.input_shape[2], device, 0, 1)
            self.model.load_model(model_path.resolve(), self.config_path)
        except BaseException:
            self.close()
            raise
        self.manifest = manifest
        self.classes = list(classes)

    def predict(self, bgr_image, confidence=0.5, iou=0.45) -> dict:
        from ai_engine import image_from_numpy

        validate_thresholds(confidence, iou)
        if iou != 0.45:
            raise ValueError("CQ_AI 原生 IoU 固定为 0.45")
        image = _image(bgr_image)
        if self.model is None:
            raise RuntimeError("模型已释放")
        started = time.perf_counter()
        raw = self.model.infer(image_from_numpy(image), confidence)
        # Preserve native letterbox, cross-class NMS and order exactly.
        detections = []
        for box in raw:
            class_id = int(box["class_id"])
            if not 0 <= class_id < len(self.classes):
                raise ValueError("CQ_AI 输出类别编号与冻结类别不兼容")
            detections.append(
                {
                    "xyxy": [float(box[key]) for key in ("x1", "y1", "x2", "y2")],
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "confidence": float(box["score"]),
                }
            )
        return {
            "detections": detections,
            "raw_detections": raw,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "engine_latency_us": self.model.last_latency_us(),
            "runtime": self.model.runtime_status(),
            "capabilities": self.capabilities,
        }

    def close(self):
        try:
            if self.model is not None:
                self.model.release()
        finally:
            self.model = None
            try:
                if self.engine is not None:
                    self.engine.close()
            finally:
                self.engine = None
                if self._owns_engine:
                    CqBackend._active = self._owns_engine = False


class NcnnBackend(_Backend):
    capabilities = {"confidence": True, "iou": True, "single_session": True}

    def __init__(self, model_path, classes, *, family="yolov8", input_size=640, device="cpu", metadata=None):
        if str(device).lower() not in {"cpu", "auto"}:
            raise ValueError("首版 NCNN 部署适配器仅验收 CPU，请选择 CPU")
        self.manifest, self.input_name, self.output_name = inspect_ncnn(
            model_path, family, classes, input_size=input_size, metadata=metadata
        )
        profile = (metadata or {}).get("profile", "ascript_v8")
        validate_contract(self.manifest, profile)
        self.end_to_end = family == "yolo26" and self.manifest.output_shapes[0][-1] == 6
        self.capabilities = {
            "confidence": True,
            "iou": not self.end_to_end,
            "single_session": True,
            "end_to_end": self.end_to_end,
        }
        import ncnn

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_packing_layout = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_bf16_storage = False
        self.net.opt.num_threads = 4
        if (
            self.net.load_param(str(model_path)) != 0
            or self.net.load_model(str(model_path.with_suffix(".bin"))) != 0
        ):
            self.close()
            raise ValueError("NCNN 模型加载失败")
        self.classes, self.input_size = list(classes), input_size
        import numpy as np

        output = self._forward(np.zeros((3, input_size, input_size), dtype=np.float32))
        expected = tuple(self.manifest.output_shapes[0])
        if output.shape != expected:
            self.close()
            raise ValueError(f"NCNN 输出布局不兼容：需要 {expected}，实际 {output.shape}")

    def _forward(self, pixels):
        import ncnn
        import numpy as np

        # Both owners must survive extract: Mat can borrow numpy memory.
        owner = np.ascontiguousarray(pixels, dtype=np.float32)
        tensor = ncnn.Mat(owner)
        extractor = self.net.create_extractor()
        if extractor.input(self.input_name, tensor) != 0:
            raise RuntimeError("NCNN 输入失败")
        code, output = extractor.extract(self.output_name)
        if code != 0:
            raise RuntimeError("NCNN 推理失败")
        return np.asarray(output).copy()

    def predict(self, bgr_image, confidence=0.5, iou=0.45):
        validate_thresholds(confidence, iou)
        if self.net is None:
            raise RuntimeError("模型已释放")
        started = time.perf_counter()
        tensor, geometry = letterbox(bgr_image, self.input_size)
        inference_start = time.perf_counter()
        output = self._forward(tensor)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        detections = decode_output(
            output, self.classes, geometry, confidence, iou, end_to_end=self.end_to_end
        )
        return {
            "detections": detections,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "inference_ms": inference_ms,
            "runtime": {"backend": "ncnn", "active": "cpu", "version": version("ncnn"), "precision": "fp32"},
            "capabilities": self.capabilities,
        }

    def close(self):
        if getattr(self, "net", None) is not None:
            self.net.clear()
            self.net = None


class OnnxBackend(_Backend):
    def __init__(self, model_path, classes, *, family="yolov8", input_size=640, device="cpu"):
        self.manifest, _ = inspect_onnx(model_path, family, classes)
        validate_contract(self.manifest, "generic")
        if self.manifest.precision not in {"FP32", "FP16"}:
            raise ValueError("ONNX 输入输出需要统一的 FP32 或 FP16 类型")
        shape, outputs = self.manifest.input_shape, self.manifest.output_shapes
        if len(shape) != 4 or shape[0] not in (1, None, "batch") or shape[1] != 3:
            raise ValueError("ONNX 必须为 Batch=1 三通道 NCHW 检测输入")
        if len(outputs) != 1 or len(outputs[0]) != 3 or outputs[0][0] not in (1, None, "batch"):
            raise ValueError("ONNX 需要一个检测输出，分割/原始多头模型不受支持")
        self.end_to_end = outputs[0][-1] == 6 and outputs[0][1] != 4 + len(classes)
        if not self.end_to_end and outputs[0][1] != 4 + len(classes):
            raise ValueError("ONNX 输出布局与冻结类别不兼容")
        sizes = [value for value in shape[2:] if isinstance(value, int)]
        if sizes and (len(sizes) != 2 or sizes[0] != sizes[1]):
            raise ValueError("首版通用 ONNX 适配器需要正方形输入")
        self.input_size = sizes[0] if sizes else input_size
        self.classes = list(classes)
        import onnxruntime as ort

        available = ort.get_available_providers()
        requested = str(device).lower()
        if requested == "cpu":
            providers = ["CPUExecutionProvider"]
        elif requested == "auto":
            providers = [
                name
                for name in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                if name in available
            ]
        else:
            if "CUDAExecutionProvider" not in available:
                raise ValueError("当前 ONNX Runtime 未提供 CUDA，请选择 CPU")
            providers = [("CUDAExecutionProvider", {"device_id": int(requested)}), "CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        model_input = self.session.get_inputs()[0]
        if model_input.type not in {"tensor(float)", "tensor(float16)"}:
            self.close()
            raise ValueError("ONNX 输入类型需要 FP32 或 FP16")
        self.input_name, self.input_dtype = model_input.name, model_input.type
        self.capabilities = {"confidence": True, "iou": not self.end_to_end, "end_to_end": self.end_to_end}
        self.runtime = {
            "backend": "onnxruntime",
            "providers": self.session.get_providers(),
            "active": self.session.get_providers()[0],
            "requested": requested,
            "version": version("onnxruntime"),
            "precision": "fp16" if "float16" in self.input_dtype else "fp32",
        }

    def predict(self, bgr_image, confidence=0.5, iou=0.45):
        import numpy as np

        validate_thresholds(confidence, iou)
        if self.session is None:
            raise RuntimeError("模型已释放")
        started = time.perf_counter()
        tensor, geometry = letterbox(bgr_image, self.input_size)
        tensor = tensor[None].astype(np.float16 if "float16" in self.input_dtype else np.float32, copy=False)
        inference_start = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        inference_ms = (time.perf_counter() - inference_start) * 1000
        detections = decode_output(
            output, self.classes, geometry, confidence, iou, end_to_end=self.end_to_end
        )
        return {
            "detections": detections,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "inference_ms": inference_ms,
            "runtime": self.runtime,
            "capabilities": self.capabilities,
        }

    def close(self):
        self.session = None


class PtBackend(_Backend):
    def __init__(self, model_path, classes=None, *, family="yolov8", input_size=640, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="detect")
        if self.model.task != "detect":
            raise ValueError("仅支持检测模型")
        self.classes = _names(self.model.names)
        if classes and classes != self.classes:
            raise ValueError("当前类别与 PT 模型冻结类别不一致")
        if str(device).lower() == "auto":
            import torch

            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)
        self.input_size = input_size
        self.end_to_end = bool(getattr(self.model.model, "end2end", False))
        self.capabilities = {"confidence": True, "iou": not self.end_to_end, "end_to_end": self.end_to_end}
        from .dataset import file_hash

        self.manifest = ModelManifest(
            family, "pt", self.classes, [1, 3, input_size, input_size], [], file_hash(model_path)
        )

    def predict(self, bgr_image, confidence=0.5, iou=0.45):
        validate_thresholds(confidence, iou)
        image = _image(bgr_image)
        started = time.perf_counter()
        results = self.model.predict(
            image,
            imgsz=self.input_size,
            device=self.device,
            conf=confidence,
            iou=iou,
            verbose=False,
            save=False,
        )
        result = results[0]
        boxes = result.boxes
        geometry = {"scale": 1, "left": 0, "top": 0, "width": image.shape[1], "height": image.shape[0]}
        detections = _detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            self.classes,
            geometry,
        )
        return {
            "detections": detections,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "inference_ms": result.speed.get("inference"),
            "runtime": {
                "backend": "ultralytics",
                "active": str(next(self.model.model.parameters()).device),
                "version": version("ultralytics"),
                "precision": str(next(self.model.model.parameters()).dtype),
            },
            "capabilities": self.capabilities,
        }

    def close(self):
        self.model = None


def create_backend(parameters: dict, work_dir: Path):
    backend = parameters.get("backend", "cq")
    path, metadata = resolve_model(Path(parameters["model"]), backend)
    family = metadata.get("family") or parameters.get("family", "yolov8")
    if family not in {"yolov8", "yolo11", "yolo26"}:
        raise ValueError("只支持 YOLOv8、YOLO11、YOLO26 检测模型")
    size = int(parameters.get("input_size", 640))
    if size < 32 or size > 4096 or size % 32:
        raise ValueError("输入尺寸必须为 32 的倍数，范围 32–4096")
    device = parameters.get("device", "cpu")
    classes = None
    if backend != "pt" or parameters.get("classes") or metadata.get("classes") or metadata.get("names"):
        classes = model_classes(path, parameters.get("classes"), metadata)
    if backend == "cq":
        devices = {"auto": 0, "directml": 1, "cpu": 2, "tensorrt": 3, "cuda": 3}
        if isinstance(device, str):
            if device.lower() not in devices:
                raise ValueError("CQ_AI 设备需为 cpu/auto/directml/tensorrt")
            device = devices[device.lower()]
        return CqBackend(path, classes, work_dir, family=family, device=device)
    if backend == "ncnn":
        frozen_shape = metadata.get("input_shape")
        if frozen_shape:
            size = frozen_shape[2]
        return NcnnBackend(path, classes, family=family, input_size=size, device=device, metadata=metadata)
    if backend == "onnx":
        return OnnxBackend(path, classes, family=family, input_size=size, device=device)
    return PtBackend(path, classes, family=family, input_size=size, device=device)
