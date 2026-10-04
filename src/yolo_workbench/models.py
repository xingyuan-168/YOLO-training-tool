"""Export contracts are checked before any native loader is invoked."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelManifest:
    family: str
    format: str
    classes: list[str]
    input_shape: list
    output_shapes: list[list]
    sha256: str
    precision: str = "FP32"
    task: str = "detect"
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def validate_contract(manifest: ModelManifest, profile: str, *, opset: int | None = None) -> None:
    if manifest.task != "detect" or not manifest.classes:
        raise ValueError("仅支持带类别映射的目标检测模型")
    if profile == "generic":
        return
    if profile not in {"cq", "ascript_v8"}:
        raise ValueError("未知导出契约")
    if manifest.family not in ({"yolov8", "yolo11"} if profile == "cq" else {"yolov8"}):
        raise ValueError("模型系列不兼容当前目标端，请选择通用导出")
    shape = manifest.input_shape
    if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3]:
        raise ValueError("需要 Batch=1、三通道、固定正方形输入")
    size = shape[2]
    if size not in ({320, 640} if profile == "cq" else {640}):
        raise ValueError("输入尺寸不符合目标端契约")
    if manifest.precision != "FP32":
        raise ValueError("首版契约仅验收 FP32")
    candidates = sum((size // stride) ** 2 for stride in (8, 16, 32))
    expected = (
        [1, 4 + len(manifest.classes), candidates] if profile == "cq" else [4 + len(manifest.classes), 8400]
    )
    if manifest.output_shapes != [expected]:
        raise ValueError(f"输出布局不兼容，需要单输出 {expected}，实际 {manifest.output_shapes}")
    if profile == "cq" and (manifest.format != "onnx" or opset != 12):
        raise ValueError("CQ_AI 需要 ONNX opset 12")
    if profile == "ascript_v8" and manifest.format != "ncnn":
        raise ValueError("AScript v8 需要 NCNN param/bin")


def inspect_onnx(path: Path, family: str, classes: list[str]) -> tuple[ModelManifest, int]:
    import onnx

    from .dataset import file_hash

    model = onnx.load(str(path), load_external_data=False)
    inputs = [v for v in model.graph.input if v.name not in {i.name for i in model.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError("首版只接受单图像输入模型")

    def dimensions(value):
        return [
            d.dim_value if d.HasField("dim_value") else d.dim_param or None
            for d in value.type.tensor_type.shape.dim
        ]

    float32 = all(
        v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in inputs + list(model.graph.output)
    )
    manifest = ModelManifest(
        family=family,
        format="onnx",
        classes=classes,
        input_shape=dimensions(inputs[0]),
        output_shapes=[dimensions(o) for o in model.graph.output],
        sha256=file_hash(path),
        precision="FP32" if float32 else "other",
    )
    opset = next((v.version for v in model.opset_import if v.domain in ("", "ai.onnx")), 0)
    return manifest, opset
