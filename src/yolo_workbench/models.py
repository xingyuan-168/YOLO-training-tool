"""Export contracts are checked before any native loader is invoked."""

from __future__ import annotations

import ast
import hashlib
import json
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
    validate_classes(manifest.classes)
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
    float16 = all(
        v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 for v in inputs + list(model.graph.output)
    )
    manifest = ModelManifest(
        family=family,
        format="onnx",
        classes=classes,
        input_shape=dimensions(inputs[0]),
        output_shapes=[dimensions(o) for o in model.graph.output],
        sha256=file_hash(path),
        precision="FP32" if float32 else "FP16" if float16 else "other",
    )
    opset = next((v.version for v in model.opset_import if v.domain in ("", "ai.onnx")), 0)
    return manifest, opset


def validate_classes(classes: list[str]) -> list[str]:
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(name, str) or not name.strip() for name in classes)
    ):
        raise ValueError("模型必须包含非空的冻结类别名称列表")
    if len(set(classes)) != len(classes):
        raise ValueError("模型类别名称不能重复")
    return classes


def _names(value) -> list[str]:
    if isinstance(value, dict):
        try:
            pairs = sorted((int(key), name) for key, name in value.items())
        except (ValueError, TypeError) as error:
            raise ValueError("非法模型类别编号") from error
        if [key for key, _ in pairs] != list(range(len(pairs))):
            raise ValueError("模型类别编号必须从 0 连续排列")
        value = [name for _, name in pairs]
    return validate_classes(value)


def resolve_model(path: Path, backend: str) -> tuple[Path, dict]:
    """Resolve one model in a package, without loading a native runtime."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise ValueError(f"模型不存在：{path}")
    explicit_manifest = path.is_file() and path.suffix.lower() == ".json"
    directory = path if path.is_dir() else path.parent
    metadata = {}
    candidates = (
        [path]
        if explicit_manifest
        else (
            []
            if path.is_dir()
            else [path.with_name(path.name + ".manifest.json"), path.with_suffix(".manifest.json")]
        )
        + [directory / "manifest.json", directory / "model_manifest.json"]
    )
    for candidate in candidates:
        if candidate.is_file():
            metadata = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if not isinstance(metadata, dict) or metadata.get("schema_version", 1) != 1:
                raise ValueError("不支持的模型清单版本")
            break
    suffix = {"cq": ".onnx", "onnx": ".onnx", "pt": ".pt", "ncnn": ".param"}.get(backend)
    if suffix is None:
        raise ValueError("未知推理后端")
    if path.is_dir() or explicit_manifest:
        # Never follow a package's path outside its directory.
        declared = metadata.get("model_path") or metadata.get("model_file")
        if not declared and explicit_manifest and path.name.endswith(".manifest.json"):
            stem = path.name[: -len(".manifest.json")]
            candidate = directory / (stem if stem.endswith(suffix) else stem + suffix)
            if candidate.is_file():
                declared = candidate.name
        if declared:
            if not isinstance(declared, str):
                raise ValueError("模型清单路径必须为字符串")
            selected = (directory / declared).resolve()
            if not selected.is_relative_to(directory) or not selected.is_file():
                raise ValueError("模型清单路径越界或文件不存在")
            path = selected
        else:
            candidates = sorted(directory.glob(f"*{suffix}"))
            if len(candidates) != 1:
                raise ValueError("模型目录需包含唯一匹配模型，或在 manifest.json 指定 model_path")
            path = candidates[0]
    if path.suffix.lower() != suffix:
        raise ValueError(f"当前后端需要 {suffix} 模型")
    if backend == "ncnn" and not path.with_suffix(".bin").is_file():
        raise ValueError("NCNN 模型缺少配对的 .bin 权重")
    return path, metadata


def model_classes(path: Path, supplied=None, metadata=None) -> list[str]:
    """Frozen package/model names win; a conflicting UI mapping is rejected."""
    metadata = metadata or {}
    frozen = metadata.get("classes") or metadata.get("names")
    if frozen is None:
        for name in ("classes.txt", "labels.txt"):
            candidate = path.parent / name
            if candidate.is_file():
                frozen = candidate.read_text(encoding="utf-8-sig").splitlines()
                break
    if frozen is None and path.suffix.lower() == ".onnx":
        import onnx

        graph = onnx.load(str(path), load_external_data=False)
        props = {prop.key: prop.value for prop in graph.metadata_props}
        if "names" in props:
            try:
                frozen = ast.literal_eval(props["names"])
            except (ValueError, SyntaxError) as error:
                raise ValueError("ONNX 类别元数据无法解析") from error
    frozen = _names(frozen) if frozen is not None else None
    supplied = _names(supplied) if supplied else None
    if supplied and frozen and supplied != frozen:
        raise ValueError("当前类别与模型冻结类别不一致，请使用模型类别映射")
    return validate_classes(frozen or supplied)


def inspect_ncnn(
    path: Path, family: str, classes: list[str], *, input_size=640, metadata=None
) -> tuple[ModelManifest, str, str]:
    """Check text graph topology before invoking NCNN. Raw heads are unsupported."""
    validate_classes(classes)
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != "7767517":
        raise ValueError("无效的 NCNN param 文件")
    inputs, produced, consumed = [], [], set()
    try:
        layer_count, blob_count = map(int, lines[1].split())
        if layer_count != len(lines) - 2 or blob_count < 1:
            raise ValueError("NCNN 图层计数不匹配")
        for line in lines[2:]:
            fields = line.split()
            bottoms, tops = int(fields[2]), int(fields[3])
            if len(fields) < 4 + bottoms + tops:
                raise ValueError("NCNN 图层字段不完整")
            consumed.update(fields[4 : 4 + bottoms])
            outputs = fields[4 + bottoms : 4 + bottoms + tops]
            produced.extend(outputs)
            if fields[0] == "Input":
                inputs.extend(outputs)
    except (IndexError, TypeError) as error:
        raise ValueError("无效的 NCNN 图层结构") from error
    outputs = [name for name in produced if name not in consumed]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("NCNN 仅支持已解码单输出；原始多检测头模型不兼容，请重新导出")
    metadata = metadata or {}
    shapes = metadata.get("output_shapes") or [
        [4 + len(classes), sum((input_size // s) ** 2 for s in (8, 16, 32))]
    ]
    if (
        len(shapes) != 1
        or len(shapes[0]) != 2
        or any(type(value) is not int or value < 1 for value in shapes[0])
    ):
        raise ValueError("NCNN 清单需要一个固定二维检测输出")
    expected = [4 + len(classes), sum((input_size // s) ** 2 for s in (8, 16, 32))]
    decoded = shapes[0] == expected
    end_to_end = family == "yolo26" and shapes[0][-1] == 6
    if not decoded and not end_to_end:
        raise ValueError("NCNN 检测输出与模型类别或端到端布局不兼容")
    # YOLO26 removes DFL/Softmax; all families must still have one terminal output.
    has_decode = family == "yolo26" or (
        any(line.startswith("Softmax ") for line in lines) and lines[-1].startswith("Concat ")
    )
    if not has_decode:
        raise ValueError("NCNN 图未发现已解码 YOLO 输出，请使用支持的通用 ONNX/PT 或 AScript 包")
    digest = hashlib.sha256()
    for file in (path, path.with_suffix(".bin")):
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    manifest = ModelManifest(
        family,
        "ncnn",
        classes,
        [1, 3, input_size, input_size],
        shapes,
        digest.hexdigest(),
        metadata.get("precision", "FP32"),
    )
    return manifest, inputs[0], outputs[0]
