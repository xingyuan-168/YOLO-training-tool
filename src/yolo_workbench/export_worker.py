"""Non-overwriting export packages, structural checks, and isolated execution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from .models import ModelManifest, inspect_onnx, validate_contract
from .storage import atomic_write, json_text
from .training import resolve_model
from .training_worker import digest


def _need(*names):
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"导出环境缺少 {', '.join(missing)}；请通过环境准备工具显式安装后重试")


def _ncnn_names(path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != "7767517":
        raise ValueError("非法 NCNN param 头")
    inputs, produced, consumed = [], set(), set()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError("非法 NCNN layer")
        bottoms, tops = int(parts[2]), int(parts[3])
        consumed.update(parts[4 : 4 + bottoms])
        output = parts[4 + bottoms : 4 + bottoms + tops]
        produced.update(output)
        if parts[0] == "Input":
            inputs.extend(output)
    outputs = produced - consumed
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("首版 NCNN 包需要单输入、单解码输出；不支持原始多头输出")
    return inputs[0], next(iter(outputs))


def validate_execution(spec: dict) -> dict:
    """Executed in a fresh native process, never in the GUI or export process."""
    import numpy as np

    folder = Path(spec["folder"])
    size, format_name = spec["imgsz"], spec["format"]
    # Patterned deterministic input exercises execution, but is not accuracy
    # or target-device acceptance evidence.
    pixels = np.linspace(0.0, 1.0, 3 * size * size, dtype=np.float32).reshape(1, 3, size, size)
    start = time.perf_counter()
    if format_name == "onnx":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        runtime = ort.InferenceSession(
            str(folder / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        outputs = runtime.run(None, {runtime.get_inputs()[0].name: pixels})
        backend = "onnxruntime_cpu"
    elif format_name == "pt":
        from .worker import prepare_ultralytics

        prepare_ultralytics()
        import torch
        from ultralytics import YOLO

        model = YOLO(str(folder / "model.pt"), task="detect")
        model.model.cpu().eval()
        with torch.inference_mode():
            output = model.model(torch.from_numpy(pixels))
        output = output[0] if isinstance(output, (tuple, list)) else output
        outputs = [output.cpu().numpy()]
        backend = "pytorch_cpu"
    elif format_name == "ncnn":
        import ncnn

        input_name, output_name = _ncnn_names(folder / "model.ncnn.param")
        network = ncnn.Net()
        network.opt.use_vulkan_compute = False
        network.opt.use_packing_layout = False
        network.opt.num_threads = 2
        if network.load_param(str(folder / "model.ncnn.param")) or network.load_model(
            str(folder / "model.ncnn.bin")
        ):
            raise RuntimeError("NCNN 无法加载生成的 param/bin")
        extractor = network.create_extractor()
        planar = np.ascontiguousarray(pixels[0])
        tensor = ncnn.Mat(planar)
        if extractor.input(input_name, tensor):
            raise RuntimeError("NCNN 输入张量失败")
        code, output = extractor.extract(output_name)
        if code:
            raise RuntimeError(f"NCNN 执行失败：{code}")
        outputs = [np.asarray(output).copy()]
        backend = "ncnn_cpu"
    else:
        raise ValueError("不支持的验证格式")
    if not outputs or any(array.size == 0 or not np.isfinite(array).all() for array in outputs):
        raise ValueError("模型输出为空或包含非有限值")
    return {
        "passed": True,
        "backend": backend,
        "input_shape": list(pixels.shape),
        "output_shapes": [list(array.shape) for array in outputs],
        "elapsed_seconds": time.perf_counter() - start,
        "input": "deterministic_ramp",
        "scope": "isolated_cpu_execution_only",
        "target_device_accepted": False,
    }


def _independent_check(folder: Path, spec: dict, run_dir: Path) -> dict:
    validation_request = {"folder": str(folder), **spec}
    request_path = folder / "validation-request.json"
    atomic_write(request_path, json_text(validation_request))
    log_path = folder / "validation.log"
    with log_path.open("w", encoding="utf-8") as stream:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "yolo_workbench.export_worker", "--validate", str(request_path)],
                cwd=run_dir,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("独立加载验证超过 180 秒，导出未验收") from exc
    if process.returncode:
        raise RuntimeError(f"独立加载验证失败（{process.returncode}），查看 {log_path}")
    result = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
    if result.get("passed") is not True:
        raise ValueError("独立验证未返回成功结果")
    request_path.unlink()
    return result


def _example(format_name: str) -> str:
    common = '''"""CPU structural execution example; this is not an accuracy measurement."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
size = manifest["input_shape"][-1]
pixels = np.linspace(0, 1, 3 * size * size, dtype=np.float32).reshape(1, 3, size, size)
'''
    if format_name == "onnx":
        return (
            common
            + """import onnxruntime as ort
session = ort.InferenceSession(str(ROOT / "model.onnx"), providers=["CPUExecutionProvider"])
outputs = session.run(None, {session.get_inputs()[0].name: pixels})
print([array.shape for array in outputs], manifest["classes"])
"""
        )
    if format_name == "pt":
        return (
            common
            + """from ultralytics import YOLO
model = YOLO(str(ROOT / "model.pt"))
image = (pixels[0].transpose(1, 2, 0) * 255).astype(np.uint8)
print(model.predict(image, imgsz=size, device="cpu", verbose=False)[0])
"""
        )
    return (
        common
        + """import ncnn
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 2
assert net.load_param(str(ROOT / "model.ncnn.param")) == 0
assert net.load_model(str(ROOT / "model.ncnn.bin")) == 0
planar = np.ascontiguousarray(pixels[0])  # Keep array and tensor alive through extract.
tensor = ncnn.Mat(planar)
extractor = net.create_extractor()
assert extractor.input(manifest["input_name"], tensor) == 0
status, output = extractor.extract(manifest["output_name"])
assert status == 0
print(np.asarray(output).shape, manifest["classes"])
"""
    )


def _ascript_example() -> str:
    return '''"""AScript Android entry. Prepare Yolov8Ncnn:1.3 on the device first.
Keep this __init__.py, param/bin and labels.txt together in the project root.
Device execution remains a separate acceptance step.
"""
import os
from ascript.android import plug

plug.load("Yolov8Ncnn:1.3")
import Yolov8Ncnn as detector

project_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(project_dir, "labels.txt"), encoding="utf-8-sig") as labels_file:
    class_names = labels_file.read().splitlines()
try:
    ready = detector.load(os.path.join(project_dir, "model.ncnn.param"),
                          os.path.join(project_dir, "model.ncnn.bin"), len(class_names), False)
    if not ready:
        raise RuntimeError("NCNN model initialization failed")
    for detection in detector.detect(target_size=640, threshold=0.5, nms_threshold=0.5):
        print(class_names[detection["class_id"]], detection["confidence"], detection["rect"])
finally:
    detector.free()
'''


def export(request: dict, emit) -> dict:
    from .worker import prepare_ultralytics

    prepare_ultralytics()
    from ultralytics import YOLO

    parameters, run_dir = request["parameters"], Path(request["run_dir"])
    format_name, profile = parameters.get("format", "onnx"), parameters.get("profile", "generic")
    if format_name not in {"pt", "onnx", "ncnn"} or profile not in {"generic", "cq", "ascript_v8"}:
        raise ValueError("未知导出格式或目标契约")
    if (profile == "cq" and format_name != "onnx") or (profile == "ascript_v8" and format_name != "ncnn"):
        raise ValueError("CQ_AI 需要 ONNX；AScript v8 需要 NCNN")
    size = parameters.get("imgsz", 640)
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError("导出输入尺寸必须是正数且为 32 的倍数")
    if profile == "cq" and size not in {320, 640} or profile == "ascript_v8" and size != 640:
        raise ValueError("CQ_AI 需要固定 320/640；AScript v8 需要固定 640")
    if parameters.get("half", False) or parameters.get("dynamic", False) or parameters.get("nms", False):
        raise ValueError("首版导出固定尺寸、Batch 1、FP32、无 NMS")
    source = Path(resolve_model(parameters["model"], allow_architecture=False))
    if format_name == "onnx":
        _need("onnx", "onnxruntime")
    elif format_name == "ncnn":
        _need("ncnn", "pnnx")
        import pnnx

        if not Path(pnnx.EXEC_PATH).is_file():
            raise RuntimeError("PNNX 本地转换器不存在；请重新准备完整训练环境")
    model = YOLO(str(source), task="detect")
    if model.task != "detect":
        raise ValueError("首版只支持目标检测模型导出")
    classes = (
        [name for _, name in sorted(model.names.items())]
        if isinstance(model.names, dict)
        else list(model.names)
    )
    yaml_name = str(getattr(model.model, "yaml", {}).get("yaml_file", ""))
    match = re.search(r"(yolov8|yolo11|yolo26)[nsmlx]?", yaml_name)
    detected_family = match[1] if match else None
    sidecar_path = source.with_suffix(".manifest.json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("classes") != classes or sidecar.get("sha256") != digest(source):
            raise ValueError("模型训练 manifest 与权重不一致")
        detected_family = sidecar["family"]
    family = parameters.get("family", detected_family)
    if family not in {"yolov8", "yolo11", "yolo26"} or detected_family and family != detected_family:
        raise ValueError("模型系列未知或与指定系列不一致")
    if profile != "generic" and family not in ({"yolov8", "yolo11"} if profile == "cq" else {"yolov8"}):
        raise ValueError("模型系列不兼容该目标端；YOLO26 应选择通用导出")
    root = Path(parameters.get("output_dir", run_dir / "exports")).resolve()
    if any(part.lower() == "input" for part in root.parts):
        raise ValueError("input/ 是只读来源目录，不能作为导出目的地")
    folder = root / f"{profile}-{format_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    # Exporters write adjacent to their input. Copy once into a fresh package
    # so neither user weights nor historical exports can be overwritten.
    shutil.copyfile(source, folder / "model.pt")
    emit(
        "progress", {"phase": "export", "package_dir": str(folder), "format": format_name, "profile": profile}
    )
    try:
        if format_name != "pt":
            model = YOLO(str(folder / "model.pt"), task="detect")
            options = dict(
                format=format_name,
                imgsz=size,
                batch=1,
                device=parameters.get("device", "cpu"),
                half=False,
                dynamic=False,
                simplify=False,
                nms=False,
                verbose=False,
            )
            if format_name == "onnx":
                options["opset"] = 12
            # TorchScript/PNNX use narrow native path APIs on Windows. Keep
            # native filenames ASCII and relative to the fresh package while
            # Python retains the user's Unicode destination path.
            previous_cwd = Path.cwd()
            try:
                os.chdir(folder)
                model.model.pt_path = "model.pt"
                result = Path(model.export(**options)).resolve()
            finally:
                os.chdir(previous_cwd)
            if format_name == "onnx":
                if result.resolve() != (folder / "model.onnx").resolve():
                    shutil.copyfile(result, folder / "model.onnx")
                import onnx

                onnx.checker.check_model(str(folder / "model.onnx"))
            else:
                for name in ("model.ncnn.param", "model.ncnn.bin"):
                    shutil.copyfile(result / name, folder / name)
                _ncnn_names(folder / "model.ncnn.param")
                # Remove only converter-owned paths inside this fresh package.
                if (
                    not result.resolve().is_relative_to(folder.resolve())
                    or result.resolve() == folder.resolve()
                ):
                    raise ValueError("转换器输出目录超出本次导出包")
                for generated in sorted(result.rglob("*"), key=lambda value: len(value.parts), reverse=True):
                    generated.rmdir() if generated.is_dir() else generated.unlink()
                result.rmdir()
            (folder / "model.pt").unlink()
        structural = None
        if format_name == "onnx":
            structural, structural_opset = inspect_onnx(folder / "model.onnx", family, classes)
            validate_contract(structural, profile, opset=structural_opset)
        emit("progress", {"phase": "isolated_validation", "package_dir": str(folder)})
        validation = _independent_check(folder, {"imgsz": size, "format": format_name}, run_dir)
        opset = None
        if format_name == "onnx":
            artifact, opset = inspect_onnx(folder / "model.onnx", family, classes)
            if artifact.output_shapes != validation["output_shapes"]:
                raise ValueError("ONNX 结构与实际输出形状不一致")
            model_file = "model.onnx"
        else:
            model_file = "model.pt" if format_name == "pt" else "model.ncnn.param"
            artifact = ModelManifest(
                family,
                format_name,
                classes,
                [1, 3, size, size],
                validation["output_shapes"],
                digest(folder / model_file),
            )
        validate_contract(artifact, profile, opset=opset)
        extra = {}
        if format_name == "ncnn":
            extra["input_name"], extra["output_name"] = _ncnn_names(folder / "model.ncnn.param")
        result = {
            **artifact.to_dict(),
            "profile": profile,
            "output_layout": "xyxy_score_class"
            if family == "yolo26" and format_name != "ncnn"
            else "xywh_class_scores",
            "nms_required": family != "yolo26" or format_name == "ncnn",
            "model_file": model_file,
            "opset": opset,
            "source_sha256": digest(source),
            "validation": validation,
            "created_by_job": request["job_id"],
            "files": {
                path.name: digest(path)
                for path in folder.iterdir()
                if path.suffix in {".pt", ".onnx", ".param", ".bin"}
            },
            **extra,
        }
        atomic_write(folder / "labels.txt", "\n".join(classes) + "\n")
        atomic_write(folder / "classes.txt", "\n".join(classes) + "\n")
        atomic_write(folder / "example.py", _example(format_name))
        if profile == "ascript_v8":
            atomic_write(folder / "__init__.py", _ascript_example())
        note = (
            "AScript v8: 固定 640，单输出 [4+C,8400]，FP32。将 param/bin 和 labels.txt 一起部署。\n"
            "__init__.py 是 AScript 项目入口；先在设备准备 Yolov8Ncnn:1.3，保持 param/bin/labels.txt 与入口同目录。\n"
            "Android 真机验收仍需在使用端执行：https://ascript.cn/docs/android/api/screen/yolo/v8/yolo/\n"
            if profile == "ascript_v8"
            else "CQ_AI: 固定 320/640，FP32，opset12，无 NMS，单输出 [1,4+C,N]。IoU 固定 0.45。\n"
            if profile == "cq"
            else "通用模型包；解码器应按 manifest.json 的实际输出形状处理。\n"
        )
        atomic_write(
            folder / "README.txt",
            note + "独立 CPU 加载及图案输入执行已通过；不代表业务准确率或目标设备验收。\n"
            "example.py 为本地 Python 运行环境示例；依赖需事先显式准备。\n",
        )
        atomic_write(folder / "manifest.json", json_text(result))
        return {
            "package_dir": str(folder),
            "manifest": str(folder / "manifest.json"),
            "profile": profile,
            "format": format_name,
            "validation": validation,
            "classes": classes,
        }
    except Exception as exc:
        atomic_write(folder / "failure.json", json_text({"state": "failed", "error": str(exc)}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.validate.read_text(encoding="utf-8"))
    from .worker import configure_offline

    configure_offline(Path(spec["folder"]))
    result = validate_execution(spec)
    atomic_write(Path(spec["folder"]) / "validation.json", json_text(result))


if __name__ == "__main__":
    main()
