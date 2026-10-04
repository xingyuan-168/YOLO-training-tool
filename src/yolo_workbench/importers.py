"""Read-only dataset discovery and bounded, cooperative YOLO/ZIP imports.

Inspection reads metadata only. Import commits each image and its metadata as one
recoverable unit; cancellation keeps completed units and never approves missing labels.
"""

from __future__ import annotations

import math
import os
import posixpath
import re
import stat
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

import yaml
from PIL import Image

from .dataset import IMAGE_SUFFIXES, DatasetService
from .labels import Box, format_labels, parse_labels, validate_classes
from .storage import (
    OperationCancelled,
    check_cancel,
    child_path,
    json_text,
    remove_owned_tree,
    report_progress,
    safe_relative_path,
)
from .training import TrainingConfig

MAX_FILES = 100_000
MAX_FILE_BYTES = 1024**3
MAX_ARCHIVE_BYTES = 20 * 1024**3
MAX_COMPRESSION_RATIO = 200
MAX_METADATA_BYTES = 4 * 1024**2
MAX_LABEL_BYTES = 16 * 1024**2
TRAINING_NAMES = {"训练参数.json", "training.json", "training_config.json", "train_config.json"}
MODEL_NAMES = {"模型配置.json", "model_config.json", "model.json"}
CLASS_NAMES = {"labels.txt", "classes.txt", "obj.names"}


def _issue(path, message, *, code="invalid_input") -> dict:
    return {"path": str(path), "code": code, "message": str(message)}


def _integer(value, name) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"{name} 必须是整数或整数字符串")


def _family(value) -> str:
    aliases = {
        "8": "yolov8",
        "v8": "yolov8",
        "yolo8": "yolov8",
        "11": "yolo11",
        "v11": "yolo11",
        "26": "yolo26",
        "v26": "yolo26",
    }
    value = str(value).strip().lower()
    value = aliases.get(value, value)
    if value not in {"yolov8", "yolo11", "yolo26"}:
        raise ValueError("不支持的 YOLO 模型系列")
    return value


def normalize_training_config(raw: dict, *, existing: dict | None = None) -> dict:
    """Convert known legacy fields; omitted fields are never injected into the result."""
    if not isinstance(raw, dict):
        raise ValueError("训练参数必须是 JSON 对象")
    aliases = {
        "version": "family",
        "模型版本": "family",
        "yolo版本": "family",
        "model_size": "scale",
        "模型规模": "scale",
        "模型大小": "scale",
        "input_size": "imgsz",
        "输入尺寸": "imgsz",
        "batch_size": "batch",
        "批次大小": "batch",
        "训练轮数": "epochs",
        "早停轮数": "patience",
        "设备": "device",
        "工作线程": "workers",
        "advanced_params": "expert_yaml",
        "高级参数": "expert_yaml",
    }
    result = {}
    fields = set(asdict(TrainingConfig())) | {"expert_yaml"}
    for key, value in raw.items():
        name = aliases.get(key, key)
        if name not in fields:
            continue
        if name in {"imgsz", "batch", "epochs", "patience", "workers"}:
            value = _integer(value, name)
        elif name == "family":
            value = _family(value)
        elif name == "device":
            if type(value) is int:
                value = str(value)
            if not isinstance(value, str):
                raise ValueError("设备必须为 CPU 或 CUDA 编号")
            value = value.strip().lower().removeprefix("cuda:")
        elif name == "scale":
            if not isinstance(value, str):
                raise ValueError("模型规格必须为文本")
            value = value.strip().lower()
        elif name == "expert_yaml" and not isinstance(value, str):
            raise ValueError("高级参数必须是 YAML 文本")
        if name in result and result[name] != value:
            raise ValueError(f"{name} 的多个配置别名值冲突")
        result[name] = value
    combined = {**(existing or {}), **result}
    validation_values = combined.copy()
    if existing is None and "device" not in combined and combined.get("batch") == -1:
        # A metadata-only preview cannot infer the current project's device.
        validation_values["device"] = "0"
    config = TrainingConfig(**{k: v for k, v in validation_values.items() if k in fields - {"expert_yaml"}})
    config.effective(combined.get("expert_yaml", ""))
    return result


def normalize_model_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("模型配置必须是 JSON 对象")
    aliases = {
        "模型类型": "format",
        "yolo版本": "family",
        "推理尺寸": "imgsz",
        "标签文件": "labels_file",
        "模型文件": "model_file",
        "置信度": "confidence",
        "conf": "confidence",
    }
    result = {}
    for key, value in raw.items():
        name = aliases.get(key, key)
        if name == "family":
            value = _family(value)
        elif name == "imgsz":
            value = _integer(value, name)
            if value < 32 or value % 32:
                raise ValueError("推理尺寸必须为正的 32 倍数")
        elif name == "format":
            if not isinstance(value, str) or value.lower() not in {
                "pt",
                "onnx",
                "ncnn",
                "cq_ai",
                "ascript_v8",
            }:
                raise ValueError("不支持的模型格式")
            value = value.lower()
        elif name in {"labels_file", "model_file"}:
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} 必须为有效路径文本")
        elif name in {"confidence", "iou"}:
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError as exc:
                    raise ValueError(f"{name} 必须为有限数值") from exc
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} 必须位于 0 到 1")
        else:
            continue
        if name in result and result[name] != value:
            raise ValueError(f"{name} 的多个配置别名值冲突")
        result[name] = value
    return result


def apply_imported_config(service: DatasetService, config: dict) -> dict:
    """Validate, then merge normalized settings without resetting unrelated existing values."""
    current = service.get_settings()
    training = normalize_training_config(config.get("training", {}), existing=current.get("training", {}))
    model = normalize_model_config(config.get("model", {}))
    legacy = config.get("legacy", {})
    if not isinstance(legacy, dict):
        raise ValueError("原配置记录必须为对象")
    return service.update_settings({"training": training, "model": model, "legacy": legacy})


class _Source:
    def __init__(self, source: Path, *, cancel=None, progress=None):
        self.source = Path(source).resolve()
        self.files: dict[str, Path | zipfile.ZipInfo] = {}
        self.lookup: dict[str, str] = {}
        self.archive = None
        self.cancel, self.progress = cancel, progress
        self.selected_image = None
        self.image_prefix = ""
        self.subsets: dict[str, str] = {}
        self.list_files: set[str] = set()
        self.warnings = []
        try:
            if self.source.is_file() and self.source.suffix.lower() == ".zip":
                self.kind = "zip"
                self.archive = zipfile.ZipFile(self.source)
                members = self.archive.infolist()
                if len(members) > MAX_FILES:
                    raise ValueError(f"ZIP 条目数超过 {MAX_FILES} 限制")
                total = 0
                all_names: set[str] = set()
                file_names: set[str] = set()
                for number, member in enumerate(members, 1):
                    check_cancel(cancel)
                    if "\x00" in member.orig_filename:
                        raise ValueError("ZIP 路径含空字符")
                    name = safe_relative_path(member.filename)
                    key = name.casefold()
                    if key in all_names:
                        raise ValueError(f"ZIP 存在重复路径或 Windows 大小写冲突：{name}")
                    all_names.add(key)
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ValueError(f"ZIP 不允许符号链接或特殊文件：{name}")
                    if member.flag_bits & 1:
                        raise ValueError("不支持加密 ZIP")
                    if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                        raise ValueError("ZIP 仅支持 stored/deflate 压缩")
                    if member.file_size > MAX_FILE_BYTES or member.file_size < 0:
                        raise ValueError(f"ZIP 单文件超过容量限制：{name}")
                    total += member.file_size
                    if (
                        total > MAX_ARCHIVE_BYTES
                        or member.file_size / max(1, member.compress_size) > MAX_COMPRESSION_RATIO
                    ):
                        raise ValueError("ZIP 展开容量或压缩比超过安全限制")
                    if not member.is_dir():
                        file_names.add(key)
                        self._add(name, member)
                    report_progress(
                        progress, phase="inspect", completed=number, total=len(members), path=name
                    )
                if any(
                    parent.as_posix().casefold() in file_names
                    for name in self.files
                    for parent in Path(name).parents
                    if parent.as_posix() != "."
                ):
                    raise ValueError("ZIP 文件路径与目录路径冲突")
                self.base = self.source.parent
            else:
                if not self.source.exists():
                    raise ValueError("导入来源不存在")
                self.kind = "directory" if self.source.is_dir() else "file"
                self.base = self.source if self.source.is_dir() else self.source.parent
                if self.source.suffix.lower() in IMAGE_SUFFIXES and self.source.is_file():
                    self.selected_image = self.source.name
                # Preserve a selected subset while finding conventional sibling labels/.
                for ancestor in (self.base, *self.base.parents):
                    if ancestor.name.lower() == "images" and (ancestor.parent / "labels").is_dir():
                        self.base = ancestor.parent
                        if self.source.is_dir():
                            self.image_prefix = self.source.relative_to(self.base).as_posix() + "/"
                        break
                for directory, directories, filenames in os.walk(self.base, followlinks=False):
                    check_cancel(cancel)
                    directories[:] = sorted(
                        d
                        for d in directories
                        if not (Path(directory) / d).is_symlink() and not (Path(directory) / d).is_junction()
                    )
                    for filename in sorted(filenames):
                        check_cancel(cancel)
                        path = Path(directory) / filename
                        if path.is_symlink() or not path.resolve().is_relative_to(self.base):
                            self.warnings.append(
                                _issue(path, "已跳过指向来源目录外的链接", code="skipped_link")
                            )
                            continue
                        self._add(path.relative_to(self.base).as_posix(), path)
                        if len(self.files) > MAX_FILES:
                            raise ValueError(f"目录文件数超过 {MAX_FILES} 限制")
                        if len(self.files) % 200 == 0:
                            report_progress(
                                progress, phase="inspect", completed=len(self.files), total=0, path=path
                            )
                if self.selected_image:
                    self.selected_image = self.source.relative_to(self.base).as_posix()
            self.images = sorted(
                n
                for n in self.files
                if Path(n).suffix.lower() in IMAGE_SUFFIXES
                and (self.selected_image is None or n == self.selected_image)
                and n.startswith(self.image_prefix)
            )
            self.image_set = set(self.images)
            report_progress(
                progress, phase="inspect", completed=len(self.files), total=len(self.files), path=self.source
            )
        except BaseException:
            self.close()
            raise

    def _add(self, name, value):
        name = safe_relative_path(name)
        key = name.casefold()
        if key in self.lookup:
            raise ValueError(f"存在重复路径或 Windows 大小写冲突：{name}")
        self.files[name], self.lookup[key] = value, name

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def text(self, name: str, *, limit=MAX_METADATA_BYTES) -> str:
        value = self.files[name]
        size = value.file_size if isinstance(value, zipfile.ZipInfo) else value.stat().st_size
        if size > limit:
            raise ValueError(f"文本文件超过 {limit} 字节限制：{name}")
        if self.archive is not None:
            with self.archive.open(value) as stream:
                data = stream.read(limit + 1)
        else:
            with value.open("rb") as stream:
                data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"文本文件读取超过容量限制：{name}")
        return data.decode("utf-8-sig")

    def label_for(self, image: str) -> str | None:
        parts = list(Path(image).with_suffix(".txt").parts)
        candidates = []
        for number, part in enumerate(parts[:-1]):
            if part.casefold() == "images":
                candidates.append(Path(*parts[:number], "labels", *parts[number + 1 :]).as_posix())
        candidates.append(Path(*parts).as_posix())
        existing = [self.lookup[c.casefold()] for c in candidates if c.casefold() in self.lookup]
        if len(set(existing)) > 1:
            raise ValueError(f"同一图片存在多个候选标签：{image}")
        return existing[0] if existing else None

    def materialize(self, name: str, staging: Path | None) -> Path:
        value = self.files[name]
        if self.archive is None:
            return value
        target = child_path(staging, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        actual = 0
        with self.archive.open(value) as source, target.open("xb") as output:
            while chunk := source.read(1024**2):
                check_cancel(self.cancel)
                actual += len(chunk)
                if actual > min(value.file_size, MAX_FILE_BYTES):
                    raise ValueError("ZIP 实际展开容量与目录记录不符")
                output.write(chunk)
        if actual != value.file_size:
            raise ValueError("ZIP 实际展开容量与目录记录不符")
        return target

    def manifest_path(self, value: str, parent="") -> str:
        """Resolve a declared dataset path within the selected source boundary."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("数据集清单路径必须为非空文本")
        value = value.strip().replace("\\", "/")
        if PureWindowsPath(value).drive or value.startswith("/"):
            if self.archive is None:
                path = Path(value).resolve()
                if path.is_relative_to(self.base):
                    return path.relative_to(self.base).as_posix()
            raise ValueError("数据集清单引用了导入来源目录之外的路径")
        combined = posixpath.normpath(posixpath.join(parent, value))
        if combined == ".":
            return ""
        return safe_relative_path(combined)

    def read_splits(self, name: str, metadata: dict) -> None:
        if not any(key in metadata for key in ("train", "val", "test")):
            return
        parent = posixpath.dirname(name)
        declared_root = metadata.get("path", ".")
        if not isinstance(declared_root, str):
            raise ValueError("YAML path 必须为目录文本")
        # Absolute old export paths are audit hints; all reads stay in the import root.
        normalized_root = declared_root.replace("\\", "/").rstrip("/")
        absolute_root = bool(PureWindowsPath(normalized_root).drive or normalized_root.startswith("/"))
        if absolute_root:
            dataset_root = parent
            self.warnings.append(
                _issue(
                    name,
                    "绝对 path 已按 YAML 所在目录重新定位，外部路径不会读取",
                    code="rebased_dataset_path",
                )
            )
        else:
            dataset_root = self.manifest_path(declared_root, parent)
        subsets = {}
        for subset in ("train", "val", "test"):
            values = metadata.get(subset, [])
            if values in (None, ""):
                continue
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"YAML {subset} 必须为路径文本或列表")
            for reference in values:
                value = reference.replace("\\", "/")
                if absolute_root and value.casefold().startswith(normalized_root.casefold() + "/"):
                    value = value[len(normalized_root) + 1 :]
                target = self.manifest_path(value, dataset_root)
                key = self.lookup.get(target.casefold())
                members = []
                if key and Path(key).suffix.lower() == ".txt":
                    self.list_files.add(key)
                    for line in self.text(key).splitlines():
                        check_cancel(self.cancel)
                        if not line.strip() or line.lstrip().startswith("#"):
                            continue
                        item = self.manifest_path(line, posixpath.dirname(key))
                        actual = self.lookup.get(item.casefold())
                        if actual not in self.image_set:
                            raise ValueError(f"图片清单引用了缺失或非图片文件：{line}")
                        members.append(actual)
                elif key and key in self.image_set:
                    members = [key]
                else:
                    prefix = target.casefold().rstrip("/") + "/" if target else ""
                    members = [image for image in self.images if image.casefold().startswith(prefix)]
                    if not members:
                        raise ValueError(f"来源划分目录不存在或没有图片：{reference}")
                for image in members:
                    if image in subsets and subsets[image] != subset:
                        raise ValueError(f"来源划分存在同一图片跨集合泄漏：{image}")
                    subsets[image] = subset
        if self.subsets and self.subsets != subsets:
            raise ValueError("多个 YAML 的数据划分互相冲突")
        self.subsets = subsets


def _classes(value) -> list[str]:
    if isinstance(value, dict):
        keys = [_integer(key, "类别编号") for key in value]
        if len(set(keys)) != len(keys) or set(keys) != set(range(len(keys))):
            raise ValueError("YAML 类别编号必须从 0 连续递增")
        value = [name for _, name in sorted(zip(keys, value.values()))]
    return validate_classes(value)


def _preview(source: _Source, *, existing_training=None) -> dict:
    import json

    result = {
        "source": str(source.source),
        "kind": source.kind,
        "classes": None,
        "config": {"training": {}, "model": {}, "legacy": {}},
        "image_count": len(source.images),
        "label_count": 0,
        "errors": [],
        "warnings": list(source.warnings),
    }
    metadata = [
        n
        for n in source.files
        if Path(n).name.lower() in CLASS_NAMES | TRAINING_NAMES | MODEL_NAMES
        or Path(n).suffix.lower() in {".names", ".yaml", ".yml"}
    ]
    # Prefer the shallowest dataset metadata; do not mix nested exported model packages.
    if metadata:
        depth = min(len(Path(n).parts) for n in metadata)
        metadata = [n for n in metadata if len(Path(n).parts) == depth]
    excluded = set(metadata)
    for name in sorted(metadata):
        check_cancel(source.cancel)
        try:
            filename = Path(name).name.lower()
            text = source.text(name)
            classes = None
            if filename in CLASS_NAMES or Path(name).suffix.lower() == ".names":
                classes = validate_classes([line.strip() for line in text.splitlines() if line.strip()])
            elif Path(name).suffix.lower() in {".yaml", ".yml"}:
                value = yaml.safe_load(text)
                if not isinstance(value, dict):
                    raise ValueError("数据集 YAML 必须为对象")
                if "names" in value:
                    classes = _classes(value["names"])
                    if "nc" in value and _integer(value["nc"], "nc") != len(classes):
                        raise ValueError("YAML 的 nc 与类别数量不一致")
                if value.get("download"):
                    result["warnings"].append(
                        _issue(name, "download 字段不会执行；仅导入本地文件", code="ignored_download")
                    )
                if source.selected_image is None and not source.image_prefix:
                    source.read_splits(name, value)
            elif filename in TRAINING_NAMES | MODEL_NAMES:
                value = json.loads(
                    text,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"非有限 JSON 数值：{token}")
                    ),
                )
                section = "training" if filename in TRAINING_NAMES else "model"
                normalized = (
                    normalize_training_config(value, existing=existing_training)
                    if section == "training"
                    else normalize_model_config(value)
                )
                previous = result["config"][section]
                if any(k in previous and previous[k] != v for k, v in normalized.items()):
                    raise ValueError("多个配置文件包含互相冲突的参数")
                previous.update(normalized)
                result["config"]["legacy"][name] = value
            if classes is not None:
                if result["classes"] is not None and result["classes"] != classes:
                    raise ValueError("多个类别文件的名称或编号顺序不一致")
                result["classes"] = classes
        except (OSError, ValueError, UnicodeError, yaml.YAMLError, zipfile.BadZipFile) as exc:
            result["errors"].append(_issue(name, exc, code="invalid_metadata"))
    if source.subsets:
        source.images = [name for name in source.images if name in source.subsets]
        result["image_count"] = len(source.images)
    result["warnings"].extend(w for w in source.warnings if w not in result["warnings"])
    matched = set()
    for name in source.images:
        check_cancel(source.cancel)
        try:
            label = source.label_for(name)
            if label:
                matched.add(label)
            else:
                result["warnings"].append(_issue(name, "缺少标签，将保留为未审核", code="missing_label"))
        except ValueError as exc:
            result["errors"].append(_issue(name, exc, code="ambiguous_label"))
    result["label_count"] = len(matched)
    excluded_labels = matched | excluded | source.list_files
    for name in source.files:
        if (
            Path(name).suffix.lower() == ".txt"
            and name not in excluded_labels
            and Path(name).name.lower()
            not in {"train.txt", "val.txt", "test.txt", "readme.txt", "license.txt"}
        ):
            result["warnings"].append(_issue(name, "未找到对应图片的标签文件", code="orphan_label"))
    return result


def inspect_dataset(source: Path, *, progress=None, cancel=None) -> dict:
    """Return class/config previews and file counts without decoding image pixels."""
    try:
        with _Source(source, progress=progress, cancel=cancel) as opened:
            return _preview(opened)
    except OperationCancelled:
        return {
            "source": str(source),
            "kind": "unknown",
            "classes": None,
            "config": {"training": {}, "model": {}, "legacy": {}},
            "image_count": 0,
            "label_count": 0,
            "errors": [],
            "warnings": [],
            "cancelled": True,
        }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "source": str(source),
            "kind": "unknown",
            "classes": None,
            "config": {"training": {}, "model": {}, "legacy": {}},
            "image_count": 0,
            "label_count": 0,
            "errors": [_issue(source, exc)],
            "warnings": [],
        }


def _subset(name: str) -> str | None:
    for part in Path(name).parts[:-1]:
        if part.lower() in {"train", "val", "valid", "validation", "test"}:
            return {"valid": "val", "validation": "val"}.get(part.lower(), part.lower())
    return None


def import_dataset(
    service: DatasetService, source: Path, trust_empty_labels=False, progress=None, cancel=None
) -> dict:
    """Import local files; result describes successful atomic units, failures and cancellation.

    Duplicates retain existing labels and review state. Class metadata maps IDs by
    exact class name; only a project with no records may adopt a new class list.
    Empty files need trust_empty_labels=True. Missing files always stay pending.
    """
    result = {
        "total": 0,
        "imported": 0,
        "duplicates": 0,
        "failed": 0,
        "skipped": 0,
        "pending": 0,
        "labeled": 0,
        "empty": 0,
        "cancelled": False,
        "errors": [],
        "warnings": [],
        "config": {"training": {}, "model": {}, "legacy": {}},
        "split": {"train": [], "val": [], "test": []},
    }
    staging = None
    split_seen = {key: set() for key in ("train", "val", "test")}
    stage_parent = Path(tempfile.gettempdir()).resolve()
    try:
        if type(trust_empty_labels) is not bool:
            raise ValueError("空标签信任选项必须为布尔值")
        source = Path(source).resolve()
        if source.is_dir() and (source.is_relative_to(service.root) or service.root.is_relative_to(source)):
            raise ValueError("导入来源不能位于当前项目内或包含当前项目")
        with _Source(source, progress=progress, cancel=cancel) as opened:
            preview = _preview(opened, existing_training=service.get_settings().get("training", {}))
            result.update(
                total=preview["image_count"],
                errors=preview["errors"],
                warnings=preview["warnings"],
                config=preview["config"],
            )
            if result["errors"]:
                result["skipped"] = result["total"]
                return result
            check_cancel(cancel)
            classes = preview["classes"]
            original_count = service.count_assets() + service.count_assets(deleted=True)
            if classes and classes != service.project["classes"]:
                if not original_count:
                    mapping = {
                        i: classes.index(n) if n in classes else None
                        for i, n in enumerate(service.project["classes"])
                    }
                    service.migrate_classes(classes, mapping)
                elif set(classes) - set(service.project["classes"]):
                    raise ValueError("导入类别未在项目中定义，请先新增类别或使用空项目")
            mapping = (
                {i: service.project["classes"].index(name) for i, name in enumerate(classes)}
                if classes
                else None
            )
            if not classes and opened.images:
                result["warnings"].append(
                    _issue(source, "未找到类别文件，标签编号按当前项目类别解释", code="missing_classes")
                )
            if opened.archive is not None:
                staging = Path(tempfile.mkdtemp(prefix="yolo-import-", dir=stage_parent))
            for number, name in enumerate(opened.images, 1):
                check_cancel(cancel)
                path = None
                try:
                    label_name = opened.label_for(name)
                    labels = opened.text(label_name, limit=MAX_LABEL_BYTES) if label_name else None
                    if labels is not None:
                        boxes = parse_labels(labels, len(classes or service.project["classes"]))
                        if mapping:
                            boxes = [Box(mapping[b.class_id], b.cx, b.cy, b.width, b.height) for b in boxes]
                        labels = format_labels(boxes, len(service.project["classes"]))
                    path = opened.materialize(name, staging)
                    subset = opened.subsets.get(name) or _subset(name)
                    metadata = {"import_source": str(source), "import_relative_path": name}
                    if subset:
                        metadata["imported_subset"] = subset
                    asset_id, created = service.import_image(
                        path,
                        labels=labels,
                        confirmed_empty=trust_empty_labels and labels is not None and not labels.strip(),
                        source_reference=f"{source}!/{name}"
                        if opened.archive is not None
                        else str(path.resolve()),
                        metadata=metadata,
                    )
                    result["imported" if created else "duplicates"] += 1
                    if created:
                        result[service.get_asset(asset_id)["status"]] += 1
                    else:
                        result["warnings"].append(
                            _issue(name, "重复内容已跳过，保留项目中的标注与审核状态", code="duplicate_image")
                        )
                    if subset and asset_id not in split_seen[subset]:
                        result["split"][subset].append(asset_id)
                        split_seen[subset].add(asset_id)
                except (
                    OSError,
                    ValueError,
                    UnicodeError,
                    zipfile.BadZipFile,
                    Image.DecompressionBombError,
                ) as exc:
                    result["failed"] += 1
                    result["errors"].append(_issue(name, exc, code="image_import_failed"))
                finally:
                    if staging is not None and path is not None:
                        child_path(staging, path.relative_to(staging).as_posix()).unlink(missing_ok=True)
                report_progress(progress, phase="import", completed=number, total=result["total"], path=name)
            check_cancel(cancel)
            if any(preview["config"].values()):
                apply_imported_config(service, preview["config"])
            if not original_count and result["split"]["train"] and result["split"]["val"]:
                issues = service.validate_split(result["split"])
                if not issues:
                    service.transaction.write(
                        {
                            "splits/default.json": json_text(
                                {**result["split"], "source": str(source), "grouped": False}
                            )
                        }
                    )
                else:
                    result["warnings"].append(
                        _issue(
                            source, "来源划分未保存：" + issues[0]["message"], code="source_split_not_saved"
                        )
                    )
    except OperationCancelled:
        result["cancelled"] = True
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        result["errors"].append(_issue(source, exc))
    finally:
        result["skipped"] = max(
            0, result["total"] - result["imported"] - result["duplicates"] - result["failed"]
        )
        if staging is not None:
            remove_owned_tree(stage_parent, staging, staging.name)
    return result
