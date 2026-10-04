"""Ultralytics train/evaluate handlers. Imported only inside owned workers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml

from .storage import atomic_write, child_path, json_text
from .training import resolve_model, validate_training_request


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def validate_model_family(model, expected: str) -> str:
    """Verify architecture metadata after loading; filenames are not authority."""
    architecture = getattr(getattr(model, "model", None), "yaml", {})
    yaml_file = architecture.get("yaml_file", "") if isinstance(architecture, dict) else ""
    name = str(yaml_file).replace("\\", "/").rsplit("/", 1)[-1].lower()
    match = re.fullmatch(r"(yolov8|yolo11|yolo26)[nsmlx]?\.ya?ml", name)
    if not match:
        raise ValueError("模型架构缺少可识别的 YOLOv8/YOLO11/YOLO26 YAML 元数据，无法验证所选模型系列")
    actual = match.group(1)
    if actual != expected:
        raise ValueError(f"权重实际架构为 {actual}，当前训练配置为 {expected}；请选择匹配的模型系列后重试")
    return actual


def prepare_snapshot(snapshot: Path, run_dir: Path) -> tuple[dict, Path]:
    """Give Ultralytics a private cache location; frozen snapshots stay read-only."""
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("classes") or not manifest.get("assets"):
        raise ValueError("快照内容或版本非法")
    target = run_dir / "data"
    target.mkdir(exist_ok=False)
    subsets = set()
    for asset in manifest["assets"]:
        subset, asset_id = asset["subset"], asset["id"]
        if subset not in {"train", "val", "test"}:
            raise ValueError("快照划分非法")
        suffix = Path(asset["file"]).suffix
        image_name = f"images/{subset}/{asset_id}{suffix}"
        label_name = f"labels/{subset}/{asset_id}.txt"
        source_image, source_label = child_path(snapshot, image_name), child_path(snapshot, label_name)
        if digest(source_image) != asset["hash"] or digest(source_label) != asset["label_sha256"]:
            raise ValueError(f"冻结快照发生变化：{asset_id}")
        image, label = child_path(target, image_name), child_path(target, label_name)
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_image, image)
        except OSError:
            shutil.copyfile(source_image, image)
        shutil.copyfile(source_label, label)
        subsets.add(subset)
    if not {"train", "val"}.issubset(subsets):
        raise ValueError("快照必须包含训练集和验证集")
    data = {
        "path": str(target),
        "names": manifest["classes"],
        **{subset: f"images/{subset}" for subset in sorted(subsets)},
    }
    data_path = target / "data.yaml"
    atomic_write(data_path, yaml.safe_dump(data, allow_unicode=True))
    return manifest, data_path


def clean_numbers(value):
    if isinstance(value, dict):
        return {str(k): clean_numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_numbers(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@lru_cache(maxsize=1)
def _resource_process():
    import psutil

    return psutil.Process()


def _resources(trainer=None):
    result = {}
    try:
        process = _resource_process()
        result.update(rss_bytes=process.memory_info().rss, cpu_percent=process.cpu_percent())
    except ImportError:
        pass
    if trainer is not None and trainer.device.type == "cuda":
        import torch

        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(trainer.device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(trainer.device),
        )
    return result


def _save_torch(path: Path, payload):
    import torch

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write(path, buffer.getvalue())


def train(request: dict, emit) -> dict:
    from .worker import prepare_ultralytics

    prepare_ultralytics()
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    parameters, run_dir = request["parameters"], Path(request["run_dir"])
    config, effective = validate_training_request(parameters)
    snapshot = Path(parameters["snapshot"]).resolve()
    manifest, data_path = prepare_snapshot(snapshot, run_dir)
    snapshot_hash = digest(snapshot / "snapshot.json")
    resume_path = parameters.get("resume_checkpoint")
    saved = None
    if resume_path:
        saved = torch.load(
            resolve_model(resume_path, allow_architecture=False), map_location="cpu", weights_only=False
        )
        metadata = saved.get("workbench_resume", {})
        if metadata.get("schema_version") != 1 or not saved.get("optimizer") or saved.get("epoch", -1) < 0:
            raise ValueError("该文件不含完整恢复状态；请选择本工具 resume.pt，或使用“基于权重继续”")
        if metadata.get("snapshot_sha256") != snapshot_hash or metadata.get("classes") != manifest["classes"]:
            raise ValueError("完整恢复必须使用原始冻结快照和类别；新数据请使用“基于权重继续”")
        if saved["epoch"] + 1 >= saved["train_args"]["epochs"]:
            raise ValueError("该训练计划已经完成；增加训练轮数请使用“基于权重继续”")
        if metadata.get("family") != config.family:
            raise ValueError("恢复配置的模型系列与检查点不一致")
        model_source = str(Path(resume_path).resolve())
    else:
        model_source = resolve_model(parameters.get("model", config.model_name))

    class WorkbenchTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            if self.resume:
                # Upstream restores the old save_dir and data. Every resume is
                # a new durable job; historical metrics/weights stay intact.
                self.args.project = str(run_dir)
                self.args.name = "train"
                self.args.save_dir = str(run_dir / "train")
                self.args.data = str(data_path)
                self.args.exist_ok = False

    model = YOLO(model_source, task="detect")
    validate_model_family(model, config.family)
    started = time.monotonic()
    counter = {"batches": 0, "last_emit": 0.0, "saved_epoch": -1, "restored": False, "epoch": 0}
    resume_target = run_dir / "train/weights/resume.pt"

    def starting(trainer):
        if saved is not None:
            metadata = saved["workbench_resume"]
            unwrap_model(trainer.model).load_state_dict(saved["model"].state_dict())
            trainer.optimizer.load_state_dict(saved["optimizer"])
            trainer.scheduler.load_state_dict(metadata["scheduler"])
            trainer.stopper.__dict__.update(metadata["stopper"])
            trainer.accumulate = metadata["accumulate"]
            random.setstate(metadata["rng_python"])
            np.random.set_state(metadata["rng_numpy"])
            torch.set_rng_state(metadata["rng_torch"])
            if metadata.get("rng_cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(metadata["rng_cuda"])
            if metadata.get("loader_generator") is not None:
                trainer.train_loader.generator.set_state(metadata["loader_generator"])
                trainer.train_loader.reset()
            counter["restored"] = True
            emit(
                "resume",
                {
                    "checkpoint": model_source,
                    "start_epoch": trainer.start_epoch + 1,
                    "optimizer_restored": True,
                    "optimizer_state_entries": len(trainer.optimizer.state),
                    "scheduler_restored": True,
                    "rng_restored": True,
                    "model_restored": "unaveraged_training_weights",
                },
            )
        emit(
            "config",
            {
                "effective": clean_numbers(vars(trainer.args)),
                "family": config.family,
                "snapshot": str(snapshot),
                "classes": manifest["classes"],
                "mode": "resume" if saved else "finetune" if parameters.get("finetune") else "train",
            },
        )

    def epoch_start(trainer):
        counter["epoch"] = 0
        if saved is not None and trainer.epoch == trainer.start_epoch:
            # The upstream loop clears gradients after on_train_start.
            for name, parameter in unwrap_model(trainer.model).named_parameters():
                gradient = saved["workbench_resume"].get("gradients", {}).get(name)
                if gradient is not None:
                    parameter.grad = gradient.to(parameter.device)
        if (run_dir / "stop.flag").exists() and resume_target.exists():
            trainer.stop = True

    def batch_end(trainer):
        counter["batches"] += 1
        counter["epoch"] += 1
        now = time.monotonic()
        if now - counter["last_emit"] < 0.3:
            return
        counter["last_emit"] = now
        remaining = (trainer.epochs - trainer.epoch - 1) * len(trainer.train_loader)
        remaining += len(trainer.train_loader) - counter["epoch"]
        emit(
            "progress",
            clean_numbers(
                {
                    "epoch": trainer.epoch + 1,
                    "epochs": trainer.epochs,
                    "batch": counter["epoch"],
                    "batches": len(trainer.train_loader),
                    "losses": trainer.label_loss_items(trainer.tloss),
                    "elapsed_seconds": now - started,
                    "eta_seconds": (now - started) / counter["batches"] * remaining,
                    "resources": _resources(trainer),
                }
            ),
        )

    def preserve(trainer):
        checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)
        # Save before final_eval strips optimizer. Do not inherit upstream's
        # FP16 optimizer or EMA-only checkpoint for full-state continuation.
        checkpoint["model"] = deepcopy(unwrap_model(trainer.model)).cpu().float()
        checkpoint["ema"] = deepcopy(unwrap_model(trainer.ema.ema)).cpu().float()
        checkpoint["optimizer"] = deepcopy(trainer.optimizer.state_dict())
        checkpoint["workbench_resume"] = {
            "schema_version": 1,
            "family": config.family,
            "classes": manifest["classes"],
            "snapshot": str(snapshot),
            "snapshot_sha256": snapshot_hash,
            "scheduler": trainer.scheduler.state_dict(),
            "stopper": vars(trainer.stopper).copy(),
            "accumulate": trainer.accumulate,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": trainer.train_loader.generator.get_state(),
            "gradients": {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in unwrap_model(trainer.model).named_parameters()
                if parameter.grad is not None
            },
        }
        _save_torch(resume_target, checkpoint)
        counter["saved_epoch"] = trainer.epoch
        emit(
            "checkpoint",
            {
                "epoch": trainer.epoch + 1,
                "resume_checkpoint": str(resume_target),
                "best": str(trainer.best),
                "last": str(trainer.last),
                "full_state": True,
            },
        )
        if (run_dir / "stop.flag").exists():
            trainer.stop = True
            emit("state", {"state": "stopping", "reason": "本轮检查点已原子保存，正在完成验证与清理"})

    def epoch_end(trainer):
        emit(
            "metrics",
            clean_numbers(
                {
                    "epoch": trainer.epoch + 1,
                    "epochs": trainer.epochs,
                    "losses": trainer.label_loss_items(trainer.tloss),
                    "metrics": trainer.metrics or {},
                    "learning_rates": trainer.lr,
                    "resources": _resources(trainer),
                }
            ),
        )
        if (run_dir / "stop.flag").exists():
            trainer.stop = True

    model.add_callback("on_train_start", starting)
    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_model_save", preserve)
    model.add_callback("on_fit_epoch_end", epoch_end)
    model.train(
        trainer=WorkbenchTrainer,
        **effective,
        data=str(data_path),
        project=str(run_dir),
        name="train",
        exist_ok=False,
        save=True,
        val=True,
        plots=True,
        verbose=False,
        resume=bool(saved),
        pretrained=not model_source.endswith((".yaml", ".yml")),
    )
    trainer = model.trainer
    artifact = {
        "schema_version": 1,
        "family": config.family,
        "format": "pt",
        "task": "detect",
        "classes": manifest["classes"],
        "input_shape": [1, 3, trainer.args.imgsz, trainer.args.imgsz],
        "output_shapes": [],
        "precision": "FP32",
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_hash,
        "job_id": request["job_id"],
    }
    for path in (trainer.best, trainer.last, resume_target):
        if path.is_file():
            atomic_write(path.with_suffix(".manifest.json"), json_text({**artifact, "sha256": digest(path)}))
    return {
        "run_dir": str(trainer.save_dir),
        "best": str(trainer.best),
        "last": str(trainer.last),
        "resume_checkpoint": str(resume_target),
        "epochs_completed": counter["saved_epoch"] + 1,
        "resumed": counter["restored"],
        "start_epoch": trainer.start_epoch + 1,
        "metrics": clean_numbers(trainer.metrics or {}),
        "results_csv": str(trainer.csv),
        "classes": manifest["classes"],
        "stopped": (run_dir / "stop.flag").exists(),
        "plots": [str(path) for path in trainer.save_dir.glob("*.png")],
    }


def evaluate(request: dict, emit) -> dict:
    from .worker import prepare_ultralytics

    prepare_ultralytics()
    from ultralytics import YOLO

    parameters, run_dir = request["parameters"], Path(request["run_dir"])
    from .training import TrainingConfig

    config = TrainingConfig(
        imgsz=parameters.get("imgsz", 640),
        batch=parameters.get("batch", 4),
        device=parameters.get("device", "cpu"),
        workers=parameters.get("workers", 0),
    )
    config.effective()
    manifest, data_path = prepare_snapshot(Path(parameters["snapshot"]), run_dir)
    split = parameters.get("split", "val")
    if split not in {"val", "test"} or not (run_dir / "data/images" / split).is_dir():
        raise ValueError("标准评估需要快照中非空的 val 或 test 划分")
    model = YOLO(resolve_model(parameters["model"], allow_architecture=False), task="detect")
    names = (
        [name for _, name in sorted(model.names.items())]
        if isinstance(model.names, dict)
        else list(model.names)
    )
    if names != manifest["classes"]:
        raise ValueError("评估类别映射与模型训练映射不同，不能混合计算标准指标")
    count = {"batch": 0}

    def progress(validator):
        count["batch"] += 1
        emit(
            "progress",
            {
                "batch": count["batch"],
                "batches": len(validator.dataloader),
                "phase": "evaluate",
                "resources": _resources(),
            },
        )

    model.add_callback("on_val_batch_end", progress)
    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        project=str(run_dir),
        name="evaluation",
        exist_ok=False,
        plots=True,
        verbose=False,
        save_json=True,
    )
    result = {
        "source": "ultralytics_standard_evaluation",
        "split": split,
        "metrics": clean_numbers(metrics.results_dict),
        "speed": clean_numbers(metrics.speed),
        "classes": names,
        "save_dir": str(metrics.save_dir),
        "plots": [str(path) for path in Path(metrics.save_dir).glob("*.png")],
    }
    atomic_write(run_dir / "metrics.json", json_text(result))
    emit("metrics", result)
    return result
