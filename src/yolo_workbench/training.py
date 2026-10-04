"""Training settings independent of the heavy runtime and desktop widgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml

EXPERT_KEYS = {
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "amp",
    "cache",
    "mosaic",
    "mixup",
    "copy_paste",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "close_mosaic",
}


@dataclass(frozen=True)
class TrainingConfig:
    family: str = "yolov8"
    scale: str = "n"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 4
    device: str = "cpu"
    workers: int = 0
    patience: int = 100

    def effective(self, expert_yaml: str = "") -> dict:
        if (
            self.family not in {"yolov8", "yolo11", "yolo26"}
            or self.scale not in "nsmlx"
            or len(self.scale) != 1
        ):
            raise ValueError("不支持的模型系列或规格")
        for key in ("imgsz", "epochs", "batch", "workers", "patience"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} 必须是整数")
        if self.imgsz < 32 or self.imgsz % 32 or self.epochs < 1 or self.workers < 0 or self.patience < 0:
            raise ValueError("输入尺寸需为 32 的倍数，轮数/Workers/早停参数非法")
        if self.device != "cpu" and not self.device.isdigit():
            raise ValueError("首版训练设备仅支持 CPU 或单个 CUDA 设备编号")
        if self.batch == 0 or self.batch < -1 or (self.device == "cpu" and self.batch < 1):
            raise ValueError("CPU 使用正数 Batch；CUDA 可用 -1 自动 Batch")
        expert = yaml.safe_load(expert_yaml) if expert_yaml.strip() else {}
        if not isinstance(expert, dict) or set(expert) - EXPERT_KEYS:
            raise ValueError("专家配置包含未知或由应用管理的参数")
        for key, value in expert.items():
            if key == "optimizer" and (
                not isinstance(value, str)
                or value not in {"auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}
            ):
                raise ValueError("不支持的优化器")
            if key == "amp" and type(value) is not bool:
                raise ValueError("AMP 必须为布尔值")
            if key == "cache" and value not in (False, True, "ram", "disk"):
                raise ValueError("非法缓存参数")
            if key not in {"optimizer", "amp", "cache"}:
                import math

                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} 必须为有限非负数")
                if (
                    key in {"mosaic", "mixup", "copy_paste", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v"}
                    and value > 1
                ):
                    raise ValueError(f"{key} 必须位于 0 到 1")
        result = {k: v for k, v in asdict(self).items() if k not in {"family", "scale"}}
        return {**result, **expert}

    @property
    def model_name(self) -> str:
        return f"{self.family}{self.scale}.pt"
