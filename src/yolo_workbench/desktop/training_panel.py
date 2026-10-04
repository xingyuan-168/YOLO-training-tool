from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..runtime import model_cache
from ..training import TrainingConfig
from .widgets import PathField


class TrainingPanel(QWidget):
    start_requested = Signal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        base = QWidget()
        form = QFormLayout(base)
        self.family = QComboBox()
        self.family.addItems(["yolov8", "yolo11", "yolo26"])
        self.scale = QComboBox()
        self.scale.addItems(list("nsmlx"))
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(100)
        self.batch = QSpinBox()
        self.batch.setRange(-1, 2048)
        self.batch.setValue(4)
        self.device = QComboBox()
        self.device.addItems(["cpu", "0"])
        self.device.setEditable(True)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.patience = QSpinBox()
        self.patience.setRange(0, 100000)
        self.patience.setValue(100)
        for title, widget in [
            ("模型系列", self.family),
            ("规格", self.scale),
            ("输入尺寸", self.imgsz),
            ("Epochs", self.epochs),
            ("Batch（-1 自动）", self.batch),
            ("设备", self.device),
            ("Workers", self.workers),
            ("早停 patience", self.patience),
        ]:
            form.addRow(title, widget)
        tabs.addTab(base, "基础")
        advanced = QWidget()
        advanced_form = QFormLayout(advanced)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(0.000001, 1)
        self.lr.setValue(0.01)
        self.decay = QDoubleSpinBox()
        self.decay.setDecimals(6)
        self.decay.setRange(0, 1)
        self.decay.setValue(0.0005)
        self.amp = QCheckBox("启用 AMP")
        self.amp.setChecked(True)
        self.cache = QComboBox()
        self.cache.addItems(["关闭", "ram", "disk"])
        self.mosaic = QDoubleSpinBox()
        self.mosaic.setRange(0, 1)
        self.mosaic.setSingleStep(0.1)
        self.mosaic.setValue(1)
        for title, widget in [
            ("优化器", self.optimizer),
            ("学习率", self.lr),
            ("权重衰减", self.decay),
            ("混合精度", self.amp),
            ("缓存", self.cache),
            ("Mosaic", self.mosaic),
        ]:
            advanced_form.addRow(title, widget)
        hint = QLabel("其余增强参数可在专家配置中填写。\n高级参数始终参与最终配置。")
        hint.setWordWrap(True)
        advanced_form.addRow(hint)
        tabs.addTab(advanced, "高级")
        expert = QWidget()
        expert_layout = QVBoxLayout(expert)
        self.expert = QPlainTextEdit()
        self.expert.setPlaceholderText(
            "# 可覆盖高级参数，例如\nlr0: 0.005\nfliplr: 0.5\n# 数据快照和输出目录由应用管理"
        )
        expert_layout.addWidget(self.expert)
        tabs.addTab(expert, "专家")
        weights_group = QGroupBox("初始权重")
        weights_layout = QVBoxLayout(weights_group)
        self.weights = PathField("留空使用已准备的官方权重")
        self.weights.button.clicked.connect(self.select_weights)
        self.weights.changed.connect(self.weight_changed)
        self.scratch = QCheckBox("从模型结构开始训练（随机初始化）")
        self.exclude_pending = QCheckBox("本次明确排除未标注图片")
        weights_layout.addWidget(self.weights)
        weights_layout.addWidget(self.scratch)
        weights_layout.addWidget(self.exclude_pending)
        layout.addWidget(weights_group)
        self.start = QPushButton("检查并开始训练")
        self.start.setProperty("primary", True)
        self.start.clicked.connect(self.request_start)
        layout.addWidget(self.start)
        note = QLabel("CPU 默认 Batch=4、Workers=0。\n单次训练；标注可继续进行。")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.device.currentTextChanged.connect(self.device_changed)

    def select_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择初始权重", str(model_cache(self.settings.values)), "PyTorch 权重 (*.pt)"
        )
        if path:
            self.weights.setText(path)
            self.scratch.setChecked(False)

    def weight_changed(self, path):
        candidate = Path(path)
        matched = re.fullmatch(r"(yolov8|yolo11|yolo26)([nsmlx])", candidate.stem)
        if matched:
            self.family.setCurrentText(matched.group(1))
            self.scale.setCurrentText(matched.group(2))
        for sidecar in (
            (
                candidate.with_suffix(candidate.suffix + ".manifest.json"),
                candidate.with_suffix(".manifest.json"),
            )
            if candidate.name
            else ()
        ):
            if sidecar.is_file():
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    if metadata.get("family"):
                        self.family.setCurrentText(metadata["family"])
                except (ValueError, OSError):
                    pass
                break

    def device_changed(self, device):
        if device == "cpu":
            self.batch.setValue(4)
            self.workers.setValue(0)
        elif device.isdigit():
            self.batch.setValue(-1)
            self.workers.setValue(4)

    def configuration(self):
        config = TrainingConfig(
            family=self.family.currentText(),
            scale=self.scale.currentText(),
            imgsz=self.imgsz.value(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            device=self.device.currentText(),
            workers=self.workers.value(),
            patience=self.patience.value(),
        )
        advanced = {
            "optimizer": self.optimizer.currentText(),
            "lr0": self.lr.value(),
            "weight_decay": self.decay.value(),
            "amp": self.amp.isChecked(),
            "cache": False if self.cache.currentIndex() == 0 else self.cache.currentText(),
            "mosaic": self.mosaic.value(),
        }
        expert = yaml.safe_load(self.expert.toPlainText()) or {}
        if not isinstance(expert, dict):
            raise ValueError("专家配置必须是 YAML 键值对象")
        text = yaml.safe_dump({**advanced, **expert}, allow_unicode=True)
        effective = config.effective(text)
        model = self.weights.text()
        if self.scratch.isChecked():
            model = f"{config.family}{config.scale}.yaml"
        elif not model:
            model = str(model_cache(self.settings.values) / config.model_name)
        if not self.scratch.isChecked() and not Path(model).is_file():
            raise ValueError(f"权重尚未准备：{model}\n请在设置中准备模型，或选择本地权重 / 随机初始化。")
        return {
            "model": model,
            "config": asdict(config),
            "expert_yaml": text,
            "effective": effective,
            "finetune": bool(self.weights.text()) and not self.scratch.isChecked(),
            "exclude_pending": self.exclude_pending.isChecked(),
            "ui_expert_yaml": self.expert.toPlainText(),
            "ui_advanced": advanced,
        }

    def request_start(self):
        from PySide6.QtWidgets import QMessageBox

        try:
            self.start_requested.emit(self.configuration())
        except (ValueError, OSError, yaml.YAMLError) as exc:
            QMessageBox.warning(self, "训练参数检查", str(exc))

    def apply_config(self, values):
        # Device signal has defaults; apply all supplied values afterwards.
        if "device" in values:
            self.device.setCurrentText(str(values["device"]))
        for key in ("family", "scale"):
            if key in values:
                getattr(self, key).setCurrentText(str(values[key]))
        for key in ("imgsz", "epochs", "batch", "workers", "patience"):
            if key in values:
                getattr(self, key).setValue(int(values[key]))
        if "expert_yaml" in values:
            self.expert.setPlainText(str(values["expert_yaml"]))
        advanced = values.get("ui_advanced", {})
        if "optimizer" in advanced:
            self.optimizer.setCurrentText(advanced["optimizer"])
        if "lr0" in advanced:
            self.lr.setValue(advanced["lr0"])
        if "weight_decay" in advanced:
            self.decay.setValue(advanced["weight_decay"])
        if "amp" in advanced:
            self.amp.setChecked(advanced["amp"])
        if "cache" in advanced:
            self.cache.setCurrentText(advanced["cache"] if advanced["cache"] else "关闭")
        if "mosaic" in advanced:
            self.mosaic.setValue(advanced["mosaic"])

    def reset_project(self):
        self.apply_config({**asdict(TrainingConfig()), "expert_yaml": ""})
        self.weights.setText("")
        self.scratch.setChecked(False)
        self.exclude_pending.setChecked(False)
        self.optimizer.setCurrentText("auto")
        self.lr.setValue(0.01)
        self.decay.setValue(0.0005)
        self.amp.setChecked(True)
        self.cache.setCurrentIndex(0)
        self.mosaic.setValue(1)
