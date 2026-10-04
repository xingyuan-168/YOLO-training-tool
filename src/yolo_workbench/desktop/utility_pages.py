from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..runtime import application_root, model_cache, runtime_paths
from .widgets import PathField


class LogPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = []
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.task = QComboBox()
        self.task.addItem("全部任务", "")
        self.level = QComboBox()
        self.level.addItems(["全部级别", "INFO", "WARNING", "ERROR"])
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索日志、路径或任务 ID")
        self.hours = QSpinBox()
        self.hours.setRange(0, 720)
        self.hours.setSpecialValueText("全部时间")
        self.hours.setSuffix(" 小时内")
        self.auto_scroll = QCheckBox("自动滚动")
        self.auto_scroll.setChecked(True)
        for widget in (self.task, self.level, self.hours, self.search, self.auto_scroll):
            toolbar.addWidget(widget)
        copy = QPushButton("复制")
        copy.clicked.connect(self.copy)
        export = QPushButton("导出日志")
        export.clicked.connect(self.export)
        toolbar.addWidget(copy)
        toolbar.addWidget(export)
        layout.addLayout(toolbar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setProperty("log", True)
        self.text.setMaximumBlockCount(10000)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text)
        for signal in (
            self.task.currentIndexChanged,
            self.level.currentIndexChanged,
            self.hours.valueChanged,
            self.search.textChanged,
        ):
            signal.connect(self.filter)

    def append(self, level, message, task=""):
        timestamp = datetime.now()
        entry = {"time": timestamp, "level": level, "task": task, "message": str(message)}
        self.entries.append(entry)
        if len(self.entries) > 10000:
            self.entries = self.entries[-10000:]
        if task and self.task.findData(task) < 0:
            self.task.addItem(task[:12], task)
        if self.matches(entry):
            scrollbar = self.text.verticalScrollBar()
            previous = scrollbar.value()
            self.text.appendPlainText(self.format(entry))
            if not self.auto_scroll.isChecked():
                scrollbar.setValue(previous)

    def matches(self, entry):
        return (
            (not self.task.currentData() or self.task.currentData() == entry["task"])
            and (self.level.currentIndex() == 0 or self.level.currentText() == entry["level"])
            and (
                not self.hours.value()
                or entry["time"] >= datetime.now() - timedelta(hours=self.hours.value())
            )
            and self.search.text().casefold() in self.format(entry).casefold()
        )

    @staticmethod
    def format(entry):
        return f"{entry['time']:%Y-%m-%d %H:%M:%S} [{entry['level']}] {entry['task'][:12]} {entry['message']}"

    def filter(self, *_):
        self.text.setPlainText("\n".join(self.format(e) for e in self.entries if self.matches(e)))
        if self.auto_scroll.isChecked():
            self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def copy(self):
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.text.textCursor().selectedText() or self.text.toPlainText())

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "workbench.log", "日志 (*.log *.txt)")
        if path:
            Path(path).write_text(self.text.toPlainText(), encoding="utf-8")


class SettingsPage(QWidget):
    saved = Signal()
    diagnose = Signal()
    prepare_models = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        layout = QVBoxLayout(self)
        runtime = QGroupBox("独立运行环境")
        form = QFormLayout(runtime)
        paths = runtime_paths(settings.values)
        self.fields = {}
        for key, label, default in [
            ("train_python", "训练 Python", paths["train"]),
            ("inference_python", "推理 Python", paths["inference"]),
            ("model_cache", "模型缓存", model_cache(settings.values)),
            ("projects_dir", "默认项目目录", application_root() / "projects"),
        ]:
            field = PathField()
            field.setText(settings.values.get(key) or default)
            field.button.clicked.connect(lambda checked=False, k=key: self.browse(k))
            self.fields[key] = field
            form.addRow(label, field)
        layout.addWidget(runtime)
        controls = QGroupBox("截图与恢复")
        controls_form = QFormLayout(controls)
        self.hotkey = QKeySequenceEdit(QKeySequence(settings.values.get("capture_hotkey", "Ctrl+E")))
        self.interval = QSpinBox()
        self.interval.setRange(1, 3600)
        self.interval.setValue(settings.values.get("capture_interval", 5))
        self.interval.setSuffix(" 秒")
        self.restore = QCheckBox("启动时恢复上次项目")
        self.restore.setChecked(settings.values.get("restore_project", True))
        controls_form.addRow("全局截图快捷键", self.hotkey)
        controls_form.addRow("定时截图间隔", self.interval)
        controls_form.addRow(self.restore)
        layout.addWidget(controls)
        row = QHBoxLayout()
        save = QPushButton("保存设置")
        save.setProperty("primary", True)
        save.clicked.connect(self.save)
        diagnose = QPushButton("运行环境诊断")
        diagnose.clicked.connect(self.diagnose)
        prepare = QPushButton("准备所选训练模型（需联网）")
        prepare.clicked.connect(self.prepare_models)
        for button in (save, diagnose, prepare):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText(
            "诊断会在独立进程中检查版本、CPU / CUDA、CQ_AI 和目录。运行训练时不会自动安装依赖。"
        )
        layout.addWidget(self.report, 1)
        note = QLabel("CQ_AI 0.14.6 · CPU 基础环境。CUDA / TensorRT 需匹配的 NVIDIA 设备及额外运行库。")
        note.setWordWrap(True)
        layout.addWidget(note)

    def browse(self, key):
        if key.endswith("_python"):
            path, _ = QFileDialog.getOpenFileName(
                self, "选择独立 Python", self.fields[key].text(), "Python (python.exe)"
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "选择目录", self.fields[key].text())
        if path:
            self.fields[key].setText(path)

    def save(self):
        for key, field in self.fields.items():
            # Keep defaults relative to the application on portable installations.
            value = field.text()
            default = (
                runtime_paths({}).get(key.removesuffix("_python"))
                if key.endswith("_python")
                else (application_root() / ("models" if key == "model_cache" else "projects"))
            )
            self.settings.values[key] = "" if default and Path(value) == default else value
        self.settings.values.update(
            capture_hotkey=self.hotkey.keySequence().toString(),
            capture_interval=self.interval.value(),
            restore_project=self.restore.isChecked(),
        )
        self.settings.save()
        self.saved.emit()

    def display_report(self, report):
        self.report.setPlainText(
            json.dumps(report, ensure_ascii=False, indent=2) if isinstance(report, dict) else str(report)
        )
