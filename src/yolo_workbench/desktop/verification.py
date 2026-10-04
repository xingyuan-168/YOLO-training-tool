from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .canvas import PALETTE
from .widgets import PathField


class DetectionView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#18212f"))
        self.setMinimumSize(240, 160)
        self.original = QImage()
        self.detections = []

    def display(self, image, detections):
        if image.isNull():
            return
        self.original = image.copy()
        self.detections = list(detections)
        painted = image.convertToFormat(QImage.Format.Format_RGB32)
        painter = QPainter(painted)
        line_width = max(2, image.width() // 600)
        font = QFont("Noto Sans SC")
        font.setPixelSize(max(13, image.width() // 65))
        painter.setFont(font)
        for detection in detections:
            color = QColor(PALETTE[int(detection["class_id"]) % len(PALETTE)])
            painter.setPen(QPen(color, line_width))
            x1, y1, x2, y2 = detection["xyxy"]
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            text = f"{detection.get('class_name', detection['class_id'])} {detection['confidence']:.2f}"
            painter.drawText(int(x1 + 2), int(max(font.pixelSize(), y1 - 4)), text)
        painter.end()
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(painted))
        self.scene().setSceneRect(QRectF(0, 0, image.width(), image.height()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.original.isNull():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class VerificationPage(QWidget):
    job_requested = Signal(str, object, str)
    evaluate_requested = Signal(object)
    stop_requested = Signal()
    feedback_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        model_line = QHBoxLayout()
        model_line.addWidget(QLabel("模型"))
        self.model = PathField("PT / ONNX / NCNN param / 导出包中的 model.json")
        self.model.button.clicked.connect(self.browse_model)
        model_line.addWidget(self.model, 1)
        self.family = QComboBox()
        self.family.addItems(["yolov8", "yolo11", "yolo26"])
        model_line.addWidget(self.family)
        layout.addLayout(model_line)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        live = QWidget()
        live_layout = QHBoxLayout(live)
        split = QSplitter()
        live_layout.addWidget(split)
        controls = QWidget()
        form = QFormLayout(controls)
        form.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.source_form = form
        self.backend = QComboBox()
        for name, value in [
            ("CQ_AI 部署验证", "cq"),
            ("NCNN 部署验证", "ncnn"),
            ("通用 PT", "pt"),
            ("通用 ONNX", "onnx"),
        ]:
            self.backend.addItem(name, value)
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["auto", "cpu", "0"])
        self.input_size = QSpinBox()
        self.input_size.setRange(32, 4096)
        self.input_size.setSingleStep(32)
        self.input_size.setValue(640)
        self.labels = PathField("自动读取模型类别；可指定 labels.txt")
        self.labels.button.clicked.connect(self.browse_labels)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.5)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0, 1)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.45)
        self.iou_note = QLabel("CQ_AI 固定 0.45")
        self.iou_note.setWordWrap(True)
        self.source = QComboBox()
        for name, value in [
            ("图片", "image"),
            ("图片目录", "folder"),
            ("视频", "video"),
            ("窗口", "window"),
            ("桌面", "desktop"),
        ]:
            self.source.addItem(name, value)
        self.source_path = PathField("选择输入图片、目录或视频")
        self.source_path.button.clicked.connect(self.browse_source)
        self.windows = QComboBox()
        self.windows.setMinimumContentsLength(15)
        self.windows.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.refresh_windows = QPushButton("刷新窗口")
        self.refresh_windows.clicked.connect(self.update_windows)
        self.monitor = QSpinBox()
        self.monitor.setRange(1, 16)
        self.client = QCheckBox("仅客户区")
        self.client.setChecked(True)
        self.max_fps = QSpinBox()
        self.max_fps.setRange(1, 120)
        self.max_fps.setValue(10)
        for title, widget in [
            ("验证后端", self.backend),
            ("设备", self.device),
            ("输入尺寸", self.input_size),
            ("类别文件", self.labels),
            ("置信度", self.confidence),
            ("IoU", self.iou),
            ("", self.iou_note),
            ("输入来源", self.source),
            ("文件 / 目录", self.source_path),
            ("目标窗口", self.windows),
            ("", self.refresh_windows),
            ("显示器", self.monitor),
            ("", self.client),
            ("显示帧率上限", self.max_fps),
        ]:
            form.addRow(title, widget)
        actions = QHBoxLayout()
        self.once = QPushButton("单次识别")
        self.once.setProperty("primary", True)
        self.continuous = QPushButton("连续识别")
        actions.addWidget(self.once)
        actions.addWidget(self.continuous)
        form.addRow(actions)
        stop = QPushButton("停止识别")
        stop.clicked.connect(self.stop_requested)
        form.addRow(stop)
        feedback = QPushButton("保存问题样本 → 待标注")
        feedback.clicked.connect(self.feedback_requested)
        form.addRow(feedback)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(380)
        scroll.setWidget(controls)
        split.addWidget(scroll)
        result = QWidget()
        result_layout = QVBoxLayout(result)
        self.view = DetectionView()
        self.runtime = QLabel("选择模型和输入，开始部署验证")
        self.runtime.setWordWrap(True)
        self.results = QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(["类别", "置信度", "原图边框 xyxy"])
        self.results.setMaximumHeight(180)
        self.results.horizontalHeader().setStretchLastSection(True)
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        result_layout.addWidget(self.runtime)
        result_layout.addWidget(self.view, 1)
        result_layout.addWidget(self.results)
        split.addWidget(result)
        split.setSizes([300, 900])
        tabs.addTab(live, "现场识别")
        evaluation = QWidget()
        eval_layout = QVBoxLayout(evaluation)
        eval_info = QLabel(
            "训练评估 · Ultralytics\n使用当前项目冻结的数据快照，计算 Precision、Recall、mAP50 和 mAP50–95。"
        )
        eval_info.setWordWrap(True)
        eval_layout.addWidget(eval_info)
        self.eval_split = QComboBox()
        self.eval_split.addItems(["val", "test"])
        eval_layout.addWidget(self.eval_split)
        evaluate = QPushButton("评估当前项目数据集")
        evaluate.setProperty("primary", True)
        evaluate.clicked.connect(self.request_evaluate)
        eval_layout.addWidget(evaluate)
        self.eval_result = QLabel("尚未运行评估")
        self.eval_result.setWordWrap(True)
        self.eval_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        eval_layout.addWidget(self.eval_result)
        eval_layout.addStretch()
        tabs.addTab(evaluation, "数据集评估")
        export = QWidget()
        export_layout = QFormLayout(export)
        self.profile = QComboBox()
        for name, value in [("通用模型", "generic"), ("CQ_AI / OpenCV", "cq"), ("AScript v8", "ascript_v8")]:
            self.profile.addItem(name, value)
        self.export_format = QComboBox()
        self.export_format.addItems(["pt", "onnx", "ncnn"])
        self.export_size = QComboBox()
        self.export_size.addItems(["640", "320"])
        self.export_note = QLabel("每次创建新的导出目录；结构和隔离加载检查成功后才成为可用模型。")
        self.export_note.setWordWrap(True)
        export_layout.addRow("使用端", self.profile)
        export_layout.addRow("格式", self.export_format)
        export_layout.addRow("固定输入尺寸", self.export_size)
        export_layout.addRow(self.export_note)
        self.export_button = QPushButton("导出并验证模型包")
        self.export_button.setProperty("primary", True)
        self.export_button.clicked.connect(self.request_export)
        export_layout.addRow(self.export_button)
        self.export_result = QLabel("尚未导出")
        self.export_result.setWordWrap(True)
        self.export_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        export_layout.addRow(self.export_result)
        tabs.addTab(export, "模型导出")
        benchmark = QWidget()
        benchmark_layout = QFormLayout(benchmark)
        self.warmup = QSpinBox()
        self.warmup.setRange(1, 1000)
        self.warmup.setValue(20)
        self.iterations = QSpinBox()
        self.iterations.setRange(2, 100000)
        self.iterations.setValue(200)
        benchmark_layout.addRow(
            QLabel("使用现场识别页的模型、图片、后端、尺寸和设备。固定 FP32；预热后计时。")
        )
        benchmark_layout.addRow("预热次数", self.warmup)
        benchmark_layout.addRow("采样次数", self.iterations)
        benchmark_button = QPushButton("开始性能测试")
        benchmark_button.clicked.connect(self.request_benchmark)
        benchmark_layout.addRow(benchmark_button)
        self.benchmark_result = QLabel("尚未测量；结果将报告 P50 / P95、吞吐量和端到端耗时。")
        self.benchmark_result.setWordWrap(True)
        self.benchmark_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        benchmark_layout.addRow(self.benchmark_result)
        tabs.addTab(benchmark, "性能测试")
        self.backend.currentIndexChanged.connect(self.update_capabilities)
        self.family.currentTextChanged.connect(self.update_capabilities)
        self.source.currentIndexChanged.connect(self.update_source)
        self.profile.currentIndexChanged.connect(self.update_profile)
        self.once.clicked.connect(lambda: self.request_infer(False))
        self.continuous.clicked.connect(lambda: self.request_infer(True))
        self.model.changed.connect(self.read_manifest)
        self.update_capabilities()
        self.update_source()

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型或模型包清单", "", "模型 (*.pt *.onnx *.param *.json);;所有文件 (*)"
        )
        if path:
            self.model.setText(path)

    def read_manifest(self, path):
        candidate = Path(path)
        inferred = {".pt": "pt", ".onnx": "cq", ".param": "ncnn"}.get(candidate.suffix.lower())
        if inferred:
            self.backend.setCurrentIndex(self.backend.findData(inferred))
        for family in ("yolov8", "yolo11", "yolo26"):
            if candidate.stem.startswith(family):
                self.family.setCurrentText(family)
                if family == "yolo26" and inferred == "cq":
                    self.backend.setCurrentIndex(self.backend.findData("onnx"))
        if candidate.is_dir():
            candidates = [candidate / "model.json", candidate / "manifest.json"]
        elif candidate.suffix == ".json":
            candidates = [candidate]
        else:
            candidates = [
                candidate.with_suffix(candidate.suffix + ".manifest.json"),
                candidate.with_suffix(".manifest.json"),
                candidate.parent / "model.json",
                candidate.parent / "manifest.json",
            ]
        for manifest in candidates:
            if manifest.is_file():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if data.get("family"):
                        self.family.setCurrentText(data["family"])
                    if data.get("input_size"):
                        self.input_size.setValue(int(data["input_size"]))
                    shape = data.get("input_shape", [])
                    if len(shape) == 4 and isinstance(shape[2], int):
                        self.input_size.setValue(shape[2])
                    fmt = data.get("format")
                    backend = "cq" if data.get("profile") == "cq" else fmt or inferred
                    index = self.backend.findData(backend)
                    if index >= 0:
                        self.backend.setCurrentIndex(index)
                except (OSError, ValueError, TypeError):
                    pass
                break
        sibling = candidate.parent / "labels.txt"
        if sibling.is_file():
            self.labels.setText(sibling)
        elif path:
            self.labels.setText("")

    def browse_labels(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择类别顺序文件", "", "类别 (*.txt *.names)")
        if path:
            self.labels.setText(path)

    def browse_source(self):
        if self.source.currentData() == "folder":
            path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择输入",
                "",
                "图片 / 视频 (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mkv);;所有文件 (*)",
            )
        if path:
            self.source_path.setText(path)

    def update_source(self):
        source = self.source.currentData()
        self.source_path.setEnabled(source in ("image", "folder", "video"))
        self.windows.setEnabled(source == "window")
        self.refresh_windows.setEnabled(source == "window")
        self.monitor.setEnabled(source == "desktop")
        self.client.setEnabled(source == "window")
        for widget, visible in (
            (self.source_path, source in ("image", "folder", "video")),
            (self.windows, source == "window"),
            (self.refresh_windows, source == "window"),
            (self.monitor, source == "desktop"),
            (self.client, source == "window"),
        ):
            self.source_form.setRowVisible(widget, visible)

    def update_windows(self):
        from ..capture import enumerate_windows

        current = self.windows.currentData()
        self.windows.clear()
        for window in enumerate_windows():
            self.windows.addItem(window["title"], window["hwnd"])
        index = self.windows.findData(current)
        if index >= 0:
            self.windows.setCurrentIndex(index)

    def update_capabilities(self):
        cq = self.backend.currentData() == "cq"
        end_to_end = self.family.currentText() == "yolo26" and self.backend.currentData() != "ncnn"
        self.iou.setEnabled(not cq and not end_to_end)
        if cq:
            self.iou.setValue(0.45)
        self.iou_note.setText(
            "CQ_AI 固定 0.45" if cq else "YOLO26 端到端输出，无 NMS" if end_to_end else "此后端支持 NMS 阈值"
        )

    def apply_capabilities(self, capabilities, manifest):
        supported = capabilities.get("iou", self.iou.isEnabled())
        self.iou.setEnabled(bool(supported))
        if capabilities.get("end_to_end"):
            self.iou_note.setText("此模型为端到端输出，无 NMS")
        elif self.backend.currentData() == "cq":
            self.iou.setValue(0.45)
            self.iou_note.setText("CQ_AI 原生 IoU 固定 0.45")
        elif supported:
            self.iou_note.setText("运行时确认：支持 NMS 阈值")
        shape = manifest.get("input_shape", [])
        if len(shape) == 4 and isinstance(shape[2], int):
            self.input_size.setValue(shape[2])

    def update_profile(self):
        profile = self.profile.currentData()
        self.export_format.setEnabled(profile == "generic")
        if profile != "generic":
            self.export_format.setCurrentText("onnx" if profile == "cq" else "ncnn")
        self.export_size.setEnabled(profile != "ascript_v8")
        if profile == "ascript_v8":
            self.export_size.setCurrentText("640")
        self.export_note.setText(
            {
                "generic": "通用模型按真实输出结构生成清单，并验证实际加载。",
                "cq": "YOLOv8 / YOLO11 · FP32 · Batch=1 · opset12 · 无内置 NMS · 固定 320/640。",
                "ascript_v8": "仅 YOLOv8 · 固定 640 · 解码后 [4+C,8400]；包含加载示例。",
            }[profile]
        )

    def parameters(self):
        model = self.model.text()
        if not model or not Path(model).exists():
            raise ValueError("请选择存在的模型或模型包")
        source = self.source.currentData()
        parameters = {
            "model": str(Path(model).resolve()),
            "family": self.family.currentText(),
            "backend": self.backend.currentData(),
            "device": self.device.currentText(),
            "input_size": self.input_size.value(),
            "confidence": self.confidence.value(),
            "iou": 0.45 if self.backend.currentData() == "cq" else self.iou.value(),
            "source": source,
            "source_path": self.source_path.text(),
            "hwnd": self.windows.currentData(),
            "monitor_index": self.monitor.value(),
            "client_only": self.client.isChecked(),
            "max_fps": self.max_fps.value(),
        }
        if self.labels.text():
            parameters["classes"] = [
                line.strip()
                for line in Path(self.labels.text()).read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        if source in ("image", "folder", "video") and not Path(self.source_path.text()).exists():
            raise ValueError("请选择存在的输入文件或目录")
        if source == "window" and not parameters["hwnd"]:
            raise ValueError("请刷新并选择目标窗口")
        return parameters

    def guarded(self, function):
        from PySide6.QtWidgets import QMessageBox

        try:
            function()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "检查输入", str(exc))

    def request_infer(self, continuous):
        def request():
            parameters = self.parameters()
            kind = "infer_stream" if continuous or parameters["source"] in ("video", "folder") else "infer"
            self.job_requested.emit(
                kind, parameters, "train" if parameters["backend"] == "pt" else "inference"
            )

        self.guarded(request)

    def request_evaluate(self):
        def request():
            if not Path(self.model.text()).exists():
                raise ValueError("请选择可由 Ultralytics 加载的模型")
            self.evaluate_requested.emit(
                {
                    "model": self.model.text(),
                    "imgsz": self.input_size.value(),
                    "batch": 4,
                    "device": "cpu" if self.device.currentText() == "auto" else self.device.currentText(),
                    "split": self.eval_split.currentText(),
                }
            )

        self.guarded(request)

    def request_export(self):
        def request():
            if not Path(self.model.text()).is_file():
                raise ValueError("请选择本地 PT 权重")
            self.job_requested.emit(
                "export",
                {
                    "model": self.model.text(),
                    "format": self.export_format.currentText(),
                    "profile": self.profile.currentData(),
                    "family": self.family.currentText(),
                    "imgsz": int(self.export_size.currentText()),
                    "device": "cpu",
                },
                "train",
            )

        self.guarded(request)

    def request_benchmark(self):
        def request():
            parameters = self.parameters()
            if parameters["source"] != "image":
                raise ValueError("性能测试请选择固定图片输入")
            parameters.update(warmup=self.warmup.value(), iterations=self.iterations.value())
            self.job_requested.emit(
                "benchmark", parameters, "train" if parameters["backend"] == "pt" else "inference"
            )

        self.guarded(request)

    def show_frame(self, image, data):
        detections = data.get("detections", [])
        self.view.display(image, detections)
        self.results.setRowCount(min(len(detections), 2000))
        for row, detection in enumerate(detections[:2000]):
            values = [
                str(detection.get("class_name", detection["class_id"])),
                f"{detection['confidence']:.3f}",
                ", ".join(f"{v:.1f}" for v in detection["xyxy"]),
            ]
            for column, value in enumerate(values):
                self.results.setItem(row, column, QTableWidgetItem(value))
        runtime = data.get("runtime", {})
        self.runtime.setText(
            f"部署验证 · {len(detections)} 个目标 · {data.get('elapsed_ms', 0):.1f} ms · {runtime}"
        )
