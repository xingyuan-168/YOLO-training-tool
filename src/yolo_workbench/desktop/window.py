from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..dataset import DatasetService
from ..runtime import Settings, application_root, data_root, model_cache, run_process, runtime_paths
from ..storage import atomic_write, json_text
from .canvas import AnnotationCanvas
from .hotkeys import CaptureHotkey
from .training_panel import TrainingPanel
from .utility_pages import LogPage, SettingsPage
from .verification import VerificationPage
from .widgets import AssetListModel, TaskThread, text_dialog

STATE_NAMES = {
    "preparing": "准备中",
    "running": "运行中",
    "stopping": "停止中",
    "stopped": "已停止",
    "succeeded": "成功",
    "failed": "失败",
    "interrupted": "意外中断",
    "queued": "等待中",
}
KIND_NAMES = {
    "train": "训练",
    "evaluate": "训练评估",
    "export": "模型导出",
    "infer": "部署验证",
    "infer_stream": "连续识别",
    "capture": "截图",
    "capture_stream": "定时截图",
    "benchmark": "性能测试",
}


class MainWindow(QMainWindow):
    def __init__(self, settings=None, *, restore=True):
        super().__init__()
        self.settings = settings or Settings()
        self.service = None
        self.current_asset = None
        self.dirty = False
        self.task_thread = None
        self._closing = False
        self._load_selection = False
        self._navigation_row = 0
        self._latest_capture = {}
        self._last_frame = None
        self._last_frame_data = {}
        self._curves = {}
        self._metric_history = {}
        self._job_items = {}
        self._capture_jobs = set()
        self._capture_import_queue = []
        self._pending_snapshot = None
        self._frame_reader = None
        self.setWindowTitle("YOLO 工作台 · 标注 · 训练 · 验证")
        self.resize(1366, 850)
        self.setMinimumSize(1080, 650)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 10, 14, 6)
        heading = QHBoxLayout()
        title = QLabel("YOLO 工作台")
        title.setStyleSheet("font-size:22px;font-weight:700;")
        heading.addWidget(title)
        subtitle = QLabel("本地数据与模型工作流")
        subtitle.setProperty("muted", True)
        heading.addWidget(subtitle)
        heading.addStretch()
        self.project_title = QLabel("尚未打开项目")
        heading.addWidget(self.project_title)
        layout.addLayout(heading)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.annotation_page = self.build_annotation()
        self.tabs.addTab(self.annotation_page, "标注训练")
        self.verification = VerificationPage()
        self.tabs.addTab(self.verification, "验证")
        self.logs = LogPage()
        self.tabs.addTab(self.logs, "日志")
        self.preferences = SettingsPage(self.settings)
        self.tabs.addTab(self.preferences, "设置")
        self.build_monitor()
        self.status = QLabel("就绪 · 新建或打开项目开始")
        self.resource_status = QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self.cancel_data = self.button(
            "取消当前数据操作", lambda: self.task_thread.cancel.set() if self.task_thread else None
        )
        self.cancel_data.hide()
        self.statusBar().addPermanentWidget(self.cancel_data)
        self.statusBar().addPermanentWidget(self.resource_status)
        self.autosave = QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.setInterval(350)
        self.autosave.timeout.connect(self.save_annotation)
        self.canvas.boxes_changed.connect(self.annotation_changed)
        self.canvas.information.connect(self.status.setText)
        self.canvas.navigate.connect(self.navigate)
        self.canvas.next_pending.connect(self.next_pending)
        self.canvas.save_requested.connect(self.save_annotation)
        self.canvas.selection_changed.connect(self.selection_changed)
        self.training.start_requested.connect(self.prepare_training)
        self.verification.job_requested.connect(self.start_job)
        self.verification.evaluate_requested.connect(self.prepare_evaluation)
        self.verification.stop_requested.connect(self.stop_inference)
        self.verification.feedback_requested.connect(self.save_feedback)
        self.preferences.saved.connect(self.settings_saved)
        self.preferences.diagnose.connect(self.diagnose)
        self.preferences.prepare_models.connect(self.prepare_models)
        self.hotkey = CaptureHotkey(QApplication.instance())
        self.hotkey.signals.activated.connect(lambda: self.capture(False))
        self.settings_saved(initial=True)
        self.set_manager(data_root() / "session")
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(80)
        self.poll_timer.timeout.connect(self.poll_jobs)
        self.poll_timer.start()
        self.resource_timer = QTimer(self)
        self.resource_timer.setInterval(1500)
        self.resource_timer.timeout.connect(self.resources)
        self.resource_timer.start()
        if self.settings.error:
            self.log("WARNING", self.settings.error)
        if restore and self.settings.values.get("restore_project", True):
            last = self.settings.values.get("last_project")
            if last and (Path(last) / "project.json").exists():
                QTimer.singleShot(0, lambda: self.open_project(Path(last)))

    def button(self, text, handler, *, primary=False):
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.clicked.connect(handler)
        return button

    def build_annotation(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.button("新建项目", self.new_project))
        open_button = self.button("打开项目", self.open_dialog)
        menu = QMenu(open_button)
        menu.addAction("选择项目目录…", self.open_dialog)
        for path in self.settings.values.get("recent_projects", []):
            menu.addAction(path, lambda p=path: self.open_project(Path(p)))
        open_button.setMenu(menu)
        toolbar.addWidget(open_button)
        import_button = self.button("导入数据", lambda: self.import_data("directory"), primary=True)
        import_menu = QMenu(import_button)
        for text, kind in [
            ("图片文件…", "images"),
            ("图片 / YOLO 目录…", "directory"),
            ("YOLO 数据集 ZIP…", "zip"),
        ]:
            import_menu.addAction(text, lambda k=kind: self.import_data(k))
        import_button.setMenu(import_menu)
        toolbar.addWidget(import_button)
        toolbar.addWidget(self.button("数据检查", self.validate_data))
        toolbar.addWidget(self.button("数据划分", self.split_dialog))
        toolbar.addStretch()
        toolbar.addWidget(
            self.button("训练监控", lambda: self.monitor.setVisible(not self.monitor.isVisible()))
        )
        layout.addLayout(toolbar)
        capture_bar = QHBoxLayout()
        capture_bar.addWidget(QLabel("截图来源"))
        self.capture_source = QComboBox()
        self.capture_source.addItem("桌面 1", {"source": "desktop", "monitor_index": 1})
        self.capture_source.setMinimumContentsLength(16)
        self.capture_source.setMaximumWidth(300)
        self.capture_source.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        capture_bar.addWidget(self.capture_source)
        capture_bar.addWidget(self.button("刷新窗口", self.refresh_capture_sources))
        capture_bar.addWidget(self.button("指向选窗", self.pick_window))
        capture_bar.addWidget(self.button("预览", lambda: self.capture(False, preview=True)))
        capture_bar.addWidget(self.button("单张截图", lambda: self.capture(False)))
        self.capture_interval = QSpinBox()
        self.capture_interval.setRange(1, 3600)
        self.capture_interval.setSuffix(" 秒")
        self.capture_interval.setValue(self.settings.values.get("capture_interval", 5))
        capture_bar.addWidget(self.capture_interval)
        self.capture_toggle = self.button("开始定时截图", self.toggle_capture)
        capture_bar.addWidget(self.capture_toggle)
        capture_bar.addStretch()
        self.save_status = QLabel("无图片")
        capture_bar.addWidget(self.save_status)
        layout.addLayout(capture_bar)
        self.splitter = QSplitter()
        layout.addWidget(self.splitter, 1)
        left = QWidget()
        left.setMinimumWidth(180)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        self.image_count = QLabel("图片 · 0")
        left_layout.addWidget(self.image_count)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索图片名称")
        self.filter = QComboBox()
        for name, value in [
            ("全部图片", None),
            ("未标注", "pending"),
            ("已标注", "labeled"),
            ("已确认无目标", "empty"),
            ("回收区", "deleted"),
        ]:
            self.filter.addItem(name, value)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.filter)
        self.images = QListView()
        self.image_model = AssetListModel(self)
        self.images.setModel(self.image_model)
        self.images.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.images.setUniformItemSizes(True)
        self.images.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.images.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.images.setAlternatingRowColors(True)
        self.images.selectionModel().currentChanged.connect(self.image_selected)
        left_layout.addWidget(self.images, 1)
        row = QHBoxLayout()
        row.addWidget(self.button("回收 / 恢复", self.recycle_selected))
        row.addWidget(self.button("下一未标注", self.next_pending))
        left_layout.addLayout(row)
        self.search.textChanged.connect(self.refresh_images)
        self.filter.currentIndexChanged.connect(self.refresh_images)
        self.splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 0, 4, 0)
        canvas_toolbar = QHBoxLayout()
        for name, handler in [
            ("绘框 W", lambda: self.set_mode("draw")),
            ("选择 V", lambda: self.set_mode("select")),
            ("适应 F", lambda: self.canvas.fit_image()),
            ("撤销", lambda: self.canvas.undo_stack.undo()),
            ("重做", lambda: self.canvas.undo_stack.redo()),
        ]:
            canvas_toolbar.addWidget(self.button(name, handler))
        canvas_toolbar.addStretch()
        self.hide_labels = QCheckBox("隐藏标签")
        self.hide_labels.toggled.connect(self.toggle_labels)
        canvas_toolbar.addWidget(self.hide_labels)
        center_layout.addLayout(canvas_toolbar)
        self.canvas = AnnotationCanvas()
        center_layout.addWidget(self.canvas, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self.button("上一张", lambda: self.navigate(-1)))
        bottom.addWidget(self.button("下一张", lambda: self.navigate(1)))
        bottom.addStretch()
        bottom.addWidget(self.button("确认无目标", self.confirm_empty))
        bottom.addWidget(self.button("保存 Ctrl+S", self.save_annotation))
        center_layout.addLayout(bottom)
        tip = QLabel("W 绘框 · V 选择 · 滚轮缩放 · 空格平移 · 1–9 类别 · Del 删除 · A/D 切图")
        tip.setProperty("muted", True)
        tip.setWordWrap(True)
        center_layout.addWidget(tip)
        self.splitter.addWidget(center)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(275)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        classes = QGroupBox("类别")
        classes_layout = QVBoxLayout(classes)
        self.classes = QListWidget()
        self.classes.setMaximumHeight(150)
        self.classes.currentRowChanged.connect(self.class_selected)
        classes_layout.addWidget(self.classes)
        class_buttons = QHBoxLayout()
        class_buttons.addWidget(self.button("管理类别", self.manage_classes))
        class_buttons.addWidget(self.button("恢复迁移", self.restore_migration))
        classes_layout.addLayout(class_buttons)
        right_layout.addWidget(classes)
        self.training = TrainingPanel(self.settings)
        right_layout.addWidget(self.training)
        right_layout.addStretch()
        right_scroll.setWidget(right)
        self.splitter.addWidget(right_scroll)
        self.splitter.setSizes([225, 750, 310])
        self.splitter.setStretchFactor(1, 1)
        return page

    def build_monitor(self):
        self.monitor = QDockWidget("任务与训练监控", self)
        self.monitor.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        self.job_progress = QProgressBar()
        self.job_progress.setValue(0)
        self.job_caption = QLabel("尚未启动任务")
        row.addWidget(self.job_caption, 1)
        row.addWidget(self.job_progress, 1)
        row.addWidget(self.button("停止并保存", lambda: self.stop_selected(False)))
        row.addWidget(self.button("强制结束", lambda: self.stop_selected(True)))
        row.addWidget(self.button("断点恢复", self.resume_selected))
        row.addWidget(self.button("权重续训", self.finetune_selected))
        row.addWidget(self.button("打开产物", self.open_job_directory))
        layout.addLayout(row)
        split = QSplitter()
        self.jobs_tree = QTreeWidget()
        self.jobs_tree.setHeaderLabels(["任务", "状态", "创建时间 / ID"])
        self.jobs_tree.setMinimumWidth(330)
        self.jobs_tree.itemDoubleClicked.connect(self.job_details)
        split.addWidget(self.jobs_tree)
        import pyqtgraph as pg

        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Epoch")
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        split.addWidget(self.plot)
        split.setSizes([440, 800])
        layout.addWidget(split)
        self.monitor.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.monitor)
        self.monitor.hide()

    def log(self, level, message, task=""):
        self.logs.append(level, message, task)
        self.status.setText(str(message).splitlines()[0][:240])
        path = data_root() / "workbench.log"
        if path.exists() and path.stat().st_size > 5_000_000:
            path.replace(path.with_suffix(".previous.log"))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now().isoformat()} [{level}] {task} {message}\n")

    def fail(self, message):
        self.log("ERROR", message)
        QMessageBox.warning(self, "操作未完成", str(message))

    def require_project(self):
        if self.task_thread:
            self.fail("数据操作进行中，请等待当前操作完成。")
            return False
        if not self.service:
            self.fail("请先新建或打开项目。")
            return False
        return True

    def new_project(self):
        if self.task_thread:
            return
        parent = QFileDialog.getExistingDirectory(
            self,
            "选择新项目的父目录",
            self.settings.values.get("projects_dir") or str(application_root() / "projects"),
        )
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "新建项目", "项目名称")
        if not ok or not name.strip():
            return
        if name in (".", "..") or any(c in name for c in '\\/:*?"<>|'):
            self.fail("项目名称不能包含路径分隔符或 Windows 非法字符")
            return
        classes, ok = QInputDialog.getMultiLineText(
            self, "初始类别", "每行一个类别；导入空项目数据时可采用数据集类别", "目标"
        )
        if not ok:
            return
        try:
            path = Path(parent) / name.strip()
            self.protect_input(path)
            created = DatasetService.create(
                path, name, [v.strip() for v in classes.splitlines() if v.strip()]
            )
            created.close()
            self.open_project(path)
        except Exception as exc:
            self.fail(str(exc))

    def open_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "打开工作台项目",
            self.settings.values.get("projects_dir") or str(application_root() / "projects"),
        )
        if path:
            self.open_project(Path(path))

    @staticmethod
    def protect_input(path):
        if path.resolve().is_relative_to((application_root() / "input").resolve()):
            raise ValueError("input 是只读材料目录，请选择项目管理区域")

    def open_project(self, path):
        if self.task_thread:
            return
        if hasattr(self, "manager") and self.manager.active_jobs():
            self.fail("任务运行期间请保持当前项目打开；任务结束后可以切换项目。")
            return
        if not self.save_annotation():
            return
        try:
            self.protect_input(path)
            if self.service and self.service.root == path.resolve():
                return
        except Exception as exc:
            self.fail(str(exc))
            return
        self.annotation_page.setEnabled(False)

        def prepare(progress, cancel):
            progress({"message": "读取项目记录并恢复索引"})
            with DatasetService(path) as service:
                return {"recovered": service.recovered}

        def opened(result):
            self.annotation_page.setEnabled(True)
            if self._closing:
                return
            try:
                service = DatasetService(path, index_prepared=True)
                if self.service:
                    self.image_model.configure(None)
                    self.service.close()
                self.service = service
                self.current_asset = None
                self.dirty = False
                self.set_manager(service.root)
                self.settings.recent(path)
                self.project_title.setText(service.project["name"])
                self.project_title.setToolTip(str(service.root))
                self.canvas.set_document(None, [], service.project["classes"])
                self.refresh_classes()
                self.refresh_images()
                self.training.reset_project()
                self.apply_project_settings()
                self.log(
                    "INFO",
                    f"已打开项目：{service.root}" + ("；已恢复未完成保存" if result["recovered"] else ""),
                )
                if self.image_model.rowCount():
                    self.images.setCurrentIndex(self.image_model.index(0))
            except Exception as exc:
                self.fail(str(exc))

        self.background(
            prepare, opened, lambda error: self.annotation_page.setEnabled(True), title="打开项目"
        )

    def apply_project_settings(self):
        values = (
            self.service.get_settings()
            if hasattr(self.service, "get_settings")
            else self.service.project.get("settings", {})
        )
        self.training.apply_config(values.get("training", {}))
        model = values.get("model", {})
        if "imgsz" in model or "input_size" in model:
            self.verification.input_size.setValue(int(model.get("imgsz", model.get("input_size"))))
        if model.get("family"):
            self.verification.family.setCurrentText(model["family"])
        if model.get("format"):
            value = {"cq_ai": "cq", "ascript_v8": "ncnn"}.get(model["format"], model["format"])
            index = self.verification.backend.findData(value)
            if index >= 0:
                self.verification.backend.setCurrentIndex(index)
        if "confidence" in model:
            self.verification.confidence.setValue(model["confidence"])

    def set_manager(self, root):
        from ..jobs import JobManager

        if hasattr(self, "manager"):
            self.manager.shutdown()
        root.mkdir(parents=True, exist_ok=True)
        self.manager = JobManager(root, runtime_paths(self.settings.values), application_root())
        self.jobs_tree.clear()
        self._job_items.clear()
        for job in self.manager.jobs.values():
            self.update_job(job)

    def refresh_images(self, *_):
        if not hasattr(self, "image_model") or self._load_selection:
            return
        if not self.save_annotation():
            return
        self._load_selection = True
        try:
            status = self.filter.currentData()
            self.image_model.configure(
                self.service,
                search=self.search.text(),
                status=None if status == "deleted" else status,
                deleted=status == "deleted",
            )
            self.image_count.setText(f"图片 · {self.image_model.total:,}")
        finally:
            self._load_selection = False

    def image_selected(self, current, previous):
        if self._load_selection or not self.service or not current.isValid():
            return
        if not self.save_annotation():
            self._load_selection = True
            self.images.setCurrentIndex(previous)
            self._load_selection = False
            return
        row = self.image_model.record(current.row())
        if not row:
            return
        try:
            self._navigation_row = current.row()
            record = self.service._record(row["id"])
            self.canvas.set_document(
                self.service.root / record["file"],
                self.service.load_boxes(row["id"]),
                self.service.project["classes"],
                confirmed_empty=record["status"] == "empty",
            )
            self.canvas.editable = not record.get("deleted")
            self.current_asset = row["id"]
            self.dirty = False
            self.save_status.setText(f"{record['name']} · {len(self.canvas.boxes)} 框 · 已保存")
            self.canvas.setFocus()
        except Exception as exc:
            self.fail(str(exc))

    def annotation_changed(self, _boxes):
        if self.current_asset:
            self.dirty = True
            self.save_status.setText(f"{len(self.canvas.boxes)} 框 · 等待保存")
            self.autosave.start()

    def save_annotation(self):
        if not self.dirty or not self.current_asset:
            return True
        if not self.service:
            return False
        try:
            self.service.save_boxes(
                self.current_asset, self.canvas.boxes, confirmed_empty=self.canvas.confirmed_empty
            )
            self.dirty = False
            self.save_status.setText(
                f"{len(self.canvas.boxes)} 框 · {'已确认无目标' if self.canvas.confirmed_empty else '已保存'}"
            )
            self.image_model.refresh_status(self.current_asset)
            status_filter = self.filter.currentData()
            if status_filter in ("pending", "labeled", "empty"):
                actual = self.service.get_asset(self.current_asset)["status"]
                if actual != status_filter:
                    QTimer.singleShot(0, self.refresh_images)
            return True
        except Exception as exc:
            self.save_status.setText("保存失败 · 请重试")
            self.fail(f"标注未保存：{exc}")
            return False

    def navigate(self, delta):
        index = self.images.currentIndex().row()
        if index < 0:
            index = self._navigation_row - (1 if delta > 0 else 0)
        target = index + delta
        if 0 <= target < self.image_model.rowCount():
            self.images.setCurrentIndex(self.image_model.index(target))

    def next_pending(self):
        if not self.require_project() or not self.save_annotation():
            return
        row = self.service.next_unreviewed(self.current_asset, search=self.search.text())
        if not row:
            self.status.setText("当前搜索范围内没有未标注图片")
            return
        self.filter.setCurrentIndex(1)
        self.refresh_images()
        # SQL ordering gives a direct position, without decoding or iterating all images.
        position = self.service.db.execute(
            "SELECT count(*) FROM assets WHERE deleted=0 AND status='pending' AND instr(name,?)>0 AND (name<? OR (name=? AND id<?))",
            (self.search.text(), row["name"], row["name"], row["id"]),
        ).fetchone()[0]
        self.images.setCurrentIndex(self.image_model.index(position))

    def set_mode(self, mode):
        self.canvas.mode = mode
        self.canvas.setFocus()
        self.status.setText("绘框模式" if mode == "draw" else "选择模式 · 拖动框或八个控制点")

    def toggle_labels(self, hidden):
        self.canvas.labels_visible = not hidden
        self.canvas.viewport().update()

    def confirm_empty(self):
        if not self.current_asset or not self.canvas.editable:
            return
        if (
            self.canvas.boxes
            and QMessageBox.question(
                self, "确认无目标", "这会清空当前图片标注，并确认图片没有目标。可通过撤销恢复。"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.canvas.confirm_empty()

    def refresh_classes(self):
        self.classes.blockSignals(True)
        self.classes.clear()
        if self.service:
            for index, name in enumerate(self.service.project["classes"]):
                self.classes.addItem(f"{index + 1}  {name}")
            self.canvas.classes = list(self.service.project["classes"])
            self.classes.setCurrentRow(min(self.canvas.current_class, self.classes.count() - 1))
            self.classes.setFixedHeight(max(60, min(150, self.classes.count() * 32 + 6)))
        self.classes.blockSignals(False)

    def class_selected(self, index):
        self.canvas.choose_class(index)
        self.canvas.setFocus()

    def selection_changed(self, index):
        if index >= 0 and index < len(self.canvas.boxes):
            self.classes.blockSignals(True)
            self.classes.setCurrentRow(self.canvas.boxes[index].class_id)
            self.classes.blockSignals(False)

    def manage_classes(self):
        if not self.require_project() or not self.save_annotation():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("类别管理 · 删除 / 重排会迁移当前标签")
        dialog.resize(440, 420)
        layout = QVBoxLayout(dialog)
        listing = QListWidget()
        for index, name in enumerate(self.service.project["classes"]):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            listing.addItem(item)
        layout.addWidget(QLabel("双击重命名；编号随列表顺序变化。历史训练快照保持原类别。"))
        layout.addWidget(listing)
        row = QHBoxLayout()

        def add():
            item = QListWidgetItem("新类别")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            listing.addItem(item)
            listing.setCurrentItem(item)
            listing.editItem(item)

        def move(delta):
            current = listing.currentRow()
            if 0 <= current + delta < listing.count():
                item = listing.takeItem(current)
                listing.insertItem(current + delta, item)
                listing.setCurrentRow(current + delta)

        for title, handler in [
            ("添加", add),
            ("上移", lambda: move(-1)),
            ("下移", lambda: move(1)),
            ("移除", lambda: listing.takeItem(listing.currentRow())),
        ]:
            row.addWidget(self.button(title, handler))
        layout.addLayout(row)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("预览迁移")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        names = [listing.item(i).text().strip() for i in range(listing.count())]
        mapping = {i: None for i in range(len(self.service.project["classes"]))}
        for target in range(listing.count()):
            old = listing.item(target).data(Qt.ItemDataRole.UserRole)
            if old is not None:
                mapping[old] = target

        def review(preview):
            if not text_dialog(
                self,
                "类别迁移预览",
                json_text(preview),
                approve=True,
                button="应用迁移",
                detail="确认后迁移当前工作标签；删除类别可能移除其标注。迁移记录可恢复。",
            ):
                return
            self.data_task(
                "迁移类别",
                lambda service, progress, cancel: service.migrate_classes(
                    names, mapping, allow_drop=True, progress=progress, cancel=cancel
                ),
            )

        self.data_task(
            "检查类别迁移",
            lambda service, progress, cancel: service.preview_class_migration(
                names, mapping, progress=progress, cancel=cancel
            ),
            review,
        )

    def restore_migration(self):
        if not self.require_project():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择类别迁移记录", str(self.service.root / "history"), "类别记录 (classes-*.json)"
        )
        if path:
            migration_id = Path(path).stem.removeprefix("classes-")
            self.data_task(
                "恢复类别迁移",
                lambda service, progress, cancel: service.restore_class_migration(migration_id),
            )

    def recycle_selected(self):
        if not self.require_project() or not self.save_annotation():
            return
        ids = [self.image_model.record(i.row())["id"] for i in self.images.selectedIndexes()]
        restore = self.filter.currentData() == "deleted"
        for asset_id in ids:
            self.service.recycle(asset_id, restore=restore)
        self.current_asset = None
        self.canvas.set_document(None, [], [])
        self.refresh_images()
        self.log("INFO", f"{'已恢复' if restore else '已移入回收区'} {len(ids)} 张图片")

    def import_data(self, kind):
        if not self.require_project():
            return
        if kind == "images":
            files, _ = QFileDialog.getOpenFileNames(
                self, "导入图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
            )
            if not files:
                return

            def import_images(service, progress, cancel):
                result = {"imported": 0, "duplicates": 0, "errors": []}
                for index, source in enumerate(files):
                    if cancel():
                        result["cancelled"] = True
                        break
                    try:
                        _, created = service.import_image(Path(source))
                        result["imported" if created else "duplicates"] += 1
                    except (ValueError, OSError) as exc:
                        result["errors"].append({"file": source, "message": str(exc)})
                    progress({"completed": index + 1, "total": len(files), "path": source})
                return result

            self.data_task("导入图片", import_images, self.import_complete)
            return
        if kind == "zip":
            source, _ = QFileDialog.getOpenFileName(self, "导入 YOLO ZIP", "", "ZIP (*.zip)")
        else:
            source = QFileDialog.getExistingDirectory(self, "导入图片 / 标准 YOLO 数据集")
        if not source:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("导入数据")
        dialog_layout = QVBoxLayout(dialog)
        label = QLabel(
            f"{source}\n图片将复制到受管理项目；重复内容去重。\n缺失标签保持未标注，源文件保持完整。"
        )
        label.setWordWrap(True)
        dialog_layout.addWidget(label)
        trust = QCheckBox("确认来源中的空标签已人工审核，可作为无目标图片")
        dialog_layout.addWidget(trust)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        from ..importers import import_dataset

        trust_empty = trust.isChecked()
        self.data_task(
            "导入数据集",
            lambda service, progress, cancel: import_dataset(
                service, Path(source), trust_empty_labels=trust_empty, progress=progress, cancel=cancel
            ),
            self.import_complete,
        )

    def import_complete(self, result):
        self.apply_project_settings()
        text_dialog(self, "导入结果", json_text(result))
        if self.image_model.rowCount():
            self.images.setCurrentIndex(self.image_model.index(0))

    def data_task(self, title, operation, callback=None):
        if not self.require_project() or not self.save_annotation():
            return
        root = self.service.root
        self.current_asset = None
        self.image_model.configure(None)
        self.service.close()
        self.service = None
        self.canvas.editable = False
        self.annotation_page.setEnabled(False)
        self.status.setText(title)
        self.job_progress.setRange(0, 0)

        def work(progress, cancel):
            with DatasetService(root) as service:
                return operation(service, progress, cancel)

        def finished(result, error=False):
            self.annotation_page.setEnabled(True)
            self.canvas.editable = True
            self.job_progress.setRange(0, 100)
            self.job_progress.setValue(0)
            try:
                self.service = DatasetService(root, index_prepared=True)
            except Exception as exc:
                self.service = None
                self.current_asset = None
                self.canvas.set_document(None, [], [])
                self.image_model.configure(None)
                self.log("ERROR", f"重新打开项目失败：{exc}\n原操作信息：{result}")
                self.project_title.setText("项目需要重新打开")
                return
            self.canvas.set_document(None, [], self.service.project["classes"])
            self.refresh_classes()
            self.refresh_images()
            queued, self._capture_import_queue = self._capture_import_queue, []
            for path, job in queued:
                self.import_capture(path, job)
            if not error:
                self.log("INFO", f"{title}完成")
                if callback and not self._closing:
                    callback(result)

        self.background(
            work, lambda result: finished(result), lambda error: finished(error, True), title=title
        )

    def background(self, function, callback=None, on_error=None, *, title="后台操作"):
        if self.task_thread:
            self.fail("已有后台数据操作正在进行")
            return
        thread = TaskThread(function, self)
        self.task_thread = thread
        self.cancel_data.show()

        def completed(result):
            self.task_thread = None
            self.cancel_data.hide()
            if callback:
                callback(result)

        def cancelled():
            self.task_thread = None
            self.cancel_data.hide()
            if on_error:
                on_error("操作已取消")
            self.log("INFO", f"{title}已取消；已提交的数据仍保留")

        def failed(error):
            self.task_thread = None
            self.cancel_data.hide()
            if on_error:
                on_error(error)
            self.fail(error)

        thread.progress.connect(lambda data: self.background_progress(title, data))
        thread.succeeded.connect(completed)
        thread.failed.connect(failed)
        thread.cancelled.connect(cancelled)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: QTimer.singleShot(0, self.close) if self._closing else None)
        thread.start()

    def background_progress(self, title, data):
        total, completed = data.get("total", 0), data.get("completed", 0)
        self.status.setText(f"{title} · {completed}/{total} · {data.get('path', data.get('message', ''))}")
        if total:
            self.job_progress.setRange(0, total)
            self.job_progress.setValue(completed)

    def validate_data(self):
        self.data_task(
            "数据检查",
            lambda service, progress, cancel: {
                "issues": service.validate(progress=progress, cancel=cancel),
                "statistics": service.statistics(progress=progress, cancel=cancel),
            },
            lambda result: text_dialog(self, "数据检查报告", json_text(result)),
        )

    def split_dialog(self):
        if not self.require_project():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("数据划分")
        layout = QFormLayout(dialog)
        train = QDoubleSpinBox()
        train.setRange(0.01, 0.99)
        train.setSingleStep(0.05)
        train.setValue(0.8)
        test = QDoubleSpinBox()
        test.setRange(0, 0.98)
        test.setSingleStep(0.05)
        seed = QSpinBox()
        seed.setRange(0, 2147483647)
        seed.setValue(42)
        grouped = QCheckBox("按采集会话分组，避免同源泄漏")
        grouped.setChecked(True)
        exclude = QCheckBox("明确排除未标注图片")
        layout.addRow("训练比例", train)
        layout.addRow("测试比例（余下为验证）", test)
        layout.addRow("随机种子", seed)
        layout.addRow(grouped)
        layout.addRow(exclude)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            kwargs = dict(
                seed=seed.value(),
                train_ratio=train.value(),
                test_ratio=test.value(),
                grouped=grouped.isChecked(),
                exclude_pending=exclude.isChecked(),
            )
            self.data_task(
                "划分数据",
                lambda service, progress, cancel: service.split(**kwargs, progress=progress, cancel=cancel),
                lambda result: text_dialog(
                    self,
                    "划分完成",
                    json_text({k: len(v) if isinstance(v, list) else v for k, v in result.items()}),
                ),
            )

    def frozen_dataset(self, service, parameters, *, exclude_pending=False, progress=None, cancel=None):
        issues = service.validate(progress=progress, cancel=cancel)
        blockers = [issue for issue in issues if issue["severity"] == "error"]
        if blockers:
            raise ValueError("数据检查未通过：\n" + json_text(blockers[:30]))
        split_path = service.root / "splits" / "default.json"
        split = (
            json.loads(split_path.read_text(encoding="utf-8"))
            if split_path.exists()
            else service.split(exclude_pending=exclude_pending, progress=progress, cancel=cancel)
        )
        if not exclude_pending:
            count = service.db.execute(
                "SELECT count(*) FROM assets WHERE deleted=0 AND status='pending'"
            ).fetchone()[0]
            if count:
                raise ValueError(f"存在 {count} 张未标注图片，请完成审核或明确排除")
        return str(service.snapshot(split, parameters, progress=progress, cancel=cancel))

    def prepare_training(self, request):
        if not self.require_project():
            return
        parameters = {
            k: v
            for k, v in request.items()
            if k not in ("effective", "exclude_pending", "ui_expert_yaml", "ui_advanced")
        }
        display = {
            "模型": request["model"],
            "生效参数": request["effective"],
            "排除未标注": request["exclude_pending"],
            "数据": "复用已保存划分；否则按 80/20、seed42、会话分组划分。训练使用冻结快照。",
        }
        if not text_dialog(
            self,
            "训练启动检查",
            yaml.safe_dump(display, allow_unicode=True, sort_keys=False),
            approve=True,
            button="冻结数据并训练",
        ):
            return
        if hasattr(self.service, "update_settings"):
            self.service.update_settings(
                {
                    "training": {
                        **request["config"],
                        "expert_yaml": request.get("ui_expert_yaml", ""),
                        "ui_advanced": request.get("ui_advanced", {}),
                    }
                }
            )
        self.data_task(
            "检查并冻结训练数据",
            lambda service, progress, cancel: self.frozen_dataset(
                service,
                request["effective"],
                exclude_pending=request["exclude_pending"],
                progress=progress,
                cancel=cancel,
            ),
            lambda snapshot: self.start_job("train", {**parameters, "snapshot": snapshot}, "train"),
        )

    def prepare_evaluation(self, parameters):
        if not self.require_project():
            return
        self.data_task(
            "冻结评估数据",
            lambda service, progress, cancel: self.frozen_dataset(
                service, parameters, progress=progress, cancel=cancel
            ),
            lambda snapshot: self.start_job("evaluate", {**parameters, "snapshot": snapshot}, "train"),
        )

    def start_job(self, kind, parameters, runtime="train"):
        try:
            if kind == "export" and self.service:
                parameters["output_dir"] = str(self.service.root / "exports")
            job = self.manager.start(kind, parameters, runtime=runtime)
            self.update_job(job)
            self.monitor.show()
            self.jobs_tree.setCurrentItem(self._job_items[job.id])
            self.log("INFO", f"已启动{KIND_NAMES.get(kind, kind)}", job.id)
            if kind == "train":
                self._metric_history.clear()
                self.plot.clear()
                self._curves.clear()
            return job
        except Exception as exc:
            self.fail(str(exc))
            return None

    def update_job(self, job):
        item = self._job_items.get(job.id)
        if not item:
            item = QTreeWidgetItem()
            item.setData(0, Qt.ItemDataRole.UserRole, job.id)
            self.jobs_tree.addTopLevelItem(item)
            self._job_items[job.id] = item
        item.setText(0, KIND_NAMES.get(job.kind, job.kind))
        item.setText(1, STATE_NAMES.get(job.state, job.state))
        item.setText(2, job.id[:24])
        item.setToolTip(0, str(job.run_dir))

    def selected_job(self):
        item = self.jobs_tree.currentItem()
        return self.manager.jobs.get(item.data(0, Qt.ItemDataRole.UserRole)) if item else None

    def stop_selected(self, force):
        job = self.selected_job()
        if not job:
            return
        if (
            force
            and QMessageBox.question(
                self,
                "强制结束",
                "当前 Epoch 尚未保存的进度会丢失。只结束本工具拥有的任务进程，保留最近完整检查点。",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            self.manager.stop(job.id, force=force)
            self.log(
                "WARNING" if force else "INFO", "已请求强制结束" if force else "等待安全保存点后停止", job.id
            )
        except Exception as exc:
            self.fail(str(exc))

    def resume_selected(self):
        job = self.selected_job()
        if not job or job.kind != "train":
            self.fail("请在监控中选择训练记录")
            return
        candidates = list(job.run_dir.rglob("resume.pt"))
        if not candidates:
            self.fail("此记录没有完整恢复检查点；可使用“权重续训”新建训练。")
            return
        parameters = {**job.parameters, "resume_checkpoint": str(candidates[-1]), "finetune": False}
        if self.service and not Path(parameters.get("snapshot", "")).is_dir():
            moved = self.service.root / "snapshots" / Path(parameters.get("snapshot", "")).name
            if moved.is_dir():
                parameters["snapshot"] = str(moved)
        self.start_job("train", parameters, "train")

    def finetune_selected(self):
        job = self.selected_job()
        if not job:
            self.tabs.setCurrentIndex(0)
            self.training.select_weights()
            return
        candidates = list(job.run_dir.rglob("best.pt")) or list(job.run_dir.rglob("last.pt"))
        if not candidates:
            self.fail("未找到可用于继续训练的权重")
            return
        self.training.weights.setText(candidates[-1])
        self.training.apply_config(job.parameters.get("config", {}))
        self.training.scratch.setChecked(False)
        self.tabs.setCurrentIndex(0)
        self.status.setText("已选择权重；检查参数后启动新的训练记录")

    def open_job_directory(self):
        job = self.selected_job()
        if job:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(job.run_dir)))

    def job_details(self, item, _column):
        job = self.manager.jobs[item.data(0, Qt.ItemDataRole.UserRole)]
        text_dialog(
            self,
            "任务记录",
            json_text(
                {
                    "id": job.id,
                    "kind": job.kind,
                    "state": job.state,
                    "parameters": job.parameters,
                    "result": job.result,
                    "error": job.error,
                    "directory": str(job.run_dir),
                }
            ),
        )

    def poll_jobs(self):
        try:
            events = self.manager.poll_events()
            latest_frames = {}
            for event in events:
                if event["type"] == "frame":
                    latest_frames[event["job_id"]] = event
                else:
                    self.handle_event(event)
            for event in latest_frames.values():
                self.handle_frame(event)
        except Exception:
            self.log("ERROR", traceback.format_exc())

    def handle_event(self, event):
        job = self.manager.jobs.get(event["job_id"])
        if not job:
            return
        kind, data = event["type"], event.get("data", {})
        self.update_job(job)
        if kind == "state":
            state = data.get("state", job.state)
            self.log(
                "ERROR" if state in ("failed", "interrupted") else "INFO",
                f"{KIND_NAMES.get(job.kind, job.kind)} · {STATE_NAMES.get(state, state)}",
                job.id,
            )
            if job.kind == "capture_stream" and state in ("succeeded", "stopped", "failed", "interrupted"):
                self.capture_toggle.setText("开始定时截图")
        elif kind == "error":
            self.log("ERROR", data.get("message", str(data)), job.id)
        elif kind == "model":
            self.verification.apply_capabilities(data.get("capabilities", {}), data.get("manifest", {}))
        elif kind == "log":
            self.logs.append(str(data.get("level", "INFO")).upper(), data.get("message", str(data)), job.id)
        elif kind == "progress":
            epoch, total = (
                data.get("epoch", data.get("completed", 0)),
                data.get("epochs", data.get("total", 0)),
            )
            if total:
                self.job_progress.setRange(0, int(total))
                self.job_progress.setValue(int(epoch))
            self.job_caption.setText(
                f"{KIND_NAMES.get(job.kind, job.kind)} · {epoch}/{total} · {data.get('message', '')} "
                f"已用 {data.get('elapsed_seconds', 0):.0f}s / 剩余 {data.get('eta_seconds', 0) or 0:.0f}s"
            )
        elif kind == "metrics":
            self.show_metrics(data)
        elif kind == "checkpoint":
            self.log("INFO", f"已保存检查点：{data}", job.id)
        elif kind == "source_status":
            self.log(
                "WARNING" if data.get("status") not in ("ready", "running", "active") else "INFO",
                str(data),
                job.id,
            )
        elif kind in ("capture_saved", "saved"):
            if job.id in self._capture_jobs:
                self.import_capture(data.get("path") or data.get("output_path"), job)
        elif kind == "result":
            self.job_result(job, data)

    def show_metrics(self, data):
        import pyqtgraph as pg

        epoch = data.get("epoch", 0)
        metrics = (
            {**data.get("losses", {}), **data.get("metrics", {})}
            if "metrics" in data or "losses" in data
            else data
        )
        colors = ["#2563eb", "#10b981", "#f59e0b", "#e11d48", "#8b5cf6", "#0891b2"]
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or key in ("epoch", "elapsed_seconds", "eta_seconds"):
                continue
            history = self._metric_history.setdefault(key, [])
            history.append((epoch, value))
            if key not in self._curves:
                self._curves[key] = self.plot.plot(
                    name=key, pen=pg.mkPen(colors[len(self._curves) % len(colors)], width=2)
                )
            self._curves[key].setData([p[0] for p in history], [p[1] for p in history])

    def handle_frame(self, event):
        data = event["data"]
        if self._frame_reader is None:
            from ..frames import SharedFrameReader

            self._frame_reader = SharedFrameReader()
        frame = self._frame_reader.read(data["frame"])
        if not frame:
            return
        fmt = (
            QImage.Format.Format_BGR888
            if frame.get("pixel_format", "BGR") == "BGR"
            else QImage.Format.Format_RGB888
        )
        image = QImage(frame["data"], frame["width"], frame["height"], frame["stride"], fmt).copy()
        self._last_frame = image
        self._last_frame_data = data
        self.verification.show_frame(image, data)

    def job_result(self, job, data):
        self.log(
            "INFO",
            f"{KIND_NAMES.get(job.kind, job.kind)}结果：{json.dumps(data, ensure_ascii=False, default=str)}",
            job.id,
        )
        if job.kind == "evaluate":
            self.verification.eval_result.setText(json_text(data))
        elif job.kind == "export":
            self.verification.export_result.setText(json_text(data))
        elif job.kind == "benchmark":
            self.verification.benchmark_result.setText(json_text(data))
        if job.kind == "train":
            candidates = list(job.run_dir.rglob("best.pt"))
            if candidates:
                self.verification.model.setText(candidates[-1])
        frame_path = data.get("output_path") or data.get("last_frame_path")
        if frame_path and Path(frame_path).is_file():
            image = QImage(str(frame_path))
            if not image.isNull():
                self._last_frame = image
                self._last_frame_data = data
                self.verification.show_frame(image, data)
            if job.id in self._capture_jobs:
                self.import_capture(frame_path, job)

    def refresh_capture_sources(self):
        from ..capture import enumerate_windows

        self.capture_source.clear()
        for index, screen in enumerate(QApplication.screens(), 1):
            self.capture_source.addItem(
                f"桌面 {index} · {screen.name()}", {"source": "desktop", "monitor_index": index}
            )
        for window in enumerate_windows():
            self.capture_source.addItem(
                window["title"], {"source": "window", "hwnd": window["hwnd"], "client_only": True}
            )

    def pick_window(self):
        self.status.setText("3 秒后选择鼠标指向的窗口…")

        def pick():
            from ctypes import wintypes

            point = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            user32 = ctypes.windll.user32
            user32.WindowFromPoint.argtypes = [wintypes.POINT]
            user32.WindowFromPoint.restype = wintypes.HWND
            user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
            user32.GetAncestor.restype = wintypes.HWND
            hwnd = int(user32.GetAncestor(user32.WindowFromPoint(point), 2) or 0)
            self.refresh_capture_sources()
            for index in range(self.capture_source.count()):
                if self.capture_source.itemData(index).get("hwnd") == hwnd:
                    self.capture_source.setCurrentIndex(index)
                    self.status.setText("已选中窗口；可先预览客户区")
                    return
            self.status.setText("未找到可采集窗口，请刷新后选择")

        QTimer.singleShot(3000, pick)

    def capture(self, continuous=False, preview=False):
        if not preview and not self.require_project():
            return
        source = self.capture_source.currentData() or {"source": "desktop", "monitor_index": 1}
        parameters = {
            **source,
            "interval_seconds": self.capture_interval.value(),
            "save_frames": not preview,
            "max_fps": 5,
            "client_only": True,
        }
        job = self.start_job("capture_stream" if continuous else "capture", parameters, "inference")
        if job:
            if not preview:
                self._capture_jobs.add(job.id)
            if continuous:
                self.capture_toggle.setText("停止定时截图")
            if preview:
                self.tabs.setCurrentIndex(1)

    def toggle_capture(self):
        active = [job for job in self.manager.active_jobs() if job.kind == "capture_stream"]
        if active:
            for job in active:
                self.manager.stop(job.id)
        else:
            self.capture(True)

    def import_capture(self, path, job):
        if not path or not Path(path).is_file():
            return
        if not self.service:
            if self.task_thread:
                self._capture_import_queue.append((path, job))
            return
        try:
            asset_id, created = self.service.import_image(
                Path(path),
                session=job.id,
                source_reference=f"capture:{job.id}/{Path(path).name}",
                metadata={"source": job.parameters, "captured_at": datetime.now().isoformat()},
            )
            if created:
                self.refresh_images()
        except Exception as exc:
            self.log("ERROR", f"截图回流失败：{exc}", job.id)

    def stop_inference(self):
        for job in self.manager.active_jobs():
            if job.kind in ("infer", "infer_stream", "benchmark"):
                self.manager.stop(job.id)

    def save_feedback(self):
        if not self.require_project():
            return
        image = self.verification.view.original
        if image.isNull():
            self.fail("当前没有可保存的识别画面")
            return
        note, ok = QInputDialog.getText(self, "保存问题样本", "问题说明（可留空）")
        if not ok:
            return
        feedback = self.service.root / "feedback"
        feedback.mkdir(exist_ok=True)
        path = feedback / f"{datetime.now():%Y%m%d-%H%M%S-%f}.png"
        if not image.save(str(path)):
            self.fail("无法保存问题图片")
            return
        asset_id, created = self.service.import_image(path)
        atomic_write(
            path.with_suffix(".json"),
            json_text(
                {
                    "asset_id": asset_id,
                    "note": note,
                    "model": self._last_frame_data.get("model_path"),
                    "confidence": self._last_frame_data.get("confidence"),
                    "frame": self._last_frame_data,
                }
            ),
        )
        self.refresh_images()
        self.log(
            "INFO", f"问题样本{'已进入待标注列表' if created else '与已有图片重复，保留已有标注'}：{asset_id}"
        )

    def settings_saved(self, initial=False):
        try:
            registered = self.hotkey.register(self.settings.values.get("capture_hotkey", "Ctrl+E"))
            if not registered:
                self.log("WARNING", "全局截图快捷键未注册，可能已被占用；仍可使用截图按钮。")
            if not initial:
                self.log("INFO", "设置已保存；新任务使用更新的环境路径")
                self.manager.runtime_paths = runtime_paths(self.settings.values)
        except Exception as exc:
            self.log("ERROR", str(exc))

    def diagnose(self):
        paths = runtime_paths(self.settings.values)
        root = application_root()

        def work(progress, cancel):
            report = {"application": str(root), "python_gui": sys.version, "runtimes": {}}
            for role, python in paths.items():
                progress({"message": f"检查 {role}"})
                code = (
                    "import json,sys,importlib.metadata as m; "
                    "d={'python':sys.version,'executable':sys.executable}; "
                    "d['packages']={x:m.version(x) for x in "
                    + repr(
                        ["ultralytics", "torch", "ncnn"]
                        if role == "train"
                        else ["cq-ai-engine", "ncnn", "onnxruntime", "windows-capture"]
                    )
                    + "}; "
                    + (
                        "import torch; d['cuda_available']=torch.cuda.is_available(); d['cuda_count']=torch.cuda.device_count(); "
                        if role == "train"
                        else ""
                    )
                    + "print(json.dumps(d))"
                )
                try:
                    result = run_process(
                        [str(python), "-c", code],
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=90,
                        cancel=cancel,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    report["runtimes"][role] = {
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except (OSError, subprocess.TimeoutExpired) as exc:
                    report["runtimes"][role] = {"error": str(exc)}
            return report

        self.background(work, self.preferences.display_report, title="环境诊断")

    def prepare_models(self):
        root = application_root()
        python = runtime_paths(self.settings.values)["train"]
        script = root / "scripts" / "prepare_models.py"
        destination = model_cache(self.settings.values)
        model_name = f"{self.training.family.currentText()}{self.training.scale.currentText()}.pt"

        def work(progress, cancel):
            process = run_process(
                [str(python), str(script), "--destination", str(destination), "--models", model_name],
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
                cancel=cancel,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if process.returncode:
                raise RuntimeError(process.stderr or process.stdout)
            return {"models": str(destination), "output": process.stdout}

        self.background(work, self.preferences.display_report, title="准备官方模型")

    def resources(self):
        import psutil

        self.resource_status.setText(
            f"CPU {psutil.cpu_percent():.0f}% · 内存 {psutil.virtual_memory().percent:.0f}%"
        )

    def closeEvent(self, event):
        if self.task_thread and self.task_thread.isRunning():
            self._closing = True
            self.task_thread.cancel.set()
            self.status.setText("正在结束数据操作并保存，请稍候…")
            event.ignore()
            return
        if not self.save_annotation():
            event.ignore()
            return
        self.poll_timer.stop()
        self.hotkey.close()
        self.manager.shutdown()
        if self._frame_reader and hasattr(self._frame_reader, "close"):
            self._frame_reader.close()
        self.image_model.pool.waitForDone(3000)
        if self.service:
            self.service.close()
            self.service = None
        self.settings.save()
        event.accept()
