from __future__ import annotations

import traceback
from collections import OrderedDict
from threading import Event

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QColor, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

STATUS_NAMES = {"pending": "○ 未标注", "labeled": "● 已标注", "empty": "✓ 无目标"}


class ThumbnailSignals(QObject):
    ready = Signal(int, str, QImage)


class Thumbnail(QRunnable):
    def __init__(self, generation, asset_id, path, signals):
        super().__init__()
        self.generation, self.asset_id, self.path, self.signals = generation, asset_id, path, signals

    def run(self):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(False)
        size = reader.size()
        if size.isValid():
            size.scale(56, 42, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(size)
        self.signals.ready.emit(self.generation, self.asset_id, reader.read())


class AssetListModel(QAbstractListModel):
    """SQL pages and a bounded thumbnail cache; no full-directory image decoding."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = None
        self.search, self.status, self.deleted = "", None, False
        self.total = 0
        self.pages, self.thumbnails = OrderedDict(), OrderedDict()
        self.pending = set()
        self.generation = 0
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = ThumbnailSignals(self)
        self.signals.ready.connect(self.thumbnail_ready)

    def configure(self, service, *, search="", status=None, deleted=False):
        self.beginResetModel()
        self.service, self.search, self.status, self.deleted = service, search, status, deleted
        self.generation += 1
        self.pool.clear()
        self.pages.clear()
        self.thumbnails.clear()
        self.pending.clear()
        if service:
            query = "SELECT count(*) FROM assets WHERE deleted=? AND instr(name, ?) > 0"
            args = [int(deleted), search]
            if status:
                query += " AND status=?"
                args.append(status)
            self.total = service.db.execute(query, args).fetchone()[0]
        else:
            self.total = 0
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.total

    def record(self, row):
        if not self.service or not 0 <= row < self.total:
            return None
        page = row // 100
        if page not in self.pages:
            self.pages[page] = self.service.list_assets(
                status=self.status, search=self.search, deleted=self.deleted, limit=100, offset=page * 100
            )
            while len(self.pages) > 12:
                self.pages.popitem(last=False)
        self.pages.move_to_end(page)
        rows = self.pages[page]
        return rows[row % 100] if row % 100 < len(rows) else None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        row = self.record(index.row()) if index.isValid() else None
        if not row:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{row['name']}\n{STATUS_NAMES.get(row['status'], row['status'])} · {row['width']} × {row['height']}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"{row['name']}\n{row['id']}\n{STATUS_NAMES.get(row['status'])}"
        if role == Qt.ItemDataRole.SizeHintRole:
            return QSize(210, 62)
        if role == Qt.ItemDataRole.DecorationRole:
            asset_id = row["id"]
            if asset_id in self.thumbnails:
                self.thumbnails.move_to_end(asset_id)
                return self.thumbnails[asset_id]
            if asset_id not in self.pending:
                self.pending.add(asset_id)
                path = self.service.root / self.service._record(asset_id)["file"]
                self.pool.start(Thumbnail(self.generation, asset_id, path, self.signals))
            pixmap = QPixmap(56, 42)
            pixmap.fill(QColor("#e2e8f0"))
            return pixmap
        if role == Qt.ItemDataRole.UserRole:
            return row["id"]
        return None

    def thumbnail_ready(self, generation, asset_id, image):
        if generation != self.generation:
            return
        self.pending.discard(asset_id)
        self.thumbnails[asset_id] = QPixmap.fromImage(image)
        while len(self.thumbnails) > 200:
            self.thumbnails.popitem(last=False)
        for page, rows in self.pages.items():
            for offset, row in enumerate(rows):
                if row["id"] == asset_id:
                    index = self.index(page * 100 + offset)
                    self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])
                    return

    def refresh_status(self, asset_id):
        if not self.service:
            return
        for page, rows in self.pages.items():
            for offset, row in enumerate(rows):
                if row["id"] == asset_id:
                    row["status"] = self.service._record(asset_id)["status"]
                    index = self.index(page * 100 + offset)
                    self.dataChanged.emit(index, index)


class TaskThread(QThread):
    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancel = Event()

    def run(self):
        try:
            result = self.function(self.progress.emit, self.cancel.is_set)
            self.succeeded.emit(result)
        except Exception:
            if self.cancel.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())


class PathField(QWidget):
    changed = Signal(str)

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton("浏览…")
        self.button.setFixedWidth(66)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.edit.textChanged.connect(self.changed)
        self.edit.textChanged.connect(self.edit.setToolTip)

    def text(self):
        return self.edit.text().strip()

    def setText(self, text):
        self.edit.setText(str(text))


def text_dialog(parent, title, text, *, approve=False, button="执行", detail=""):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(710, 510)
    layout = QVBoxLayout(dialog)
    if detail:
        label = QLabel(detail)
        label.setWordWrap(True)
        layout.addWidget(label)
    editor = QPlainTextEdit(str(text))
    editor.setReadOnly(True)
    layout.addWidget(editor)
    flags = (
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        if approve
        else QDialogButtonBox.StandardButton.Close
    )
    buttons = QDialogButtonBox(flags)
    if approve:
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(button)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return dialog.exec() == QDialog.DialogCode.Accepted


STYLE = """
QWidget { font-family: 'Noto Sans SC', 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; color: #172238; }
QMainWindow, QDialog { background: #f3f5f8; }
QTabWidget::pane { border: 0; background: #f3f5f8; }
QTabBar::tab { background: #e8edf4; padding: 11px 25px; margin-right: 3px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: white; color: #1d4ed8; border-top: 3px solid #2563eb; }
QGroupBox { background: white; border: 1px solid #dce3ed; border-radius: 8px; margin-top: 18px; padding: 12px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QPushButton, QToolButton { background: white; border: 1px solid #cbd5e1; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover, QToolButton:hover { background: #eff6ff; border-color: #60a5fa; }
QPushButton:pressed { background: #dbeafe; }
QPushButton:disabled { color: #94a3b8; background: #e8edf4; }
QPushButton[primary='true'] { background: #2563eb; color: white; border-color: #2563eb; font-weight: 600; }
QPushButton[primary='true']:hover { background: #1d4ed8; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white; border: 1px solid #cbd5e1; border-radius: 4px; min-height: 26px; padding: 2px 5px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #2563eb; }
QComboBox::drop-down { width: 20px; border: 0; }
QListView, QListWidget, QTreeWidget, QTableWidget { background: white; border: 1px solid #dce3ed; border-radius: 5px; alternate-background-color: #f7f9fc; }
QListView::item:selected, QListWidget::item:selected { background: #dbeafe; color: #172238; }
QPlainTextEdit { background: white; border: 1px solid #dce3ed; border-radius: 5px; padding: 5px; }
QPlainTextEdit[log='true'] { background: #111827; color: #cbd5e1; font-family: 'Cascadia Mono', Consolas; font-size: 12px; }
QSplitter::handle { background: #e2e8f0; }
QProgressBar { border: 1px solid #dce3ed; border-radius: 4px; text-align: center; background: white; }
QProgressBar::chunk { background: #2563eb; }
QScrollArea { border: 0; }
QStatusBar { background: white; }
QLabel[muted='true'] { color: #64748b; }
"""
