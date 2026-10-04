"""Qt interaction tests: coordinates, undo, negative review, persistence and lazy index."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from yolo_workbench.dataset import DatasetService
from yolo_workbench.desktop.canvas import AnnotationCanvas
from yolo_workbench.desktop.widgets import AssetListModel
from yolo_workbench.labels import Box


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    widget = AnnotationCanvas()
    widget.resize(800, 500)
    image = QImage(640, 320, QImage.Format.Format_RGB32)
    image.fill(QColor("#abcdef"))
    widget.set_image(image, [], ["目标", "另一个类别"])
    widget.show()
    app.processEvents()
    yield widget
    widget.close()


def drag(canvas, start, end):
    origin = canvas.mapFromScene(QPointF(*start))
    target = canvas.mapFromScene(QPointF(*end))
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=origin)
    QTest.mouseMove(canvas.viewport(), target)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=target)
    QApplication.processEvents()


def test_canvas_draw_move_resize_class_and_undo(canvas):
    drag(canvas, (64, 32), (256, 160))
    assert len(canvas.boxes) == 1
    original = canvas.boxes[0]
    assert original.cx == pytest.approx(0.25, abs=0.005)
    assert original.width == pytest.approx(0.3, abs=0.005)
    canvas.mode = "select"
    drag(canvas, (160, 96), (200, 116))
    moved = canvas.boxes[0]
    assert moved.cx > original.cx
    canvas.undo_stack.undo()
    assert canvas.boxes == [original]
    canvas.undo_stack.redo()
    assert canvas.boxes == [moved]
    rect = canvas.items_by_index[0].rect()
    drag(canvas, (rect.right(), rect.bottom()), (rect.right() + 30, rect.bottom() + 20))
    assert canvas.boxes[0].width > moved.width
    canvas.choose_class(1)
    assert canvas.boxes[0].class_id == 1
    QTest.keyClick(canvas, Qt.Key.Key_Delete)
    assert not canvas.boxes
    QTest.keyClick(canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert canvas.boxes[0].class_id == 1


def test_confirmed_negative_is_explicit_and_undoable(canvas):
    canvas.confirm_empty()
    assert canvas.confirmed_empty
    canvas.undo_stack.undo()
    assert not canvas.confirmed_empty
    canvas.undo_stack.redo()
    assert canvas.confirmed_empty
    drag(canvas, (20, 20), (100, 100))
    assert not canvas.confirmed_empty
    canvas.undo_stack.undo()
    assert canvas.confirmed_empty and not canvas.boxes


def test_draw_is_bounded_and_copy_preserves_class(canvas):
    drag(canvas, (10, 10), (900, 500))
    canvas.boxes[0].validate(2)
    canvas.copy_selected()
    assert len(canvas.boxes) == 2
    for box in canvas.boxes:
        box.validate(2)


def test_new_image_clears_undo_history(canvas):
    drag(canvas, (10, 10), (100, 100))
    image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.set_image(image, [Box(1, 0.5, 0.5, 0.2, 0.2)], ["a", "b"])
    canvas.undo_stack.undo()
    assert canvas.boxes == [Box(1, 0.5, 0.5, 0.2, 0.2)]


def test_asset_list_is_paged_without_decode(tmp_path, app):
    with DatasetService.create(tmp_path / "项目 空格", "测试", ["目标"]) as service:
        with service.db:
            service.db.executemany(
                "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)",
                [
                    (f"{i:032d}", f"h{i}", f"image-{i:05d}.png", "pending", 640, 480, None, 0)
                    for i in range(10000)
                ],
            )
        model = AssetListModel()
        model.configure(service)
        assert model.rowCount() == 10000
        assert not model.pages and not model.thumbnails
        assert model.record(9999)["name"] == "image-09999.png"
        assert len(model.pages) == 1
        for row in range(0, 10000, 100):
            model.record(row)
        assert len(model.pages) == 12
        assert not model.thumbnails
        model.configure(None)


def test_window_edit_save_and_reopen(tmp_path, app, monkeypatch):
    from PIL import Image
    from PySide6.QtWidgets import QMessageBox

    from yolo_workbench.desktop.window import MainWindow
    from yolo_workbench.runtime import Settings

    monkeypatch.setenv("YOLO_WORKBENCH_HOME", str(tmp_path / "userdata"))
    failures = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: failures.append(args[-1]))
    image = tmp_path / "中文图片.png"
    Image.new("RGB", (640, 320), "blue").save(image)
    root = tmp_path / "中文 项目"
    with DatasetService.create(root, "桌面测试", ["类别一"]) as service:
        asset_id, _ = service.import_image(image)
    window = MainWindow(Settings(tmp_path / "settings.json"), restore=False)
    window.open_project(root)
    window.show()
    for _ in range(200):
        app.processEvents()
        if window.service and not window.task_thread:
            break
        QTest.qWait(10)
    assert window.current_asset == asset_id
    drag(window.canvas, (30, 30), (180, 150))
    QTest.qWait(450)
    assert not window.dirty
    saved = window.service.load_boxes(asset_id)
    assert len(saved) == 1
    window.canvas.confirm_empty()
    assert window.save_annotation()
    assert window.service.get_asset(asset_id)["status"] == "empty"
    window.canvas.undo_stack.undo()
    assert window.save_annotation()
    assert window.service.load_boxes(asset_id) == saved
    window.close()
    app.processEvents()
    with DatasetService(root) as service:
        assert service.load_boxes(asset_id) == saved
    assert not failures
    assert not any(
        name in __import__("sys").modules for name in ("torch", "ai_engine", "ncnn", "onnxruntime")
    )


def test_imported_config_and_fresh_project_do_not_leak_values(tmp_path, app):
    from yolo_workbench.desktop.training_panel import TrainingPanel
    from yolo_workbench.runtime import Settings

    panel = TrainingPanel(Settings(tmp_path / "config.json"))
    panel.apply_config(
        {"epochs": 3000, "device": "0", "batch": -1, "workers": 12, "imgsz": 320, "expert_yaml": "lr0: 0.123"}
    )
    assert panel.batch.value() == -1 and panel.workers.value() == 12
    panel.apply_config({"expert_yaml": ""})
    assert panel.expert.toPlainText() == ""
    panel.reset_project()
    assert panel.device.currentText() == "cpu"
    assert panel.batch.value() == 4 and panel.workers.value() == 0
    assert panel.epochs.value() == 100 and panel.imgsz.value() == 640


def test_runtime_capabilities_override_family_assumption(app):
    from yolo_workbench.desktop.verification import VerificationPage

    page = VerificationPage()
    page.family.setCurrentText("yolo26")
    page.backend.setCurrentIndex(page.backend.findData("ncnn"))
    page.apply_capabilities({"iou": True, "end_to_end": False}, {"input_shape": [1, 3, 320, 320]})
    assert page.iou.isEnabled() and page.input_size.value() == 320
    page.apply_capabilities({"iou": False, "end_to_end": True}, {})
    assert not page.iou.isEnabled()
