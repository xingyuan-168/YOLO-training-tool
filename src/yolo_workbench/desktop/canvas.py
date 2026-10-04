from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..labels import Box

PALETTE = ("#4c9aff", "#34d399", "#fbbf24", "#fb7185", "#c084fc", "#2dd4bf", "#fb923c", "#e879f9")


class BoxCommand(QUndoCommand):
    def __init__(self, canvas, before, after, text, *, confirmed_empty=False):
        super().__init__(text)
        self.canvas, self.before, self.after = canvas, list(before), list(after)
        self.before_empty, self.after_empty = canvas.confirmed_empty, confirmed_empty

    def undo(self):
        self.canvas._apply(self.before, self.before_empty)

    def redo(self):
        self.canvas._apply(self.after, self.after_empty)


class AnnotationItem(QGraphicsRectItem):
    def __init__(self, rect, class_id, canvas, index):
        super().__init__(rect)
        self.class_id, self.canvas, self.index = class_id, canvas, index
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(1)

    def paint(self, painter, option, widget=None):
        color = QColor(PALETTE[self.class_id % len(PALETTE)])
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(25 if not self.isSelected() else 50)
        painter.setBrush(fill)
        painter.drawRect(self.rect())
        scale = max(abs(self.canvas.transform().m11()), 0.001)
        label = (
            self.canvas.classes[self.class_id]
            if self.class_id < len(self.canvas.classes)
            else str(self.class_id)
        )
        label = f"{self.class_id + 1} · {label}"
        if self.canvas.labels_visible:
            painter.save()
            painter.translate(self.rect().topLeft())
            painter.scale(1 / scale, 1 / scale)
            font = QFont("Noto Sans SC", 9)
            painter.setFont(font)
            bounds = painter.fontMetrics().boundingRect(label).adjusted(-5, -3, 5, 3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(QRectF(0, -bounds.height(), bounds.width(), bounds.height()))
            painter.setPen(QColor("#08111f"))
            painter.drawText(5, -5, label)
            painter.restore()
        if self.isSelected():
            painter.setBrush(QColor("white"))
            size = 7 / scale
            for point in self.canvas.handles(self.rect()):
                painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size))


class AnnotationCanvas(QGraphicsView):
    boxes_changed = Signal(list)
    selection_changed = Signal(int)
    navigate = Signal(int)
    next_pending = Signal()
    save_requested = Signal()
    information = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#18212f"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.image_size = (0, 0)
        self.classes = []
        self.boxes = []
        self.confirmed_empty = False
        self.items_by_index = []
        self.current_class = 0
        self.mode = "draw"
        self.labels_visible = True
        self.editable = True
        self.undo_stack = QUndoStack(self)
        self._gesture = None
        self._temporary = None
        self._pan = None
        self._space = False
        self._fit = True
        self.scene().selectionChanged.connect(self._selection)

    def _selection(self):
        selected = self.selected_index()
        self.selection_changed.emit(selected)

    def selected_index(self):
        return next((item.index for item in self.items_by_index if item.isSelected()), -1)

    def set_document(self, path, boxes, classes, *, confirmed_empty=False):
        image = QImage(str(path)) if path else QImage()
        if path and image.isNull():
            raise ValueError("无法解码图片")
        self.set_image(image, boxes, classes)
        self.confirmed_empty = confirmed_empty

    def set_image(self, image: QImage, boxes=None, classes=None, *, fit=True):
        self._gesture = None
        self._temporary = None
        self.items_by_index = []
        self.scene().clear()
        self.undo_stack.clear()
        self.image_size = (image.width(), image.height())
        self.classes = list(classes or [])
        self.current_class = min(self.current_class, max(0, len(self.classes) - 1))
        self.scene().addItem(QGraphicsPixmapItem(QPixmap.fromImage(image)))
        self.scene().setSceneRect(QRectF(0, 0, image.width(), image.height()))
        self.boxes = list(boxes or [])
        self.confirmed_empty = False
        self._render()
        if fit:
            self.fit_image()

    def _render(self, selected=-1):
        for item in self.items_by_index:
            self.scene().removeItem(item)
        self.items_by_index = []
        width, height = self.image_size
        if not width or not height:
            return
        for index, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.xyxy(width, height)
            item = AnnotationItem(QRectF(x1, y1, x2 - x1, y2 - y1), box.class_id, self, index)
            self.scene().addItem(item)
            self.items_by_index.append(item)
            if index == selected:
                item.setSelected(True)

    def _apply(self, boxes, confirmed_empty=False):
        selected = self.selected_index()
        self.boxes = list(boxes)
        self.confirmed_empty = confirmed_empty
        self._render(min(selected, len(boxes) - 1))
        self.boxes_changed.emit(list(self.boxes))

    def change(self, boxes, description):
        if self.editable and list(boxes) != self.boxes:
            self.undo_stack.push(BoxCommand(self, self.boxes, boxes, description))

    def confirm_empty(self):
        if self.editable and (self.boxes or not self.confirmed_empty):
            self.undo_stack.push(BoxCommand(self, self.boxes, [], "确认无目标", confirmed_empty=True))

    def delete_selected(self):
        index = self.selected_index()
        if index >= 0:
            self.change(self.boxes[:index] + self.boxes[index + 1 :], "删除标注")

    def copy_selected(self):
        index = self.selected_index()
        if index >= 0:
            box = self.boxes[index]
            dx = min(12 / self.image_size[0], 1 - box.cx - box.width / 2)
            dy = min(12 / self.image_size[1], 1 - box.cy - box.height / 2)
            clone = Box(box.class_id, box.cx + dx, box.cy + dy, box.width, box.height)
            self.change([*self.boxes, clone], "复制标注")
            self.items_by_index[-1].setSelected(True)

    def choose_class(self, class_id, apply_selection=True):
        if not 0 <= class_id < len(self.classes):
            return
        self.current_class = class_id
        index = self.selected_index()
        if apply_selection and index >= 0:
            b = self.boxes[index]
            boxes = list(self.boxes)
            boxes[index] = Box(class_id, b.cx, b.cy, b.width, b.height)
            self.change(boxes, "修改类别")

    def fit_image(self):
        if all(self.image_size):
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_image()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.015 < self.transform().m11() * factor < 40:
            self.scale(factor, factor)
            self._fit = False
        event.accept()

    def bounded(self, point):
        width, height = self.image_size
        return QPointF(max(0, min(width, point.x())), max(0, min(height, point.y())))

    @staticmethod
    def handles(rect):
        return [
            rect.topLeft(),
            QPointF(rect.center().x(), rect.top()),
            rect.topRight(),
            QPointF(rect.right(), rect.center().y()),
            rect.bottomRight(),
            QPointF(rect.center().x(), rect.bottom()),
            rect.bottomLeft(),
            QPointF(rect.left(), rect.center().y()),
        ]

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.MiddleButton or self._space:
            self._pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if not self.editable or event.button() != Qt.MouseButton.LeftButton or not all(self.image_size):
            super().mousePressEvent(event)
            return
        point = self.bounded(self.mapToScene(event.position().toPoint()))
        selected = self.selected_index()
        if selected >= 0:
            rect = self.items_by_index[selected].rect()
            radius = 10 / max(self.transform().m11(), 0.001)
            for handle, position in enumerate(self.handles(rect)):
                if (position - point).manhattanLength() < radius:
                    self._gesture = ("resize", point, selected, QRectF(rect), handle)
                    return
        if self.mode == "draw":
            if not self.classes:
                self.information.emit("请先添加类别")
                return
            self.scene().clearSelection()
            self._gesture = ("draw", point, -1, QRectF(point, point), -1)
            self._temporary = self.scene().addRect(QRectF(point, point), QPen(QColor("#4c9aff"), 2))
        else:
            item = self.itemAt(event.position().toPoint())
            self.scene().clearSelection()
            if isinstance(item, AnnotationItem):
                item.setSelected(True)
                self._gesture = ("move", point, item.index, QRectF(item.rect()), -1)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self._fit = False
            return
        if self._gesture is None:
            super().mouseMoveEvent(event)
            return
        kind, start, index, rect, handle = self._gesture
        point = self.bounded(self.mapToScene(event.position().toPoint()))
        if kind == "draw":
            self._temporary.setRect(QRectF(start, point).normalized())
        elif kind == "move":
            delta = point - start
            dx = max(-rect.left(), min(self.image_size[0] - rect.right(), delta.x()))
            dy = max(-rect.top(), min(self.image_size[1] - rect.bottom(), delta.y()))
            self.items_by_index[index].setRect(rect.translated(dx, dy))
        else:
            updated = QRectF(rect)
            if handle in (0, 6, 7):
                updated.setLeft(min(point.x(), rect.right() - 1))
            if handle in (2, 3, 4):
                updated.setRight(max(point.x(), rect.left() + 1))
            if handle in (0, 1, 2):
                updated.setTop(min(point.y(), rect.bottom() - 1))
            if handle in (4, 5, 6):
                updated.setBottom(max(point.y(), rect.top() + 1))
            self.items_by_index[index].setRect(updated)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
        elif self._gesture:
            kind, _start, index, _rect, _handle = self._gesture
            self._gesture = None
            if kind == "draw":
                rect = self._temporary.rect()
                self.scene().removeItem(self._temporary)
                self._temporary = None
                if rect.width() >= 2 and rect.height() >= 2:
                    box = Box.from_xyxy(
                        self.current_class,
                        (rect.left(), rect.top(), rect.right(), rect.bottom()),
                        *self.image_size,
                    )
                    self.change([*self.boxes, box], "绘制标注")
                    self.items_by_index[-1].setSelected(True)
            else:
                rect = self.items_by_index[index].rect()
                boxes = list(self.boxes)
                boxes[index] = Box.from_xyxy(
                    boxes[index].class_id,
                    (rect.left(), rect.top(), rect.right(), rect.bottom()),
                    *self.image_size,
                )
                self.change(boxes, "移动标注" if kind == "move" else "调整标注")
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl and key == Qt.Key.Key_Z:
            self.undo_stack.undo()
        elif ctrl and key == Qt.Key.Key_Y:
            self.undo_stack.redo()
        elif ctrl and key == Qt.Key.Key_C:
            self.copy_selected()
        elif ctrl and key == Qt.Key.Key_S:
            self.save_requested.emit()
        elif key == Qt.Key.Key_Delete:
            self.delete_selected()
        elif key == Qt.Key.Key_W:
            self.mode = "draw"
            self.information.emit("绘框模式 · W")
        elif key == Qt.Key.Key_V:
            self.mode = "select"
            self.information.emit("选择模式 · V")
        elif key == Qt.Key.Key_F:
            self.fit_image()
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.navigate.emit(1)
        elif key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.navigate.emit(-1)
        elif key == Qt.Key.Key_N:
            self.next_pending.emit()
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self.choose_class(key - Qt.Key.Key_1)
        elif key == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key == Qt.Key.Key_Escape:
            self.scene().clearSelection()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space = False
            self._pan = None
            self.unsetCursor()
        super().keyReleaseEvent(event)
