"""Render the real Qt pages for repeatable resolution/DPI review."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", default="1366x768")
    parser.add_argument("--scale", default="1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=args.scale)

    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from yolo_workbench.desktop.widgets import STYLE
    from yolo_workbench.desktop.window import MainWindow
    from yolo_workbench.runtime import Settings, application_root

    app = QApplication([])
    QFontDatabase.addApplicationFont(str(application_root() / "fonts/NotoSansSC.ttf"))
    app.setFont(QFont("Noto Sans SC", 9))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow(Settings(args.output / "settings.json"), restore=False)
    window.resize(*map(int, args.size.split("x")))
    window.open_project(args.project)
    window.show()
    started = time.monotonic()
    while window.task_thread:
        QTest.qWait(25)
        if time.monotonic() - started > 60:
            raise TimeoutError("项目打开超时")
    if window.image_model.rowCount() > 1:
        window.images.setCurrentIndex(window.image_model.index(1))
    for index, name in enumerate(("annotate", "verify", "logs", "settings")):
        window.tabs.setCurrentIndex(index)
        QTest.qWait(100)
        if not window.grab().save(str(args.output / f"{name}.png")):
            raise RuntimeError("无法写入界面截图")
    window.close()
    app.processEvents()


if __name__ == "__main__":
    main()
