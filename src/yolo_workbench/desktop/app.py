from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLO 本地标注、训练和验证工作台")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument("--smoke-screenshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-size", default="1366x768", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-close-ms", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-job", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QMessageBox

    from ..runtime import application_root
    from .widgets import STYLE
    from .window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("YOLOWorkbench")
    app.setOrganizationName("YOLOWorkbench")
    font_path = application_root() / "fonts" / "NotoSansSC.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Noto Sans SC" if font_path.is_file() else "Microsoft YaHei UI", 9))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow(restore=not args.no_restore and not args.project)

    def unhandled(kind, value, tb):
        message = "".join(traceback.format_exception(kind, value, tb))
        window.log("ERROR", message)
        QMessageBox.critical(window, "操作发生错误", str(value) + "\n完整信息已保存到日志。")

    sys.excepthook = unhandled
    if args.project:
        window.open_project(args.project)
    window.show()
    if args.smoke_job:
        from ..storage import atomic_write, json_text

        window.fail = lambda message: window.log("ERROR", message)
        request = json.loads(args.smoke_job.read_text(encoding="utf-8"))
        state = {"job": None}
        timer = QTimer(window)

        def run_job():
            if window.task_thread:
                return
            if state["job"] is None:
                state["job"] = window.start_job(
                    request["kind"], request["parameters"], request.get("runtime", "train")
                )
                if state["job"] is None:
                    timer.stop()
                    atomic_write(args.smoke_report, json_text({"state": "failed", "error": "启动失败"}))
                    window.close()
            elif state["job"].state in ("succeeded", "failed", "stopped", "interrupted"):
                job = state["job"]
                atomic_write(
                    args.smoke_report,
                    json_text(
                        {
                            "state": job.state,
                            "result": job.result,
                            "error": job.error,
                            "run_dir": str(job.run_dir),
                            "gui_imported_heavy": [
                                m for m in ("torch", "ai_engine", "ncnn", "onnxruntime") if m in sys.modules
                            ],
                        }
                    ),
                )
                timer.stop()
                window.close()

        timer.timeout.connect(run_job)
        timer.start(100)
    if args.smoke_screenshot:
        width, height = map(int, args.smoke_size.split("x"))
        window.resize(width, height)

        def screenshot():
            if window.task_thread:
                QTimer.singleShot(100, screenshot)
                return
            if window.image_model.rowCount() > 1:
                window.images.setCurrentIndex(window.image_model.index(1))
            args.smoke_screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.smoke_screenshot)):
                raise OSError("无法保存界面检查截图")

        QTimer.singleShot(1000, screenshot)
    if args.smoke_close_ms:
        QTimer.singleShot(args.smoke_close_ms, window.close)
    return app.exec()
