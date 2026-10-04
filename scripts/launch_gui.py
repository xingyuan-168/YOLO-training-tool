"""PyInstaller entry point, also usable with the prepared GUI Python."""

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    try:
        from yolo_workbench.desktop.app import main

        raise SystemExit(main())
    except Exception:
        import os
        import sys
        import tempfile
        import traceback
        from pathlib import Path

        # Qt cannot show our normal error screen if a native DLL fails to load.
        # Persist the earliest traceback before PyInstaller displays its dialog.
        preferred = os.environ.get("YOLO_WORKBENCH_HOME")
        destination = Path(preferred) if preferred else Path(sys.executable).parent / "userdata"
        try:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "startup-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            (Path(tempfile.gettempdir()) / "YOLOWorkbench-startup-error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
