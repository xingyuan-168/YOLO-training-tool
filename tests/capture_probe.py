"""Opt-in real WGC check; captures only the temporary window created here."""

import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

from yolo_workbench.capture import CaptureSource, enumerate_windows, window_status
from yolo_workbench.worker_inference import save_image


def probe(output_path):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        ctypes.c_void_p,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    hwnd = user32.CreateWindowExW(
        0,
        "STATIC",
        "YOLO Workbench owned capture validation",
        0x10CF0000,
        100,
        100,
        360,
        240,
        None,
        None,
        None,
        None,
    )
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    capture = None
    try:
        user32.ShowWindow(hwnd, 5)
        user32.UpdateWindow(hwnd)
        assert any(window["hwnd"] == hwnd for window in enumerate_windows())
        status = window_status(hwnd)
        assert status["status"] == "ready"
        capture = CaptureSource(source="window", hwnd=hwnd, client_only=True).start()
        deadline = time.monotonic() + 10
        frame = None
        while time.monotonic() < deadline:
            message = wintypes.MSG()
            while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
            item = capture.read(0.1)
            if item:
                frame, metadata = item
                break
        assert frame is not None, capture.status()
        rect = metadata["client_rect"]
        assert frame.shape == (rect[3] - rect[1], rect[2] - rect[0], 3)
        assert frame.max() > 0
        save_image(Path(output_path), frame)
        user32.ShowWindow(hwnd, 6)
        assert window_status(hwnd)["status"] == "minimized"
        assert capture.read(0.01) is None
        capture.close()
        capture = None
        user32.DestroyWindow(hwnd)
        assert window_status(hwnd)["status"] == "closed"
        hwnd = None
        return {
            "backend": "WGC",
            "client_size": list(frame.shape[:2]),
            "dpi": metadata["dpi"],
            "status_checks": ["ready", "minimized", "closed"],
            "output_path": str(output_path),
        }
    finally:
        if capture:
            capture.close()
        if hwnd:
            user32.DestroyWindow(hwnd)


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1]), ensure_ascii=False))
