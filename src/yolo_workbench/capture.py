"""Windows HWND/WGC capture and DXGI desktop sources, loaded only by workers."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from contextlib import contextmanager
from ctypes import wintypes
from datetime import datetime, timezone


def _user32():
    if os.name != "nt":
        raise RuntimeError("窗口/桌面采集仅支持 Windows")
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.IsWindow.argtypes = [wintypes.HWND]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    api.IsIconic.argtypes = [wintypes.HWND]
    api.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    api.GetDpiForWindow.argtypes = [wintypes.HWND]
    api.GetDpiForWindow.restype = wintypes.UINT
    api.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    api.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    return api


@contextmanager
def _physical_coordinates(api):
    previous = api.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
    try:
        yield
    finally:
        if previous:
            api.SetThreadDpiAwarenessContext(previous)


def window_status(hwnd: int) -> dict:
    api = _user32()
    hwnd = int(hwnd)
    if hwnd <= 0 or not api.IsWindow(hwnd):
        return {"hwnd": hwnd, "status": "closed"}
    title = ctypes.create_unicode_buffer(api.GetWindowTextLengthW(hwnd) + 1)
    api.GetWindowTextW(hwnd, title, len(title))
    pid = wintypes.DWORD()
    api.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    with _physical_coordinates(api):
        client, window = wintypes.RECT(), wintypes.RECT()
        origin = wintypes.POINT()
        if not (
            api.GetClientRect(hwnd, ctypes.byref(client))
            and api.GetWindowRect(hwnd, ctypes.byref(window))
            and api.ClientToScreen(hwnd, ctypes.byref(origin))
        ):
            return {"hwnd": hwnd, "status": "unavailable", "title": title.value, "pid": pid.value}
        extended = wintypes.RECT()
        dwm = ctypes.WinDLL("dwmapi")
        dwm.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        code = dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(extended), ctypes.sizeof(extended))
        cloaked = wintypes.DWORD()
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    status = "minimized" if api.IsIconic(hwnd) else "ready" if api.IsWindowVisible(hwnd) else "hidden"
    if status == "ready" and (client.right <= 0 or client.bottom <= 0):
        status = "empty"
    if status == "ready" and cloaked.value:
        status = "unavailable"
    return {
        "hwnd": hwnd,
        "title": title.value,
        "pid": pid.value,
        "status": status,
        "dpi": int(api.GetDpiForWindow(hwnd) or 96),
        "client_rect": [origin.x, origin.y, origin.x + client.right, origin.y + client.bottom],
        "window_rect": [window.left, window.top, window.right, window.bottom],
        "capture_rect": (
            [extended.left, extended.top, extended.right, extended.bottom]
            if code == 0
            else [window.left, window.top, window.right, window.bottom]
        ),
    }


def enumerate_windows() -> list[dict]:
    api = _user32()
    windows = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def visit(hwnd, _):
        try:
            if api.IsWindowVisible(hwnd) and api.GetWindowTextLengthW(hwnd) > 0:
                entry = window_status(int(hwnd))
                if entry["status"] in {"ready", "minimized"}:
                    windows.append(entry)
        except (OSError, ValueError):
            pass
        return True

    api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    if not api.EnumWindows(visit, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return sorted(windows, key=lambda value: (value["title"].casefold(), value["hwnd"]))


def client_crop(image, metadata):
    """Map physical client bounds onto the captured extended-window surface."""
    height, width = image.shape[:2]
    client = metadata["client_rect"]
    cw, ch = client[2] - client[0], client[3] - client[1]
    if (width, height) == (cw, ch):
        return image.copy()
    candidates = [metadata["capture_rect"], metadata["window_rect"]]
    rect = min(candidates, key=lambda r: abs((r[2] - r[0]) - width) + abs((r[3] - r[1]) - height))
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    if rw <= 0 or rh <= 0 or abs(rw - width) > 2 or abs(rh - height) > 2:
        raise ValueError("采集帧与窗口尺寸不同步，等待下一帧")
    left, top = round((client[0] - rect[0]) * width / rw), round((client[1] - rect[1]) * height / rh)
    right, bottom = round((client[2] - rect[0]) * width / rw), round((client[3] - rect[1]) * height / rh)
    if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
        raise ValueError("客户区不在有效采集帧内")
    return image[top:bottom, left:right].copy()


class LatestFrameQueue:
    """One-item queue: a slow inference consumer never accumulates stale frames."""

    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self.dropped = 0
        self.closed = False

    def put(self, item):
        with self._condition:
            if self.closed:
                return
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._condition.notify()

    def get(self, timeout=0.25):
        with self._condition:
            self._condition.wait_for(lambda: self._item is not None or self.closed, timeout=timeout)
            item, self._item = self._item, None
            return item

    def clear(self):
        with self._condition:
            self._item = None

    def close(self):
        with self._condition:
            self.closed = True
            self._item = None
            self._condition.notify_all()


class CaptureSource:
    def __init__(self, *, source="window", hwnd=None, monitor_index=1, client_only=True, max_fps=10):
        if source not in {"window", "desktop"}:
            raise ValueError("采集来源必须为窗口或桌面")
        if not 0 < max_fps <= 120:
            raise ValueError("采集帧率必须位于 0–120")
        self.source, self.hwnd, self.monitor_index = source, int(hwnd or 0), int(monitor_index)
        self.client_only, self.max_fps = bool(client_only), max_fps
        self.queue = LatestFrameQueue()
        self._control = self._capture = self._desktop = None
        self._closed = False
        self._failure = None
        self._last_arrival = time.monotonic()
        self._last_sequence = None
        self.session_id = datetime.now(timezone.utc).strftime("capture-%Y%m%dT%H%M%S.%fZ")
        self._pid = None

    def start(self):
        from windows_capture import DxgiDuplicationSession, WindowsCapture

        if self.source == "desktop":
            self._desktop = DxgiDuplicationSession(monitor_index=self.monitor_index)
            return self
        status = window_status(self.hwnd)
        if status["status"] != "ready":
            raise ValueError(f"目标窗口不可采集：{status['status']}")
        self._pid = status["pid"]
        self._capture = WindowsCapture(
            window_hwnd=self.hwnd,
            cursor_capture=False,
            minimum_update_interval=max(1, round(1000 / self.max_fps)),
        )

        @self._capture.event
        def on_frame_arrived(frame, control):
            if self._closed:
                control.stop()
                return
            try:
                current = window_status(self.hwnd)
                if current["status"] != "ready" or current.get("pid") != self._pid:
                    self.queue.clear()
                    return
                if self._last_sequence == frame.timespan:
                    return
                if not frame.frame_buffer[:, :, 3].any():
                    # A fully transparent surface is not a usable capture.
                    self.queue.clear()
                    return
                pixels = frame.frame_buffer[:, :, :3]
                pixels = client_crop(pixels, current) if self.client_only else pixels.copy()
                if pixels.size == 0:
                    return
                self._last_sequence = frame.timespan
                self._last_arrival = time.monotonic()
                self.queue.put((pixels, self._metadata(current, frame.timespan)))
            except ValueError:
                # Resize and capture delivery race; discard this frame.
                self.queue.clear()
            except Exception as error:
                self._failure = str(error)
                self.queue.clear()

        @self._capture.event
        def on_closed():
            self._closed = True
            self.queue.close()

        self._control = self._capture.start_free_threaded()
        return self

    def _metadata(self, current, sequence):
        return {
            **current,
            "kind": self.source,
            "capture_backend": "WGC" if self.source == "window" else "DXGI",
            "client_only": self.client_only if self.source == "window" else False,
            "session": self.session_id,
            "frame_sequence": int(sequence),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "received_monotonic": time.monotonic(),
        }

    def status(self):
        if self.source == "window":
            current = window_status(self.hwnd)
            if current.get("pid") != self._pid and current["status"] == "ready":
                current["status"] = "closed"
            if current["status"] != "ready":
                self.queue.clear()
                return current
        else:
            current = {"kind": "desktop", "monitor_index": self.monitor_index, "status": "ready"}
        if self._failure:
            return {**current, "status": "unavailable", "error": self._failure}
        if self._closed:
            return {**current, "status": "closed"}
        if time.monotonic() - self._last_arrival > 2:
            return {**current, "status": "no_new_frame"}
        return current

    def read(self, timeout=0.25):
        if self._closed:
            return None
        if self.source == "window":
            status = self.status()
            if status["status"] not in {"ready", "no_new_frame"}:
                return None
            item = self.queue.get(timeout)
            if item is not None and self.status()["status"] in {"ready", "no_new_frame"}:
                return item
            return None
        frame = self._desktop.acquire_frame(timeout_ms=min(1000, max(1, round(timeout * 1000))))
        if frame is None:
            return None
        pixels = frame.to_bgr().copy()
        self._last_arrival = time.monotonic()
        return pixels, self._metadata(
            {"monitor_index": self.monitor_index, "status": "ready"}, time.monotonic_ns()
        )

    def close(self):
        self._closed = True
        self.queue.close()
        if self._control is not None:
            self._control.stop()
            self._control = None
        self._capture = self._desktop = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
