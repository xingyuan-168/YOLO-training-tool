"""The global shortcut is registered only for this application instance."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal


class HotkeySignals(QObject):
    activated = Signal()


class CaptureHotkey(QAbstractNativeEventFilter):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.signals = HotkeySignals()
        self.identifier = 0x594F
        self.registered = False
        app.installNativeEventFilter(self)

    def register(self, sequence):
        self.close()
        if sys.platform != "win32" or not sequence:
            return False
        parts = sequence.upper().split("+")
        modifiers = 0x4000
        for part in parts[:-1]:
            if part not in {"CTRL", "ALT", "SHIFT", "META", "WIN"}:
                raise ValueError("全局快捷键仅支持 Ctrl / Alt / Shift / Win + 单个字母、数字或 F1–F24")
            modifiers |= {"CTRL": 2, "ALT": 1, "SHIFT": 4, "META": 8, "WIN": 8}[part]
        last = parts[-1]
        if len(last) == 1 and last.isascii() and last.isalnum():
            key = ord(last)
        elif last.startswith("F") and last[1:].isdigit() and 1 <= int(last[1:]) <= 24:
            key = 0x70 + int(last[1:]) - 1
        else:
            raise ValueError("全局快捷键需要字母、数字或 F1–F24")
        self.registered = bool(ctypes.windll.user32.RegisterHotKey(None, self.identifier, modifiers, key))
        return self.registered

    def close(self):
        if self.registered:
            ctypes.windll.user32.UnregisterHotKey(None, self.identifier)
            self.registered = False

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32" and event_type in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0312 and msg.wParam == self.identifier:
                self.signals.activated.emit()
                return True, 0
        return False, 0
