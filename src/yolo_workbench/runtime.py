"""Portable paths; importing this module never loads a training/inference library."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .storage import atomic_write, json_text

_LAUNCH_LOCK = threading.Lock()


def launch_process(arguments, **kwargs):
    """Keep the frozen GUI's DLL search path out of isolated worker runtimes."""
    environment = dict(kwargs.pop("env", os.environ))
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    if getattr(sys, "frozen", False):
        bundle = str(Path(sys._MEIPASS).resolve()).casefold()
        environment["PATH"] = os.pathsep.join(
            p for p in environment.get("PATH", "").split(os.pathsep) if not p.casefold().startswith(bundle)
        )
    with _LAUNCH_LOCK:
        if os.name == "nt" and getattr(sys, "frozen", False):
            import ctypes

            ctypes.windll.kernel32.SetDllDirectoryW(None)
            try:
                return subprocess.Popen(arguments, env=environment, **kwargs)
            finally:
                ctypes.windll.kernel32.SetDllDirectoryW(str(sys._MEIPASS))
        return subprocess.Popen(arguments, env=environment, **kwargs)


def run_process(arguments, *, timeout=None, cancel=None, **kwargs):
    from .jobs import _ProcessTree
    from .storage import OperationCancelled

    process = launch_process(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    tree = _ProcessTree()
    started = time.monotonic()
    try:
        tree.attach(process)
        while True:
            if cancel and cancel():
                tree.terminate()
                process.communicate()
                raise OperationCancelled("操作已取消")
            if timeout and time.monotonic() - started >= timeout:
                tree.terminate()
                process.communicate()
                raise subprocess.TimeoutExpired(arguments, timeout)
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        tree.close()


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    override = os.environ.get("YOLO_WORKBENCH_HOME")
    if override:
        root = Path(override)
    elif getattr(sys, "frozen", False):
        root = application_root() / "userdata"
    else:
        root = application_root() / ".runtime-cache" / "desktop"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def runtime_paths(settings: dict | None = None) -> dict[str, Path]:
    settings = settings or {}
    root = application_root()
    result = {}
    for role in ("train", "inference"):
        configured = settings.get(f"{role}_python")
        portable = root / "runtime" / role / "python.exe"
        result[role] = (
            Path(configured)
            if configured
            else (portable if portable.exists() else root / ".runtimes" / role / "Scripts" / "python.exe")
        )
    return result


def model_cache(settings: dict | None = None) -> Path:
    path = Path((settings or {}).get("model_cache") or application_root() / "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = path or data_root() / "settings.json"
        self.error = ""
        try:
            self.values = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            if not isinstance(self.values, dict):
                raise ValueError("设置文件必须为对象")
        except (OSError, ValueError) as exc:
            self.values = {}
            self.error = f"无法读取设置，将使用默认值：{exc}"

    def save(self):
        atomic_write(self.path, json_text(self.values))

    def recent(self, path: Path):
        absolute = str(path.resolve())
        self.values["recent_projects"] = [absolute] + [
            p for p in self.values.get("recent_projects", []) if p != absolute
        ][:9]
        self.values["last_project"] = absolute
        self.save()
