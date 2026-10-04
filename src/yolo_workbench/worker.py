"""Worker entry point. Stdout is reserved for protocol even for native DLL logs."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import traceback
from pathlib import Path

from .protocol import PROTOCOL_VERSION, EventWriter
from .storage import atomic_write, json_text


def protocol_stream():
    sys.stdout.flush()
    descriptor = os.dup(sys.stdout.fileno())
    os.set_inheritable(descriptor, False)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if os.name == "nt":
        import msvcrt
        from ctypes import wintypes as wt

        set_handle = ctypes.WinDLL("kernel32", use_last_error=True).SetStdHandle
        set_handle.argtypes = [wt.DWORD, wt.HANDLE]
        set_handle.restype = wt.BOOL
        if not set_handle(wt.DWORD(-11), msvcrt.get_osfhandle(sys.stderr.fileno())):
            raise ctypes.WinError(ctypes.get_last_error())
    sys.stdout = sys.stderr
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


def configure_offline(run_dir: Path):
    config_dir = run_dir / "runtime-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", YOLO_CONFIG_DIR=str(config_dir))
    # Ultralytics can check optional services and fonts while importing. Block
    # outbound network at the worker boundary while retaining local IPC.
    import ipaddress
    import socket

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def local(host):
        if host in {"localhost", "", None}:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def connect(sock, address):
        if isinstance(address, tuple) and not local(address[0]):
            raise OSError("离线任务禁止访问网络；请先显式准备所需资源")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if isinstance(address, tuple) and not local(address[0]):
            return 10013 if os.name == "nt" else 13
        return original_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if not local(host):
            raise OSError("离线任务禁止 DNS 查询")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo


def prepare_ultralytics():
    import ultralytics
    from ultralytics.utils import SETTINGS, checks

    integrations = {
        name: False
        for name in (
            "sync",
            "hub",
            "clearml",
            "comet",
            "dvc",
            "mlflow",
            "neptune",
            "raytune",
            "tensorboard",
            "wandb",
        )
        if name in SETTINGS
    }
    SETTINGS.update(integrations)

    def local_font(font="Arial.ttf"):
        font_env = "YOLO_WORKBENCH_FONT_CJK" if "unicode" in str(font).lower() else "YOLO_WORKBENCH_FONT"
        candidates = [
            Path(os.environ.get(font_env, "")),
            Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics" / Path(font).name,
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        # Pillow's bundled default font needs no download.
        return None

    checks.check_font = local_font
    import ultralytics.data.utils
    import ultralytics.utils.plotting

    ultralytics.data.utils.check_font = local_font
    ultralytics.utils.plotting.check_font = local_font
    return ultralytics


def dispatch(request: dict, emit):
    kind = request["kind"]
    if kind in {"train", "evaluate"}:
        from .training_worker import evaluate, train

        return (train if kind == "train" else evaluate)(request, emit)
    if kind == "export":
        from .export_worker import export

        return export(request, emit)
    if kind in {"infer", "infer_stream", "capture", "capture_stream", "benchmark"}:
        from .worker_inference import handle

        return handle(request, emit)
    raise ValueError(f"不支持的任务类型：{kind}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("不支持的请求协议版本")
    run_dir = Path(request["run_dir"]).resolve()
    if args.request.resolve().parent != run_dir:
        raise ValueError("任务请求必须位于该任务目录")
    if request.get("launch_guard"):
        deadline = time.monotonic() + 20
        while not (run_dir / "launch.flag").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("监督进程未取得 Worker 所有权")
            time.sleep(0.02)
    with protocol_stream() as stream:
        writer = EventWriter(request["job_id"], stream)
        terminal = False

        def emit(kind, data):
            nonlocal terminal
            if kind == "state" and data.get("state") in {"succeeded", "failed", "stopped", "interrupted"}:
                terminal = True
            return writer.emit(kind, data)

        try:
            configure_offline(run_dir)
            emit("state", {"state": "running", "kind": request["kind"]})
            if (run_dir / "stop.flag").exists():
                emit("state", {"state": "stopped", "reason": "开始前取消"})
                return 0
            result = dispatch(request, emit)
            if isinstance(result, dict):
                atomic_write(run_dir / "result.json", json_text(result))
                emit("result", result)
            if not terminal:
                state = "stopped" if (run_dir / "stop.flag").exists() else "succeeded"
                emit("state", {"state": state})
            return 0
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit(
                "error", {"code": type(exc).__name__, "message": str(exc), "log": str(run_dir / "stderr.log")}
            )
            emit("state", {"state": "failed"})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
