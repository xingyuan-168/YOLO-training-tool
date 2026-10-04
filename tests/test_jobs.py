from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yolo_workbench.jobs import JobManager
from yolo_workbench.storage import atomic_write, json_text

FAKE_WORKER = r"""
import json, os, pathlib, subprocess, sys, time
from datetime import datetime, timezone
request = json.loads(pathlib.Path(sys.argv[-1]).read_text(encoding="utf-8"))
root = pathlib.Path(request["run_dir"])
while not (root / "launch.flag").exists():
    time.sleep(.01)
sequence = 0
def emit(kind, data):
    global sequence
    sequence += 1
    print(json.dumps(dict(protocol_version=1,job_id=request["job_id"],sequence=sequence,
                         timestamp=datetime.now(timezone.utc).isoformat(),type=kind,data=data)), flush=True)
emit("state", {"state": "running"})
print("native backend diagnostics", file=sys.stderr, flush=True)
mode = request["parameters"].get("mode", "success")
if mode == "invalid":
    print("native protocol pollution", flush=True)
    time.sleep(30)
elif mode == "crash":
    os._exit(27)
elif mode in {"wait", "child"}:
    if mode == "child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (root / "child.pid").write_text(str(child.pid))
    while not (root / "stop.flag").exists():
        time.sleep(.02)
    emit("state", {"state": "stopped"})
else:
    emit("result", {"answer": 42})
    emit("state", {"state": "succeeded"})
"""


@pytest.fixture
def manager(tmp_path):
    app = tmp_path / "app"
    package = app / "src/yolo_workbench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    with JobManager(tmp_path / "project", {"train": Path(sys.executable)}, app) as result:
        yield result


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for worker")


def test_durable_events_and_reopen(manager):
    job = manager.start("infer", {})
    wait_for(lambda: not manager.active_jobs())
    assert job.state == "succeeded"
    assert job.result == {"answer": 42}
    assert job.creation_identity
    events = manager.poll_events()
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert "native backend" in (job.run_dir / "stderr.log").read_text()
    assert len((job.run_dir / "events.jsonl").read_text().splitlines()) == 3
    manager.shutdown()
    with JobManager(manager.project_root, manager.runtime_paths, manager.app_root) as reopened:
        assert reopened.jobs[job.id].state == "succeeded"
        assert reopened.jobs[job.id].result["answer"] == 42


def test_stop_and_per_device_conflicts(manager):
    job = manager.start("infer_stream", {"mode": "wait", "device": "cpu"})
    with pytest.raises(RuntimeError, match="设备"):
        manager.start("evaluate", {"device": "cpu"})
    other = manager.start("capture", {"mode": "wait"})
    manager.stop(job.id)
    manager.stop(other.id)
    wait_for(lambda: not manager.active_jobs())
    assert job.state == other.state == "stopped"


@pytest.mark.parametrize("mode,code", [("crash", "worker_exit"), ("invalid", "protocol_error")])
def test_native_crash_and_protocol_noise_fail_visibly(manager, mode, code):
    job = manager.start("infer", {"mode": mode})
    wait_for(lambda: not manager.active_jobs())
    assert job.state == "failed"
    assert any(event["data"].get("code") == code for event in manager.poll_events())


def test_force_stop_only_owned_tree(manager):
    import ctypes

    if os.name != "nt":
        pytest.skip("Windows Job Object ownership regression")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        job = manager.start("infer", {"mode": "child"})
        wait_for(lambda: (job.run_dir / "child.pid").exists())
        child_pid = int((job.run_dir / "child.pid").read_text())
        child = kernel.OpenProcess(0x00100000, False, child_pid)
        assert child
        manager.stop(job.id, force=True)
        wait_for(lambda: not manager.active_jobs())
        try:
            wait_for(lambda: kernel.WaitForSingleObject(child, 0) == 0)
        finally:
            kernel.CloseHandle(child)
        assert job.state == "interrupted"
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_crash_reconciliation_never_acts_on_stale_pid(manager):
    job = manager.start("infer", {})
    wait_for(lambda: not manager.active_jobs())
    manager.shutdown()
    record = job.to_dict()
    record.update(state="running", pid=os.getpid(), creation_identity="stale identity", sequence=0, result={})
    atomic_write(job.run_dir / "job.json", json_text(record))
    atomic_write(job.run_dir / "events.jsonl", '{"truncated":')
    with JobManager(manager.project_root, manager.runtime_paths, manager.app_root) as restored:
        assert restored.jobs[job.id].state == "interrupted"
        assert os.getpid() == record["pid"]


def test_worker_native_stdout_is_redirected(tmp_path):
    script = r"""
import os, sys
from yolo_workbench.worker import protocol_stream
from yolo_workbench.protocol import EventWriter
with protocol_stream() as stream:
    os.write(1, b"native log\n")
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetStdHandle.argtypes = [ctypes.c_ulong]
        kernel.GetStdHandle.restype = ctypes.c_void_p
        kernel.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong,
                                     ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
        written = ctypes.c_ulong()
        assert kernel.WriteFile(kernel.GetStdHandle(-11), b"win32 log\n", 10, ctypes.byref(written), None)
    print("python log")
    EventWriter("owned", stream).emit("result", {"ok": True})
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["data"]["ok"] is True
    assert "native log" in result.stderr and "python log" in result.stderr
    if os.name == "nt":
        assert "win32 log" in result.stderr


def test_worker_enforces_offline_boundary(tmp_path):
    script = r"""
import os, socket, sys
from pathlib import Path
from yolo_workbench.worker import configure_offline
configure_offline(Path(sys.argv[1]))
assert os.environ["YOLO_AUTOINSTALL"] == "false"
assert Path(os.environ["YOLO_CONFIG_DIR"]).is_dir()
for action in (lambda: socket.socket().connect(("8.8.8.8", 443)),
               lambda: socket.getaddrinfo("example.com", 443)):
    try:
        action()
    except OSError:
        pass
    else:
        raise AssertionError("Outbound operation was permitted")
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows parent-crash Job Object lifetime")
def test_supervisor_crash_kills_owned_worker_and_reconciles(manager, tmp_path):
    import ctypes

    root = tmp_path / "crashed-project"
    script = r"""
import os, sys, time
from pathlib import Path
from yolo_workbench.jobs import JobManager
root, app = Path(sys.argv[1]), Path(sys.argv[2])
manager = JobManager(root, {"train": Path(sys.executable)}, app)
job = manager.start("infer", {"mode": "child"})
while not (job.run_dir / "child.pid").exists():
    time.sleep(.02)
(root / "ready.txt").write_text(job.id)
while not (root / "crash.flag").exists():
    time.sleep(.02)
os._exit(99)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(manager.app_root)],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handles = []
    try:
        wait_for(lambda: (root / "ready.txt").exists())
        job_id = (root / "ready.txt").read_text()
        run = root / "jobs" / job_id
        record = json.loads((run / "job.json").read_text(encoding="utf-8"))
        for pid in (record["pid"], int((run / "child.pid").read_text())):
            handle = kernel.OpenProcess(0x00100000, False, pid)
            assert handle
            handles.append(handle)
        (root / "crash.flag").touch()
        assert process.wait(timeout=10) == 99
        for handle in handles:
            wait_for(lambda: kernel.WaitForSingleObject(handle, 0) == 0)
        with JobManager(root, manager.runtime_paths, manager.app_root) as reopened:
            assert reopened.jobs[job_id].state == "interrupted"
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
