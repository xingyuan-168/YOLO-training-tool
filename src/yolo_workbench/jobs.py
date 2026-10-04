"""Durable, framework-free subprocess supervisor for the desktop workbench."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .protocol import PROTOCOL_VERSION, EventReader
from .runtime import launch_process
from .storage import ProjectLock, atomic_write, json_text

TERMINAL_STATES = {"stopped", "succeeded", "failed", "interrupted"}
KINDS = {"train", "evaluate", "export", "infer", "infer_stream", "capture", "capture_stream", "benchmark"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _ProcessTree:
    """A Windows kernel Job owns descendants, including converter processes.

    The worker cannot import native libraries or spawn children before attach().
    Closing the last handle also cleans up children when the GUI crashes. No PID
    scan or process-name kill is used, so a recycled PID is never a kill target.
    """

    def __init__(self):
        self.handle = None
        self.process = None
        if os.name != "nt":
            return
        from ctypes import wintypes as wt

        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong),
                ("job_time", ctypes.c_longlong),
                ("flags", wt.DWORD),
                ("min_ws", ctypes.c_size_t),
                ("max_ws", ctypes.c_size_t),
                ("active", wt.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wt.DWORD),
                ("scheduling", wt.DWORD),
            ]

        class Counters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", Counters),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE),
            ("SetInformationJobObject", [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL),
            ("AssignProcessToJobObject", [wt.HANDLE, wt.HANDLE], wt.BOOL),
            ("TerminateJobObject", [wt.HANDLE, wt.UINT], wt.BOOL),
            ("CloseHandle", [wt.HANDLE], wt.BOOL),
        ):
            method = getattr(self.kernel, name)
            method.argtypes, method.restype = args, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach(self, process):
        self.process = process
        if self.handle and not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if self.handle:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
        elif self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _creation_identity(process) -> str:
    if os.name == "nt":
        from ctypes import wintypes as wt

        times = [wt.FILETIME() for _ in range(4)]
        function = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
        function.argtypes = [wt.HANDLE] + [ctypes.POINTER(wt.FILETIME)] * 4
        function.restype = wt.BOOL
        if not function(int(process._handle), *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    try:
        return Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except OSError:
        return f"owned:{time.time_ns()}"


@dataclass
class Job:
    id: str
    kind: str
    state: str
    run_dir: Path
    parameters: dict = field(default_factory=dict)
    runtime: str = "train"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    pid: int | None = None
    creation_identity: str | None = None
    result: dict = field(default_factory=dict)
    error: str | None = None
    returncode: int | None = None
    sequence: int = 0
    _process: object = field(default=None, repr=False)
    _tree: object = field(default=None, repr=False)
    _thread: object = field(default=None, repr=False)
    _forced: bool = field(default=False, repr=False)
    _cleanup: bool = field(default=False, repr=False)

    def to_dict(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            **{
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(self).items()
                if not name.startswith("_")
            },
        }


class JobManager:
    def __init__(self, project_root: Path, runtime_paths: dict[str, Path], app_root: Path):
        self.project_root = Path(project_root).resolve()
        self.app_root = Path(app_root).resolve()
        self.runtime_paths = {key: Path(value).resolve() for key, value in runtime_paths.items()}
        self.root = self.project_root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._ownership = ProjectLock(self.root / ".supervisor.lock")
        self._lock = threading.RLock()
        self._events = queue.Queue(maxsize=2048)
        self.jobs: dict[str, Job] = {}
        self._closed = False
        self._reconcile()

    def _save(self, job: Job):
        job.updated_at = _now()
        atomic_write(job.run_dir / "job.json", json_text(job.to_dict()))

    def _reconcile(self):
        for record in sorted(self.root.glob("*/job.json")):
            try:
                data = json.loads(record.read_text(encoding="utf-8"))
                if data.pop("protocol_version", None) != PROTOCOL_VERSION:
                    continue
                data["run_dir"] = record.parent.resolve()
                job = Job(**data)
                if job.id != record.parent.name:
                    continue
                # Replay only complete records. A truncated final JSONL line is
                # expected after power loss and never destroys earlier events.
                events_path = record.parent / "events.jsonl"
                if events_path.exists():
                    for line in events_path.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                            if event["job_id"] != job.id:
                                break
                            job.sequence = max(job.sequence, event["sequence"])
                            self._apply(job, event)
                        except (ValueError, KeyError, TypeError):
                            break
                self.jobs[job.id] = job
                if job.state not in TERMINAL_STATES:
                    self._local(
                        job, "state", {"state": "interrupted", "reason": "监督进程异常结束，已保留最近检查点"}
                    )
                else:
                    self._save(job)
            except (OSError, ValueError, TypeError, KeyError):
                # Keep corrupt records intact for diagnosis; don't trust their
                # executable, PID, or paths to perform recovery actions.
                continue

    @staticmethod
    def _device(job: Job) -> str:
        value = str(
            job.parameters.get("config", {}).get("device", job.parameters.get("device", "cpu"))
        ).lower()
        return {"cuda": "0", "cuda:0": "0"}.get(value, value.removeprefix("cuda:"))

    def active_jobs(self) -> list[Job]:
        with self._lock:
            return [
                job
                for job in self.jobs.values()
                if job._process is not None
                and (job._process.poll() is None or job.state not in TERMINAL_STATES)
            ]

    def start(self, kind: str, parameters: dict, runtime: str = "train") -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("任务管理器已经关闭")
            if kind not in KINDS or not isinstance(parameters, dict):
                raise ValueError("不支持的任务类型或参数")
            python = self.runtime_paths.get(runtime)
            if python is None or not python.is_file():
                raise ValueError(f"{runtime} 运行环境未准备：请先使用环境准备工具")
            parameters = json.loads(
                json.dumps(
                    parameters,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=lambda value: str(value) if isinstance(value, Path) else value,
                )
            )
            if kind == "train":
                from .training import validate_training_request

                candidate = parameters.get("model")
                if not candidate and not parameters.get("resume_checkpoint"):
                    from .training import TrainingConfig

                    candidate = TrainingConfig(**parameters.get("config", {})).model_name
                if (
                    candidate
                    and not Path(candidate).is_file()
                    and (self.app_root / "models" / candidate).is_file()
                ):
                    parameters["model"] = str((self.app_root / "models" / candidate).resolve())
                validate_training_request(parameters)
            job_id = uuid4().hex
            job = Job(job_id, kind, "preparing", self.root / job_id, parameters, runtime)
            for active in self.active_jobs():
                capture_only = {"capture", "capture_stream"}
                if kind == active.kind == "train":
                    raise RuntimeError("已有训练任务运行；请先停止或等待完成")
                if (
                    kind not in capture_only
                    and active.kind not in capture_only
                    and (
                        self._device(job) == self._device(active)
                        or "auto" in (self._device(job), self._device(active))
                    )
                ):
                    raise RuntimeError(f"设备 {self._device(job)} 正被任务 {active.id[:8]} 使用")
            job.run_dir.mkdir()
            self.jobs[job.id] = job
            self._save(job)
            request = {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job.id,
                "kind": kind,
                "parameters": parameters,
                "run_dir": str(job.run_dir),
                "launch_guard": True,
            }
            atomic_write(job.run_dir / "request.json", json_text(request))
            config_dir = job.run_dir / "runtime-config"
            config_dir.mkdir()
            source = (
                self.app_root / "src"
                if (self.app_root / "src/yolo_workbench").is_dir()
                else self.app_root / "app"
            )
            if not (source / "yolo_workbench").is_dir():
                source = self.app_root
            env = {
                **os.environ,
                "PYTHONPATH": str(source),
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
                "YOLO_CONFIG_DIR": str(config_dir),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "YOLO_WORKBENCH_FONT": str(self.app_root / "fonts/DejaVuSans.ttf"),
                "YOLO_WORKBENCH_FONT_CJK": str(self.app_root / "fonts/NotoSansSC.ttf"),
            }
            stderr = (job.run_dir / "stderr.log").open("ab", buffering=0)
            try:
                job._tree = _ProcessTree()
                job._process = launch_process(
                    [
                        str(python),
                        "-m",
                        "yolo_workbench.worker",
                        "--request",
                        str(job.run_dir / "request.json"),
                    ],
                    cwd=job.run_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
                    if os.name == "nt"
                    else 0,
                    start_new_session=os.name != "nt",
                )
                job._tree.attach(job._process)
                job.pid = job._process.pid
                job.creation_identity = _creation_identity(job._process)
                self._save(job)
                job._thread = threading.Thread(
                    target=self._read, args=(job,), name=f"job-{job.id[:8]}", daemon=True
                )
                job._thread.start()
                atomic_write(job.run_dir / "launch.flag", "owned\n")
            except BaseException as exc:
                if job._process and job._process.poll() is None:
                    job._process.kill()
                    job._process.wait(timeout=10)
                if job._tree:
                    job._tree.close()
                job.error = str(exc)
                self._local(job, "state", {"state": "failed", "reason": str(exc)})
                raise
            finally:
                stderr.close()
            return job

    @staticmethod
    def _apply(job: Job, event: dict):
        if event["type"] == "state":
            job.state = event["data"]["state"]
        elif event["type"] == "result":
            job.result = event["data"]
        elif event["type"] == "error":
            job.error = event["data"].get("message", "未知任务错误")

    def _publish(self, job: Job, event: dict):
        with self._lock:
            job.sequence = event["sequence"]
            self._apply(job, event)
            with (job.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._save(job)
            if self._events.full():
                self._events.get_nowait()
            self._events.put_nowait(event)

    def _local(self, job: Job, kind: str, data: dict):
        self._publish(
            job,
            {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job.id,
                "sequence": job.sequence + 1,
                "timestamp": _now(),
                "type": kind,
                "data": data,
            },
        )

    def _read(self, job: Job):
        reader = EventReader(job.id)
        try:
            while True:
                line = job._process.stdout.readline(1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 1024 * 1024 or not line.endswith("\n"):
                    raise ValueError("Worker 事件过长或不完整")
                self._publish(job, reader.parse(line))
            job.returncode = job._process.wait()
            with self._lock:
                if job._cleanup and job.state in TERMINAL_STATES:
                    self._save(job)
                elif job._forced:
                    self._local(
                        job,
                        "state",
                        {"state": "interrupted", "reason": "用户强制终止；仅最近完整检查点可恢复"},
                    )
                elif job.returncode != 0 or job.state not in TERMINAL_STATES:
                    reason = f"Worker 异常退出（{job.returncode}），查看 stderr.log"
                    if not job.error:
                        self._local(job, "error", {"message": reason, "code": "worker_exit"})
                    self._local(job, "state", {"state": "failed", "reason": reason})
                else:
                    self._save(job)
        except Exception as exc:
            with self._lock:
                if job._tree:
                    job._tree.terminate()
                job.returncode = job._process.wait(timeout=10)
                self._local(job, "error", {"message": str(exc), "code": "protocol_error"})
                self._local(job, "state", {"state": "failed"})
        finally:
            job._process.stdout.close()
            job._tree.close()

    def poll_events(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def stop(self, job_id: str, force: bool = False):
        with self._lock:
            job = self.jobs[job_id]
            if job._process is None or job._process.poll() is not None:
                return
            if job.state in TERMINAL_STATES:
                if force:
                    job._cleanup = True
                    job._tree.terminate()
                return
            if force:
                job._forced = True
                job._tree.terminate()
            else:
                atomic_write(job.run_dir / "stop.flag", "stop at safe boundary\n")
                # The worker owns sequence numbers until stdout closes.
                job.state = "stopping"
                self._save(job)

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            jobs = self.active_jobs()
            for job in jobs:
                self.stop(job.id)
        deadline = time.monotonic() + 3
        for job in jobs:
            job._thread.join(max(0, deadline - time.monotonic()))
        for job in jobs:
            if job._process.poll() is None:
                self.stop(job.id, force=True)
            job._thread.join(10)
        self._ownership.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
