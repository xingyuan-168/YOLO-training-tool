"""Small durable file transactions. Readers use Project's exclusive writer lock."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


class OperationCancelled(RuntimeError):
    """A cooperative operation stopped between durable work units."""


def check_cancel(cancel=None) -> None:
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise OperationCancelled("操作已取消")


def report_progress(progress, *, phase: str, completed: int, total: int, path="") -> None:
    if progress is not None:
        progress({"phase": phase, "completed": completed, "total": total, "path": str(path)})


def atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data.encode("utf-8") if isinstance(data, str) else data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def child_path(root: Path, relative: str) -> Path:
    relative = safe_relative_path(relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("路径必须位于项目目录内")
    return path


def safe_relative_path(value: str) -> str:
    """Reject Windows aliases as well as traversal, even on non-Windows hosts."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("非法相对路径")
    name = value.replace("\\", "/")
    windows = PureWindowsPath(name)
    parts = name.rstrip("/").split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    reserved.update(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))
    reserved.update(f"{prefix}{i}" for prefix in ("COM", "LPT") for i in "¹²³")
    if windows.drive or windows.root or PurePosixPath(name).is_absolute():
        raise ValueError("路径不得为绝对路径或包含磁盘/UNC 前缀")
    if any(
        part in ("", ".", "..")
        or part.endswith((".", " "))
        or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
        or part.split(".", 1)[0].rstrip(" .").upper() in reserved
        for part in parts
    ):
        raise ValueError("路径含越界、Windows 保留名称或非法字符")
    return "/".join(parts)


def remove_owned_tree(parent: Path, target: Path, expected_name: str) -> None:
    """Remove only a fresh direct child explicitly owned by the calling operation."""
    parent = parent.resolve()
    resolved = target.resolve()
    if (
        resolved.parent != parent
        or resolved.name != expected_name
        or target.is_symlink()
        or (hasattr(target, "is_junction") and target.is_junction())
    ):
        raise RuntimeError("清理目标不在预期目录")
    if resolved.exists():
        shutil.rmtree(resolved)


class ProjectLock:
    """OS-owned lock; a crashed process releases it without trusting a stale PID."""

    def __init__(self, path: Path):
        self.stream = path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("该项目已由另一个实例打开") from exc

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()


class FileTransaction:
    """One journal per locked project. Interrupted writes roll back at next open."""

    def __init__(self, root: Path):
        self.root = root
        self.journal = root / ".transaction.json"

    def recover(self) -> bool:
        if not self.journal.exists():
            return False
        record = json.loads(self.journal.read_text(encoding="utf-8"))
        if type(record.get("committed")) is not bool:
            raise ValueError("事务日志的提交状态无效")
        values = record["after" if record["committed"] else "before"]
        targets = self._targets(values)
        for relative, content in values.items():
            target = targets[relative]
            if content is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write(target, content)
        self.journal.unlink()
        return True

    def _targets(self, changes: dict) -> dict[str, Path]:
        if not isinstance(changes, dict) or any(
            v is not None and not isinstance(v, str) for v in changes.values()
        ):
            raise ValueError("事务内容必须是文本文件映射")
        targets = {name: child_path(self.root, name) for name in changes}
        if self.journal.resolve() in targets.values() or self.root / ".writer.lock" in targets.values():
            raise ValueError("事务不能覆盖锁或事务日志")
        return targets

    def write(self, changes: dict[str, str | None]) -> None:
        if self.journal.exists():
            raise RuntimeError("存在未恢复事务，请重新打开项目")
        targets = self._targets(changes)
        record = {
            "committed": False,
            "before": {n: p.read_text(encoding="utf-8") if p.exists() else None for n, p in targets.items()},
            "after": changes,
        }
        atomic_write(self.journal, json_text(record))
        try:
            for name, content in changes.items():
                if content is None:
                    targets[name].unlink(missing_ok=True)
                else:
                    atomic_write(targets[name], content)
            record["committed"] = True
            atomic_write(self.journal, json_text(record))
            self.journal.unlink()
        except BaseException:
            self.recover()
            raise
