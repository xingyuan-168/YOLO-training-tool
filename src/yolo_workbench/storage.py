"""Small durable file transactions. Readers use Project's exclusive writer lock."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


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
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("路径必须位于项目目录内")
    return path


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
        values = record["after" if record["committed"] else "before"]
        for relative, content in values.items():
            target = child_path(self.root, relative)
            if content is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write(target, content)
        self.journal.unlink()
        return True

    def write(self, changes: dict[str, str | None]) -> None:
        if self.journal.exists():
            raise RuntimeError("存在未恢复事务，请重新打开项目")
        targets = {name: child_path(self.root, name) for name in changes}
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
