"""Managed dataset facts, rebuildable SQLite index, grouped splits and immutable snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import warnings
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml
from PIL import Image

from .labels import Box, format_labels, parse_labels, validate_classes
from .storage import (
    FileTransaction,
    ProjectLock,
    atomic_write,
    check_cancel,
    child_path,
    json_text,
    remove_owned_tree,
    report_progress,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
STATUSES = {"pending", "labeled", "empty"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    """Verify the container and fully decode pixels under Pillow's size bound."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                dimensions = image.size
                image.verify()
            with Image.open(path) as image:
                image.load()
        return dimensions
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("图片像素数量超过安全限制") from exc


class DatasetService:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / "project.json").is_file():
            raise ValueError("不是工作台项目目录")
        self.lock = ProjectLock(self.root / ".writer.lock")
        try:
            self.transaction = FileTransaction(self.root)
            self.recovered = self.transaction.recover()
            self.project = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
            if self.project["schema_version"] != 1:
                raise ValueError("不支持的项目版本")
            validate_classes(self.project["classes"])
            self._recover_asset_import()
            self._cleanup_interrupted_builds()
            self.db = sqlite3.connect(self.root / "index.sqlite3")
            try:
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                healthy = self.db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            except sqlite3.DatabaseError:
                healthy, version = False, 0
            if version > 1:
                raise ValueError("索引版本高于当前工具")
            if healthy:
                columns = [row[1] for row in self.db.execute("PRAGMA table_info(assets)")]
                healthy = not columns or columns == [
                    "id",
                    "hash",
                    "name",
                    "status",
                    "width",
                    "height",
                    "session",
                    "deleted",
                ]
            if not healthy:
                self.db.close()
                # Only this derived file is discarded; JSON facts remain authoritative.
                for suffix in ("", "-wal", "-shm", "-journal"):
                    child_path(self.root, "index.sqlite3" + suffix).unlink(missing_ok=True)
                self.db = sqlite3.connect(self.root / "index.sqlite3")
            self.db.execute("""CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, hash TEXT UNIQUE, name TEXT, status TEXT,
                width INTEGER, height INTEGER, session TEXT, deleted INTEGER)""")
            self.db.execute("CREATE INDEX IF NOT EXISTS asset_listing ON assets(deleted, name, id)")
            self.db.execute("CREATE INDEX IF NOT EXISTS asset_status ON assets(deleted, status, name, id)")
            self.db.execute("PRAGMA user_version=1")
            self.db.row_factory = sqlite3.Row
            self.rebuild_index()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.lock.close()
            raise

    @classmethod
    def create(cls, root: Path, name: str, classes: list[str]) -> DatasetService:
        classes = validate_classes(classes)
        if not name.strip():
            raise ValueError("项目名称不能为空")
        if root.exists() and any(root.iterdir()):
            raise ValueError("新项目目录必须为空")
        root.mkdir(parents=True, exist_ok=True)
        for directory in ("assets", "labels", "records", "splits", "snapshots", "runs", "exports", "history"):
            (root / directory).mkdir(exist_ok=True)
        atomic_write(
            root / "project.json",
            json_text(
                {
                    "schema_version": 1,
                    "id": uuid4().hex,
                    "name": name.strip(),
                    "classes": classes,
                    "created_at": utc_now(),
                    "class_revision": 1,
                }
            ),
        )
        atomic_write(root / "labels.txt", "\n".join(classes) + "\n")
        return cls(root)

    def close(self) -> None:
        self.db.close()
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _record(self, asset_id: str) -> dict:
        if not isinstance(asset_id, str) or not asset_id.isascii() or not asset_id.isalnum():
            raise ValueError("非法资产 ID")
        record = json.loads((self.root / "records" / f"{asset_id}.json").read_text(encoding="utf-8"))
        self._validate_record(record, asset_id)
        return record

    def _validate_record(self, record: dict, asset_id: str) -> None:
        if not isinstance(record, dict) or record.get("id") != asset_id:
            raise ValueError(f"资产记录 ID 不一致：{asset_id}")
        digest = record.get("hash", "")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"资产哈希无效：{asset_id}")
        if record.get("status") not in STATUSES or type(record.get("deleted")) is not bool:
            raise ValueError(f"资产审核或回收状态无效：{asset_id}")
        if any(type(record.get(k)) is not int or record[k] < 1 for k in ("width", "height", "revision")):
            raise ValueError(f"资产尺寸或修订号无效：{asset_id}")
        if (
            not isinstance(record.get("name"), str)
            or record.get("session") is not None
            and not isinstance(record["session"], str)
        ):
            raise ValueError(f"资产名称或会话无效：{asset_id}")
        path = child_path(self.root, record.get("file"))
        if path.parent != (self.root / "assets").resolve() or path.stem != digest:
            raise ValueError(f"资产文件路径无效：{asset_id}")

    def get_asset(self, asset_id: str) -> dict:
        """Return a fresh authoritative record, including source and relative asset path."""
        return self._record(asset_id)

    def get_path(self, asset_id: str) -> Path:
        """Resolve the managed copy, never the original source path."""
        return child_path(self.root, self._record(asset_id)["file"])

    def _index(self, record: dict) -> None:
        self.db.execute(
            """INSERT INTO assets VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
            hash=excluded.hash, name=excluded.name, status=excluded.status,
            width=excluded.width, height=excluded.height, session=excluded.session, deleted=excluded.deleted""",
            (
                record["id"],
                record["hash"],
                record["name"],
                record["status"],
                record["width"],
                record["height"],
                record.get("session"),
                int(record.get("deleted", False)),
            ),
        )

    def _recover_asset_import(self) -> None:
        journal = self.root / ".asset-import.json"
        if not journal.exists():
            return
        pending = json.loads(journal.read_text(encoding="utf-8"))
        asset_id = pending.get("id")
        if (
            not isinstance(asset_id, str)
            or len(asset_id) != 32
            or any(c not in "0123456789abcdef" for c in asset_id)
        ):
            raise ValueError("导入恢复日志的资产 ID 无效")
        target = child_path(self.root, pending.get("file"))
        digest = pending.get("hash")
        if (
            target.parent != (self.root / "assets").resolve()
            or target.stem != digest
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("导入恢复日志的目标路径无效")
        record_path = self.root / "records" / f"{asset_id}.json"
        if record_path.exists():
            record = self._record(asset_id)
            if record["hash"] != digest or record["file"] != pending["file"] or not target.is_file():
                raise ValueError("导入恢复日志与已提交记录不一致")
        elif target.exists():
            if file_hash(target) != digest:
                raise ValueError("未完成导入的资产已被外部修改，拒绝清理")
            # Never discard an asset already referenced by a different valid record.
            if not any(
                self._record(p.stem)["file"] == pending["file"]
                for p in (self.root / "records").glob("*.json")
            ):
                target.unlink()
        journal.unlink()

    def _cleanup_interrupted_builds(self) -> None:
        for path in (self.root / "assets").glob(".*.importing"):
            token = path.name.removeprefix(".").removesuffix(".importing")
            if len(token) == 32 and all(c in "0123456789abcdef" for c in token) and not path.is_symlink():
                child_path(self.root, path.relative_to(self.root).as_posix()).unlink()
        for path in (self.root / "snapshots").glob("*.building"):
            token = path.name.removesuffix(".building")
            if len(token) == 32 and all(c in "0123456789abcdef" for c in token):
                remove_owned_tree(self.root / "snapshots", path, path.name)

    def rebuild_index(self, *, progress=None, cancel=None) -> int:
        paths = sorted((self.root / "records").glob("*.json"))
        with self.db:
            self.db.execute("DELETE FROM assets")
            for number, path in enumerate(paths, 1):
                check_cancel(cancel)
                try:
                    self._index(self._record(path.stem))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"资产记录存在重复内容哈希：{path.name}") from exc
                report_progress(progress, phase="index", completed=number, total=len(paths), path=path)
        return len(paths)

    def _query(self, status=None, search="", deleted=False) -> tuple[str, list]:
        if status is not None and status not in STATUSES:
            raise ValueError("未知审核状态")
        if not isinstance(search, str):
            raise ValueError("搜索词必须是文本")
        query = "FROM assets WHERE deleted=? AND instr(name, ?) > 0"
        args: list = [int(deleted), search]
        if status:
            query += " AND status=?"
            args.append(status)
        return query, args

    def count_assets(self, status=None, search="", deleted=False) -> int:
        query, args = self._query(status, search, deleted)
        return self.db.execute("SELECT COUNT(*) " + query, args).fetchone()[0]

    def list_assets(
        self,
        *,
        status: str | None = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
        deleted: bool = False,
    ) -> list[dict]:
        if type(limit) is not int or type(offset) is not int or limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("非法分页参数")
        query, args = self._query(status, search, deleted)
        rows = self.db.execute(
            "SELECT * " + query + " ORDER BY name, id LIMIT ? OFFSET ?", args + [limit, offset]
        )
        return [dict(row) for row in rows]

    def iter_assets(self, *, status=None, search="", deleted=False, page_size=200):
        """Yield index records a page at a time without opening image pixels."""
        for page in self.iter_asset_pages(status=status, search=search, deleted=deleted, page_size=page_size):
            yield from page

    def iter_asset_pages(self, *, status=None, search="", deleted=False, page_size=200):
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("非法分页参数")
        query, args = self._query(status, search, deleted)
        cursor = self.db.execute("SELECT * " + query + " ORDER BY name, id", args)
        try:
            while rows := cursor.fetchmany(page_size):
                yield [dict(row) for row in rows]
        finally:
            cursor.close()

    def next_unreviewed(self, current_id=None, *, search="", wrap=True) -> dict | None:
        query, args = self._query("pending", search)
        if current_id is not None:
            current = self._record(current_id)
            row = self.db.execute(
                "SELECT * " + query + " AND (name, id) > (?, ?) ORDER BY name, id LIMIT 1",
                args + [current["name"], current_id],
            ).fetchone()
            if row is not None:
                return dict(row)
            if not wrap:
                return None
        row = self.db.execute("SELECT * " + query + " ORDER BY name, id LIMIT 1", args).fetchone()
        return dict(row) if row else None

    def update_settings(self, settings: dict) -> dict:
        """Merge validated import settings by section, preserving absent existing values."""
        if not isinstance(settings, dict) or set(settings) - {"training", "model", "legacy"}:
            raise ValueError("未知项目配置部分")
        result = json.loads(json_text(self.project.get("settings", {})))
        for section, values in settings.items():
            if not isinstance(values, dict):
                raise ValueError("配置部分必须为对象")
            result[section] = {**result.get(section, {}), **values}
        project = {**self.project, "settings": result}
        self.transaction.write({"project.json": json_text(project)})
        self.project = project
        return json.loads(json_text(result))

    def get_settings(self) -> dict:
        return json.loads(json_text(self.project.get("settings", {})))

    def import_image(
        self,
        source: Path,
        *,
        labels: str | None = None,
        confirmed_empty: bool = False,
        session: str | None = None,
        source_reference: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[str, bool]:
        source = Path(source)
        if session is not None and not isinstance(session, str):
            raise ValueError("采集会话必须是文本")
        if type(confirmed_empty) is not bool:
            raise ValueError("负样本确认必须是布尔值")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("来源元数据必须为对象")
        if source_reference is not None and not isinstance(source_reference, str):
            raise ValueError("来源引用必须为文本")
        json_text(metadata)  # Reject non-JSON and non-finite values before copying bytes.
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("不支持的图片格式")
        if self.transaction.journal.exists():
            raise RuntimeError("存在未恢复事务，请重新打开项目")
        self._recover_asset_import()
        boxes = parse_labels(labels, len(self.project["classes"])) if labels is not None else []
        if confirmed_empty and boxes:
            raise ValueError("有目标的图片不能确认无目标")
        digest = file_hash(source)
        existing = self.db.execute("SELECT id FROM assets WHERE hash=?", (digest,)).fetchone()
        if existing:
            return existing[0], False
        asset_id = uuid4().hex
        relative = f"assets/{digest}{source.suffix.lower()}"
        destination = child_path(self.root, relative)
        # New bytes are copied into the managed area. Source is never hardlinked.
        temporary = destination.with_name(f".{uuid4().hex}.importing")
        created_file = False
        try:
            if destination.exists():
                if file_hash(destination) != digest:
                    raise ValueError("托管资产已损坏，拒绝原位覆盖")
                candidate = destination
            else:
                with source.open("rb") as stream, temporary.open("xb") as target:
                    shutil.copyfileobj(stream, target)
                    target.flush()
                    os.fsync(target.fileno())
                if file_hash(temporary) != digest:
                    raise ValueError("导入期间源文件发生变化，请重试")
                candidate = temporary
            # Validate the exact copied bytes, never a potentially changing source.
            width, height = image_dimensions(candidate)
            if candidate == temporary:
                atomic_write(
                    self.root / ".asset-import.json",
                    json_text({"id": asset_id, "file": relative, "hash": digest}),
                )
                os.replace(temporary, destination)
                created_file = True
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "id": asset_id,
            "hash": digest,
            "file": relative,
            "name": source.name,
            "source": source_reference if source_reference is not None else str(source.resolve()),
            "imported_at": utc_now(),
            "width": width,
            "height": height,
            "session": session,
            "deleted": False,
            "revision": 1,
            "status": "labeled" if boxes else "empty" if confirmed_empty else "pending",
        }
        if metadata:
            record["metadata"] = json.loads(json_text(metadata))
        changes = {f"records/{asset_id}.json": json_text(record)}
        if labels is not None or confirmed_empty:
            changes[f"labels/{asset_id}.txt"] = format_labels(boxes, len(self.project["classes"]))
        try:
            self.transaction.write(changes)
        except BaseException:
            # An interrupted transaction is recoverable. Never remove a committed asset.
            if (
                created_file
                and not (self.root / "records" / f"{asset_id}.json").exists()
                and not self.transaction.journal.exists()
            ):
                destination.unlink(missing_ok=True)
                (self.root / ".asset-import.json").unlink(missing_ok=True)
            raise
        with self.db:
            self._index(record)
        (self.root / ".asset-import.json").unlink(missing_ok=True)
        return asset_id, True

    def load_boxes(self, asset_id: str) -> list[Box]:
        self._record(asset_id)
        path = self.root / "labels" / f"{asset_id}.txt"
        return (
            parse_labels(path.read_text(encoding="utf-8"), len(self.project["classes"]))
            if path.exists()
            else []
        )

    def save_boxes(self, asset_id: str, boxes: list[Box], *, confirmed_empty: bool = False) -> None:
        if type(confirmed_empty) is not bool:
            raise ValueError("负样本确认必须是布尔值")
        record = self._record(asset_id)
        if record["deleted"]:
            raise ValueError("请先恢复已回收的图片")
        if confirmed_empty and boxes:
            raise ValueError("有目标的图片不能确认无目标")
        text = format_labels(boxes, len(self.project["classes"]))
        record.update(
            status="labeled" if boxes else "empty" if confirmed_empty else "pending",
            revision=record["revision"] + 1,
        )
        self.transaction.write(
            {f"labels/{asset_id}.txt": text, f"records/{asset_id}.json": json_text(record)}
        )
        with self.db:
            self._index(record)

    def update_status(self, asset_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError("未知审核状态")
        record = self._record(asset_id)
        if record["deleted"]:
            raise ValueError("请先恢复已回收的图片")
        boxes = self.load_boxes(asset_id)
        if status == "labeled" and not boxes or status == "empty" and boxes:
            raise ValueError("审核状态与标签内容不一致")
        if status == "empty":
            self.save_boxes(asset_id, [], confirmed_empty=True)
            return
        record.update(status=status, revision=record["revision"] + 1)
        self.transaction.write({f"records/{asset_id}.json": json_text(record)})
        with self.db:
            self._index(record)

    def recycle(self, asset_id: str, *, restore: bool = False) -> None:
        record = self._record(asset_id)
        record["deleted"] = not restore
        self.transaction.write({f"records/{asset_id}.json": json_text(record)})
        with self.db:
            self._index(record)

    def _class_migration(self, names, mapping, *, progress=None, cancel=None) -> tuple[dict, dict]:
        names = validate_classes(names)
        old = self.project["classes"]
        if (
            not isinstance(mapping, dict)
            or any(type(k) is not int for k in mapping)
            or set(mapping) != set(range(len(old)))
        ):
            raise ValueError("类别迁移必须覆盖全部旧编号")
        if any(v is not None and (type(v) is not int or not 0 <= v < len(names)) for v in mapping.values()):
            raise ValueError("目标类别编号非法")
        changes: dict[str, str] = {}
        dropped = affected = total_boxes = 0
        class_counts = [0] * len(old)
        paths = sorted((self.root / "labels").glob("*.txt"))
        for number, path in enumerate(paths, 1):
            check_cancel(cancel)
            boxes = parse_labels(path.read_text(encoding="utf-8"), len(old))
            migrated = []
            for box in boxes:
                class_counts[box.class_id] += 1
                target = mapping[box.class_id]
                if target is None:
                    dropped += 1
                else:
                    migrated.append(Box(target, box.cx, box.cy, box.width, box.height))
            changes[path.relative_to(self.root).as_posix()] = format_labels(migrated, len(names))
            record = self._record(path.stem)
            if boxes and not migrated:
                record["status"] = "pending"  # Removing a class does not approve a negative.
            record["revision"] += 1
            changes[f"records/{path.stem}.json"] = json_text(record)
            total_boxes += len(boxes)
            affected += int(
                any(
                    mapping[b.class_id] != b.class_id
                    or mapping[b.class_id] is not None
                    and old[b.class_id] != names[mapping[b.class_id]]
                    for b in boxes
                )
            )
            report_progress(progress, phase="classes", completed=number, total=len(paths), path=path)
        return changes, {
            "classes": names,
            "mapping": mapping.copy(),
            "dropped": dropped,
            "affected_images": affected,
            "total_boxes": total_boxes,
            "class_counts": class_counts,
            "requires_confirmation": bool(dropped),
        }

    def preview_class_migration(
        self, names: list[str], mapping: dict[int, int | None], *, progress=None, cancel=None
    ) -> dict:
        """Validate all current labels and show impact without touching any file."""
        return self._class_migration(names, mapping, progress=progress, cancel=cancel)[1]

    def _label_inventory(self) -> dict[str, str]:
        return {
            p.relative_to(self.root).as_posix(): file_hash(p)
            for directory in ("records", "labels")
            for p in sorted((self.root / directory).glob("*"))
            if p.is_file()
        }

    def migrate_classes(
        self,
        names: list[str],
        mapping: dict[int, int | None],
        *,
        allow_drop=False,
        progress=None,
        cancel=None,
    ) -> dict:
        changes, preview = self._class_migration(names, mapping, progress=progress, cancel=cancel)
        names, dropped = preview["classes"], preview["dropped"]
        if dropped and not allow_drop:
            raise ValueError(f"迁移将删除 {dropped} 个标注，必须明确确认")
        project = {**self.project, "classes": names, "class_revision": self.project["class_revision"] + 1}
        changes["project.json"] = json_text(project)
        changes["labels.txt"] = "\n".join(names) + "\n"
        before = {key: child_path(self.root, key).read_text(encoding="utf-8") for key in changes}
        migration_id = uuid4().hex
        inventory_after = self._label_inventory()
        for name, text in changes.items():
            if name.startswith(("records/", "labels/")):
                inventory_after[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        changes[f"history/classes-{migration_id}.json"] = json_text(
            {"before": before, "after": changes.copy(), "inventory_after": inventory_after}
        )
        check_cancel(cancel)
        self.transaction.write(changes)
        self.project = project
        self.rebuild_index()
        return {**preview, "migration_id": migration_id}

    def restore_class_migration(self, migration_id: str) -> None:
        if not isinstance(migration_id, str) or not migration_id.isascii() or not migration_id.isalnum():
            raise ValueError("非法迁移 ID")
        history = json.loads(
            (self.root / "history" / f"classes-{migration_id}.json").read_text(encoding="utf-8")
        )
        if "inventory_after" in history and self._label_inventory() != history["inventory_after"]:
            raise ValueError("迁移后已有新修改，不能覆盖；请创建新的反向迁移")
        if any(
            child_path(self.root, k).read_text(encoding="utf-8") != v for k, v in history["after"].items()
        ):
            raise ValueError("迁移后已有新修改，不能覆盖；请创建新的反向迁移")
        self.transaction.write(history["before"])
        self.project = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        self.rebuild_index()

    def split(
        self,
        *,
        seed=42,
        train_ratio=0.8,
        test_ratio=0.0,
        grouped=True,
        exclude_pending=False,
        progress=None,
        cancel=None,
    ) -> dict:
        if (
            type(seed) is not int
            or type(train_ratio) not in (int, float)
            or type(test_ratio) not in (int, float)
            or not math.isfinite(train_ratio + test_ratio)
            or not 0 < train_ratio < 1
            or not 0 <= test_ratio < 1
            or train_ratio + test_ratio >= 1
        ):
            raise ValueError("训练/测试比例非法，验证集比例必须大于零")
        records = [dict(r) for r in self.db.execute("SELECT * FROM assets WHERE deleted=0 ORDER BY id")]
        if not exclude_pending and any(r["status"] == "pending" for r in records):
            raise ValueError("存在未审核图片，请明确排除或完成标注")
        groups: dict[str, list[str]] = {}
        for number, r in enumerate(records, 1):
            check_cancel(cancel)
            if r["status"] != "pending":
                key = (r["session"] or r["id"]) if grouped else r["id"]
                groups.setdefault(key, []).append(r["id"])
            report_progress(progress, phase="split", completed=number, total=len(records))
        units = sorted(groups)
        required = 3 if test_ratio else 2
        if len(units) < required:
            raise ValueError(f"至少需要 {required} 个独立样本组才能划分训练/验证/测试集")
        random.Random(seed).shuffle(units)
        test_count = max(1, min(len(units) - 2, round(len(units) * test_ratio))) if test_ratio else 0
        cutoff = max(1, min(len(units) - 1 - test_count, round(len(units) * train_ratio)))
        val_end = len(units) - test_count
        result = {
            "seed": seed,
            "grouped": grouped,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "train": [i for key in units[:cutoff] for i in groups[key]],
            "val": [i for key in units[cutoff:val_end] for i in groups[key]],
            "test": [i for key in units[val_end:] for i in groups[key]],
        }
        check_cancel(cancel)
        self.transaction.write({"splits/default.json": json_text(result)})
        return result

    def validate_split(self, split: dict) -> list[dict]:
        issues = []
        if not isinstance(split, dict) or any(
            not isinstance(split.get(k, []), list) for k in ("train", "val", "test")
        ):
            return [
                {
                    "id": "split",
                    "severity": "error",
                    "code": "invalid_split",
                    "message": "划分必须包含图片 ID 列表",
                }
            ]
        if not split.get("train") or not split.get("val"):
            issues.append(
                {
                    "id": "split",
                    "severity": "error",
                    "code": "empty_split",
                    "message": "训练集与验证集不能为空",
                }
            )
        ids: dict[str, str] = {}
        hashes: dict[str, str] = {}
        sessions: dict[str, str] = {}
        for subset in ("train", "val", "test"):
            for asset_id in split.get(subset, []):
                try:
                    record = self._record(asset_id)
                    if asset_id in ids or record["hash"] in hashes:
                        raise ValueError("划分中存在重复图片或内容泄漏")
                    if record["deleted"] or record["status"] == "pending":
                        raise ValueError("划分中的图片已删除或尚未审核，请重新划分")
                    session = record.get("session")
                    if session and session in sessions and sessions[session] != subset:
                        issues.append(
                            {
                                "id": asset_id,
                                "severity": "error",
                                "code": "session_leakage",
                                "message": f"采集会话 {session} 跨越 {sessions[session]}/{subset}",
                            }
                        )
                    if session:
                        sessions[session] = subset
                    ids[asset_id], hashes[record["hash"]] = subset, subset
                except (OSError, ValueError) as exc:
                    issues.append(
                        {
                            "id": str(asset_id),
                            "severity": "error",
                            "code": "invalid_split",
                            "message": str(exc),
                        }
                    )
        return issues

    def snapshot(self, split: dict, parameters: dict, *, progress=None, cancel=None) -> Path:
        issues = self.validate_split(split)
        if issues:
            raise ValueError(issues[0]["message"])
        json_text(parameters)
        members = [i for key in ("train", "val", "test") for i in split.get(key, [])]
        if not split.get("train") or not split.get("val") or len(members) != len(set(members)):
            raise ValueError("划分必须有非空训练/验证集且不得重复")
        snapshot_id = uuid4().hex
        pending = self.root / "snapshots" / (snapshot_id + ".building")
        pending.mkdir()
        records = []
        try:
            for subset in ("train", "val", "test"):
                for asset_id in split.get(subset, []):
                    check_cancel(cancel)
                    record = self._record(asset_id)
                    if record["deleted"] or record["status"] == "pending":
                        raise ValueError("划分中的图片已删除或尚未审核，请重新划分")
                    source = child_path(self.root, record["file"])
                    if file_hash(source) != record["hash"]:
                        raise ValueError("原始资产内容与记录不符")
                    image_path = pending / "images" / subset / (asset_id + source.suffix)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, image_path)
                    except OSError:
                        shutil.copyfile(source, image_path)
                    labels = format_labels(self.load_boxes(asset_id), len(self.project["classes"]))
                    if record["status"] == "labeled" and not labels:
                        raise ValueError("已标注图片缺少标签")
                    if record["status"] == "empty" and labels:
                        raise ValueError("负样本状态与标签不一致")
                    atomic_write(pending / "labels" / subset / f"{asset_id}.txt", labels)
                    records.append(
                        {
                            **record,
                            "subset": subset,
                            "label_sha256": hashlib.sha256(labels.encode()).hexdigest(),
                        }
                    )
                    report_progress(
                        progress, phase="snapshot", completed=len(records), total=len(members), path=source
                    )
            manifest = {
                "schema_version": 1,
                "id": snapshot_id,
                "created_at": utc_now(),
                "classes": self.project["classes"],
                "split": split,
                "parameters": parameters,
                "assets": records,
            }
            atomic_write(pending / "snapshot.json", json_text(manifest))
            atomic_write(
                pending / "data.yaml",
                yaml.safe_dump(
                    {
                        "train": "images/train",
                        "val": "images/val",
                        **({"test": "images/test"} if split.get("test") else {}),
                        "names": self.project["classes"],
                    },
                    allow_unicode=True,
                ),
            )
            destination = pending.with_suffix("")
            check_cancel(cancel)
            pending.rename(destination)
            return destination
        except BaseException:
            # Delete only this function's fresh, bounded construction directory.
            remove_owned_tree(self.root / "snapshots", pending, snapshot_id + ".building")
            raise

    def statistics(self, *, progress=None, cancel=None) -> dict:
        """Count class/image distributions without decoding image pixels."""
        classes = [
            {"id": i, "name": name, "boxes": 0, "images": 0} for i, name in enumerate(self.project["classes"])
        ]
        result = {
            "total": self.count_assets(),
            "deleted": self.count_assets(deleted=True),
            "statuses": {s: 0 for s in sorted(STATUSES)},
            "classes": classes,
            "boxes": 0,
            "sizes": {},
            "issues": [],
        }
        for number, row in enumerate(self.iter_assets(), 1):
            check_cancel(cancel)
            result["statuses"][row["status"]] += 1
            size = f"{row['width']}x{row['height']}"
            result["sizes"][size] = result["sizes"].get(size, 0) + 1
            try:
                boxes = self.load_boxes(row["id"])
                for box in boxes:
                    classes[box.class_id]["boxes"] += 1
                for class_id in {b.class_id for b in boxes}:
                    classes[class_id]["images"] += 1
                result["boxes"] += len(boxes)
            except (OSError, ValueError) as exc:
                result["issues"].append({"id": row["id"], "severity": "error", "message": str(exc)})
            report_progress(progress, phase="statistics", completed=number, total=result["total"])
        return result

    def validate(self, *, split: dict | None = None, progress=None, cancel=None) -> list[dict]:
        issues = []
        total = self.count_assets()
        hashes: dict[str, str] = {}
        for number, row in enumerate(self.iter_assets(), 1):
            check_cancel(cancel)
            try:
                record = self._record(row["id"])
                path = child_path(self.root, record["file"])
                if image_dimensions(path) != (record["width"], record["height"]):
                    raise ValueError("图片尺寸与记录不符")
                if file_hash(path) != record["hash"]:
                    raise ValueError("图片内容已被外部修改")
                if record["hash"] in hashes:
                    issues.append(
                        {
                            "id": row["id"],
                            "severity": "error",
                            "code": "duplicate_image",
                            "message": f"内容与 {hashes[record['hash']]} 重复",
                        }
                    )
                hashes[record["hash"]] = row["id"]
                boxes = self.load_boxes(row["id"])
                if row["status"] != "pending" and not (self.root / "labels" / f"{row['id']}.txt").exists():
                    raise ValueError("已审核图片缺失标签文件")
                if row["status"] == "labeled" and not boxes:
                    raise ValueError("已标注图片缺失标签")
                if row["status"] == "empty" and boxes:
                    raise ValueError("负样本状态与标签不一致")
                if row["status"] == "pending":
                    issues.append(
                        {"id": row["id"], "severity": "warning", "code": "pending", "message": "尚未审核"}
                    )
                if len(boxes) != len(set(boxes)):
                    issues.append(
                        {
                            "id": row["id"],
                            "severity": "warning",
                            "code": "duplicate_box",
                            "message": "存在重复框",
                        }
                    )
                if (
                    min(record["width"], record["height"]) < 32
                    or max(record["width"] / record["height"], record["height"] / record["width"]) > 10
                ):
                    issues.append(
                        {
                            "id": row["id"],
                            "severity": "warning",
                            "code": "abnormal_size",
                            "message": "图片尺寸过小或宽高比超过 10",
                        }
                    )
                if any(b.width * record["width"] < 2 or b.height * record["height"] < 2 for b in boxes):
                    issues.append(
                        {
                            "id": row["id"],
                            "severity": "warning",
                            "code": "tiny_box",
                            "message": "存在边长不足 2 像素的框",
                        }
                    )
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                issues.append(
                    {"id": row["id"], "severity": "error", "code": "invalid_asset", "message": str(exc)}
                )
            report_progress(progress, phase="validate", completed=number, total=total)
        known = {r[0] for r in self.db.execute("SELECT id FROM assets")}
        for path in (self.root / "labels").glob("*.txt"):
            check_cancel(cancel)
            if path.stem not in known:
                issues.append(
                    {"id": path.stem, "severity": "error", "code": "orphan_label", "message": "孤立标签"}
                )
        if split is None and (self.root / "splits" / "default.json").exists():
            try:
                split = json.loads((self.root / "splits" / "default.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                issues.append(
                    {"id": "split", "severity": "error", "code": "invalid_split", "message": str(exc)}
                )
        if split is not None:
            issues.extend(self.validate_split(split))
        return issues
