"""Managed dataset facts, rebuildable SQLite index, grouped splits and immutable snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml
from PIL import Image

from .labels import Box, format_labels, parse_labels, validate_classes
from .storage import FileTransaction, ProjectLock, atomic_write, child_path, json_text

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
            self.db = sqlite3.connect(self.root / "index.sqlite3")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise ValueError("索引版本高于当前工具")
            self.db.execute("""CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, hash TEXT UNIQUE, name TEXT, status TEXT,
                width INTEGER, height INTEGER, session TEXT, deleted INTEGER)""")
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
        if not asset_id.isalnum():
            raise ValueError("非法资产 ID")
        return json.loads((self.root / "records" / f"{asset_id}.json").read_text(encoding="utf-8"))

    def _index(self, record: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?,?)",
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

    def rebuild_index(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM assets")
            for path in (self.root / "records").glob("*.json"):
                self._index(json.loads(path.read_text(encoding="utf-8")))

    def list_assets(
        self,
        *,
        status: str | None = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
        deleted: bool = False,
    ) -> list[dict]:
        if limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("非法分页参数")
        query = "SELECT * FROM assets WHERE deleted=? AND instr(name, ?) > 0"
        args: list = [int(deleted), search]
        if status:
            query += " AND status=?"
            args.append(status)
        rows = self.db.execute(query + " ORDER BY name, id LIMIT ? OFFSET ?", args + [limit, offset])
        return [dict(row) for row in rows]

    def import_image(
        self,
        source: Path,
        *,
        labels: str | None = None,
        confirmed_empty: bool = False,
        session: str | None = None,
    ) -> tuple[str, bool]:
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("不支持的图片格式")
        with Image.open(source) as image:
            width, height = image.size
            image.verify()
        # Decode after verify to catch truncated pixel data as well.
        with Image.open(source) as image:
            image.load()
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
        temporary = destination.with_suffix(destination.suffix + ".importing")
        try:
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != digest:
                raise ValueError("导入期间源文件发生变化，请重试")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "id": asset_id,
            "hash": digest,
            "file": relative,
            "name": source.name,
            "source": str(source.resolve()),
            "imported_at": utc_now(),
            "width": width,
            "height": height,
            "session": session,
            "deleted": False,
            "revision": 1,
            "status": "labeled" if boxes else "empty" if confirmed_empty else "pending",
        }
        changes = {f"records/{asset_id}.json": json_text(record)}
        if labels is not None or confirmed_empty:
            changes[f"labels/{asset_id}.txt"] = format_labels(boxes, len(self.project["classes"]))
        self.transaction.write(changes)
        with self.db:
            self._index(record)
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

    def recycle(self, asset_id: str, *, restore: bool = False) -> None:
        record = self._record(asset_id)
        record["deleted"] = not restore
        self.transaction.write({f"records/{asset_id}.json": json_text(record)})
        with self.db:
            self._index(record)

    def migrate_classes(self, names: list[str], mapping: dict[int, int | None], *, allow_drop=False) -> dict:
        names = validate_classes(names)
        old = self.project["classes"]
        if set(mapping) != set(range(len(old))):
            raise ValueError("类别迁移必须覆盖全部旧编号")
        if any(v is not None and (type(v) is not int or not 0 <= v < len(names)) for v in mapping.values()):
            raise ValueError("目标类别编号非法")
        changes: dict[str, str] = {}
        dropped = 0
        for path in (self.root / "labels").glob("*.txt"):
            boxes = parse_labels(path.read_text(encoding="utf-8"), len(old))
            migrated = []
            for box in boxes:
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
        if dropped and not allow_drop:
            raise ValueError(f"迁移将删除 {dropped} 个标注，必须明确确认")
        project = {**self.project, "classes": names, "class_revision": self.project["class_revision"] + 1}
        changes["project.json"] = json_text(project)
        changes["labels.txt"] = "\n".join(names) + "\n"
        before = {key: child_path(self.root, key).read_text(encoding="utf-8") for key in changes}
        migration_id = uuid4().hex
        changes[f"history/classes-{migration_id}.json"] = json_text(
            {"before": before, "after": changes.copy()}
        )
        self.transaction.write(changes)
        self.project = project
        self.rebuild_index()
        return {"migration_id": migration_id, "dropped": dropped, "classes": names}

    def restore_class_migration(self, migration_id: str) -> None:
        if not migration_id.isalnum():
            raise ValueError("非法迁移 ID")
        history = json.loads(
            (self.root / "history" / f"classes-{migration_id}.json").read_text(encoding="utf-8")
        )
        if any(
            child_path(self.root, k).read_text(encoding="utf-8") != v for k, v in history["after"].items()
        ):
            raise ValueError("迁移后已有新修改，不能覆盖；请创建新的反向迁移")
        self.transaction.write(history["before"])
        self.project = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        self.rebuild_index()

    def split(self, *, seed=42, train_ratio=0.8, grouped=True, exclude_pending=False) -> dict:
        if not 0 < train_ratio < 1:
            raise ValueError("训练比例必须在 0 和 1 之间")
        records = [dict(r) for r in self.db.execute("SELECT * FROM assets WHERE deleted=0 ORDER BY id")]
        if not exclude_pending and any(r["status"] == "pending" for r in records):
            raise ValueError("存在未审核图片，请明确排除或完成标注")
        groups: dict[str, list[str]] = {}
        for r in records:
            if r["status"] != "pending":
                key = (r["session"] or r["id"]) if grouped else r["id"]
                groups.setdefault(key, []).append(r["id"])
        units = sorted(groups)
        if len(units) < 2:
            raise ValueError("至少需要两个独立样本组才能划分训练集和验证集")
        random.Random(seed).shuffle(units)
        cutoff = max(1, min(len(units) - 1, round(len(units) * train_ratio)))
        result = {
            "seed": seed,
            "grouped": grouped,
            "train_ratio": train_ratio,
            "train": [i for key in units[:cutoff] for i in groups[key]],
            "val": [i for key in units[cutoff:] for i in groups[key]],
            "test": [],
        }
        self.transaction.write({"splits/default.json": json_text(result)})
        return result

    def snapshot(self, split: dict, parameters: dict) -> Path:
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
                    atomic_write(pending / "labels" / subset / f"{asset_id}.txt", labels)
                    records.append(
                        {
                            **record,
                            "subset": subset,
                            "label_sha256": hashlib.sha256(labels.encode()).hexdigest(),
                        }
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
                        "names": self.project["classes"],
                    },
                    allow_unicode=True,
                ),
            )
            destination = pending.with_suffix("")
            pending.rename(destination)
            return destination
        except BaseException:
            # Delete only this function's fresh, bounded construction directory.
            resolved = pending.resolve()
            if (
                resolved.parent != (self.root / "snapshots").resolve()
                or resolved.name != snapshot_id + ".building"
            ):
                raise RuntimeError("快照清理目标不在预期目录")
            shutil.rmtree(resolved)
            raise

    def validate(self) -> list[dict]:
        issues = []
        for row in self.db.execute("SELECT * FROM assets WHERE deleted=0"):
            record = self._record(row["id"])
            try:
                path = child_path(self.root, record["file"])
                with Image.open(path) as image:
                    image.verify()
                if file_hash(path) != record["hash"]:
                    raise ValueError("图片内容已被外部修改")
                boxes = self.load_boxes(row["id"])
                if row["status"] == "labeled" and not boxes:
                    raise ValueError("已标注图片缺失标签")
                if row["status"] == "empty" and boxes:
                    raise ValueError("负样本状态与标签不一致")
                if row["status"] == "pending":
                    issues.append({"id": row["id"], "severity": "warning", "message": "尚未审核"})
                if len(boxes) != len(set(boxes)):
                    issues.append({"id": row["id"], "severity": "warning", "message": "存在重复框"})
            except (OSError, ValueError) as exc:
                issues.append({"id": row["id"], "severity": "error", "message": str(exc)})
        known = {r[0] for r in self.db.execute("SELECT id FROM assets")}
        for path in (self.root / "labels").glob("*.txt"):
            if path.stem not in known:
                issues.append({"id": path.stem, "severity": "error", "message": "孤立标签"})
        return issues
