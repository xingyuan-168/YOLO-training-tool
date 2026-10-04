from __future__ import annotations

import json

import pytest
from PIL import Image

from yolo_workbench.dataset import DatasetService, file_hash
from yolo_workbench.labels import Box, format_labels, parse_labels
from yolo_workbench.storage import FileTransaction, atomic_write


@pytest.fixture
def project(tmp_path):
    with DatasetService.create(tmp_path / "中文 项目", "识别项目", ["正常", "宝剑"]) as service:
        yield service


def sample(tmp_path, name="原始 图片.png", color="red"):
    path = tmp_path / name
    Image.new("RGB", (101, 79), color).save(path)
    return path


@pytest.mark.parametrize(
    "line",
    [
        "0 nan .5 .2 .2",
        "0 .5 .5 0 .2",
        "2 .5 .5 .2 .2",
        "0.0 .5 .5 .2 .2",
        "0 .1 .5 .5 .5",
        "0 .5 .5 inf .1",
        "-1 .5 .5 .2 .2",
        "0 .5 .5 .2",
        "0 0 0 -1 -1",
    ],
)
def test_reject_invalid_detection_labels(line):
    with pytest.raises(ValueError):
        parse_labels(line, 2)


@pytest.mark.parametrize("size", [(101, 79), (1920, 1080), (320, 640)])
def test_pixel_roundtrip(size):
    pixels = (1.25, 2.5, size[0] - 0.75, size[1] - 1.1)
    box = Box.from_xyxy(1, pixels, *size)
    decoded = parse_labels(format_labels([box], 2), 2)[0]
    assert decoded.xyxy(*size) == pytest.approx(pixels, abs=1e-6)


def test_import_does_not_mutate_or_link_source_and_deduplicates(project, tmp_path):
    source = sample(tmp_path)
    before = file_hash(source)
    asset, created = project.import_image(source)
    duplicate, created_again = project.import_image(source)
    assert created and not created_again and duplicate == asset
    assert file_hash(source) == before
    source.write_bytes(b"external modification")
    assert not any(i["severity"] == "error" for i in project.validate())


def test_missing_or_empty_labels_require_explicit_negative_confirmation(project, tmp_path):
    asset, _ = project.import_image(sample(tmp_path), labels="")
    assert project.list_assets()[0]["status"] == "pending"
    project.save_boxes(asset, [], confirmed_empty=True)
    assert project.list_assets()[0]["status"] == "empty"
    project.save_boxes(asset, [Box(0, 0.5, 0.5, 0.2, 0.2)])
    project.save_boxes(asset, [])
    assert project.list_assets()[0]["status"] == "pending"


def test_invalid_import_leaves_source_and_project_intact(project, tmp_path):
    source = sample(tmp_path)
    with pytest.raises(ValueError):
        project.import_image(source, labels="0 1 1 1 1")
    assert source.exists() and not project.list_assets()


def test_lock_prevents_second_writer(project):
    with pytest.raises(RuntimeError, match="另一个实例"):
        DatasetService(project.root)


def test_index_rebuild_and_recycle_preserve_assets(project, tmp_path):
    asset, _ = project.import_image(sample(tmp_path), confirmed_empty=True)
    record_path = project.root / "records" / f"{asset}.json"
    original = record_path.read_bytes()
    project.db.execute("DELETE FROM assets")
    project.db.commit()
    project.rebuild_index()
    assert project.list_assets()[0]["id"] == asset
    project.recycle(asset)
    assert not project.list_assets()
    assert len(project.list_assets(deleted=True)) == 1
    project.recycle(asset, restore=True)
    assert record_path.read_bytes() == original


def audited_assets(project, tmp_path, count=6):
    ids = []
    for i in range(count):
        source = sample(tmp_path, f"{i}.png", (i * 30, 40, 120))
        asset, _ = project.import_image(source, labels="0 .5 .5 .2 .2", session=f"session-{i // 2}")
        ids.append(asset)
    return ids


def test_grouped_split_reproducible_and_no_session_leakage(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    split = project.split()
    assert split == project.split(seed=42)
    assert set(split["train"]).isdisjoint(split["val"])
    for i in range(0, len(ids), 2):
        assert (ids[i] in split["train"]) == (ids[i + 1] in split["train"])


def test_snapshot_immutable_after_label_and_class_changes(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    frozen = project.snapshot(project.split(), {"epochs": 2})
    before = {p.relative_to(frozen): file_hash(p) for p in frozen.rglob("*") if p.is_file()}
    project.save_boxes(ids[0], [Box(1, 0.3, 0.3, 0.2, 0.2)])
    project.migrate_classes(["宝剑", "正常"], {0: 1, 1: 0})
    project.recycle(ids[1])
    assert all(file_hash(frozen / path) == digest for path, digest in before.items())
    assert json.loads((frozen / "snapshot.json").read_text(encoding="utf-8"))["classes"] == ["正常", "宝剑"]


def test_class_migration_requires_drop_consent_and_is_recoverable(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    with pytest.raises(ValueError, match="明确确认"):
        project.migrate_classes(["宝剑"], {0: None, 1: 0})
    result = project.migrate_classes(["宝剑", "正常"], {0: 1, 1: 0})
    assert project.load_boxes(ids[0])[0].class_id == 1
    project.restore_class_migration(result["migration_id"])
    assert project.load_boxes(ids[0])[0].class_id == 0


def test_migration_restore_refuses_to_overwrite_new_edits(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    result = project.migrate_classes(["宝剑", "正常"], {0: 1, 1: 0})
    project.save_boxes(ids[0], [Box(1, 0.6, 0.6, 0.1, 0.1)])
    with pytest.raises(ValueError, match="新修改"):
        project.restore_class_migration(result["migration_id"])


def test_snapshot_failure_removes_only_own_build_directory(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    existing = project.snapshot(project.split(), {})
    project.recycle(ids[0])
    with pytest.raises(ValueError):
        project.snapshot({"train": ids[:3], "val": ids[3:]}, {})
    assert existing.exists()
    assert not list((project.root / "snapshots").glob("*.building"))


def test_transaction_rolls_back_after_mid_write_failure(tmp_path, monkeypatch):
    import yolo_workbench.storage as storage

    atomic_write(tmp_path / "a.txt", "before")
    write_count = 0
    original = storage.atomic_write

    def fail_once(path, data):
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise OSError("simulated disk failure")
        original(path, data)

    monkeypatch.setattr(storage, "atomic_write", fail_once)
    with pytest.raises(OSError):
        FileTransaction(tmp_path).write({"a.txt": "after", "b.txt": "new"})
    assert (tmp_path / "a.txt").read_text() == "before"
    assert not (tmp_path / "b.txt").exists()
    assert not (tmp_path / ".transaction.json").exists()


def test_recovery_after_process_interruption(tmp_path):
    import subprocess
    import sys

    atomic_write(tmp_path / "a.txt", "before")
    code = """
import os,sys
from pathlib import Path
import yolo_workbench.storage as storage
original=storage.atomic_write
def crash(path,data):
    original(path,data)
    if path.name == 'a.txt': os._exit(9)
storage.atomic_write=crash
storage.FileTransaction(Path(sys.argv[1])).write({'a.txt':'after','b.txt':'new'})
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], timeout=15)
    assert result.returncode == 9
    assert FileTransaction(tmp_path).recover()
    assert (tmp_path / "a.txt").read_text() == "before"
    assert not (tmp_path / "b.txt").exists()


def test_pending_requires_explicit_split_exclusion(project, tmp_path):
    audited_assets(project, tmp_path)
    project.import_image(sample(tmp_path, "pending.png", "green"))
    with pytest.raises(ValueError, match="未审核"):
        project.split()
    assert project.split(exclude_pending=True)


def test_corrupt_image_and_orphan_label_reported(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    record = project._record(ids[0])
    (project.root / record["file"]).write_bytes(b"broken")
    (project.root / "labels" / "orphan.txt").write_text("0 .5 .5 .2 .2")
    issues = project.validate()
    assert sum(i["severity"] == "error" for i in issues) == 2
