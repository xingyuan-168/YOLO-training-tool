from __future__ import annotations

import json

import pytest
from PIL import Image

from yolo_workbench.dataset import DatasetService, file_hash
from yolo_workbench.labels import Box, format_labels, parse_labels
from yolo_workbench.storage import FileTransaction, OperationCancelled, atomic_write, remove_owned_tree


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


def test_pagination_count_and_review_navigation_do_not_decode_images(project, tmp_path, monkeypatch):
    ids = []
    for i in range(5):
        asset, _ = project.import_image(sample(tmp_path, f"item-{i}.png", (i * 40, 10, 30)))
        ids.append(asset)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("index navigation must not decode image pixels")

    monkeypatch.setattr(Image, "open", forbidden)
    assert project.count_assets(status="pending", search="item-") == 5
    assert [r["id"] for r in project.iter_assets(page_size=2)] == ids
    assert project.list_assets(limit=2, offset=2)[0]["id"] == ids[2]
    assert project.next_unreviewed(ids[0])["id"] == ids[1]
    assert project.next_unreviewed(ids[-1])["id"] == ids[0]
    assert project.next_unreviewed(ids[-1], wrap=False) is None
    project.update_status(ids[1], "empty")
    assert project.next_unreviewed(ids[0])["id"] == ids[2]
    assert project.get_asset(ids[1])["status"] == "empty"
    assert project.get_path(ids[0]).is_relative_to(project.root / "assets")


def test_class_preview_is_read_only_and_restore_refuses_new_assets(project, tmp_path):
    ids = audited_assets(project, tmp_path)
    before = {p: file_hash(p) for name in ("labels", "records") for p in (project.root / name).iterdir()}
    preview = project.preview_class_migration(["宝剑"], {0: None, 1: 0})
    assert preview["dropped"] == 6 and preview["affected_images"] == 6
    assert preview["requires_confirmation"]
    assert all(file_hash(p) == digest for p, digest in before.items())
    result = project.migrate_classes(["宝剑"], {0: None, 1: 0}, allow_drop=True)
    assert project.count_assets(status="pending") == 6
    assert not project.load_boxes(ids[0])
    project.import_image(sample(tmp_path, "new.png", "blue"), confirmed_empty=True)
    with pytest.raises(ValueError, match="新修改"):
        project.restore_class_migration(result["migration_id"])


def test_explicit_test_split_snapshot_and_session_leakage(project, tmp_path):
    ids = audited_assets(project, tmp_path, 8)
    split = project.split(train_ratio=0.5, test_ratio=0.25)
    assert all(split[key] for key in ("train", "val", "test"))
    assert not project.validate_split(split)
    assert set(ids) == {i for key in ("train", "val", "test") for i in split[key]}
    snapshot = project.snapshot(split, {"epochs": 1})
    import yaml

    config = yaml.safe_load((snapshot / "data.yaml").read_text(encoding="utf-8"))
    assert config["test"] == "images/test"
    assert len(list((snapshot / "images/test").iterdir())) == len(split["test"])
    leaked = {"train": ids[::2], "val": ids[1::2], "test": []}
    assert any(i["code"] == "session_leakage" for i in project.validate_split(leaked))
    with pytest.raises(ValueError, match="跨越"):
        project.snapshot(leaked, {})


def test_validation_decodes_pixels_even_when_header_and_updated_hash_match(project, tmp_path):
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (100, 80), "red").save(path)
    asset, _ = project.import_image(path, confirmed_empty=True)
    managed = project.get_path(asset)
    managed.write_bytes(managed.read_bytes()[:-12])
    with Image.open(managed) as image:
        image.verify()
    record = project.get_asset(asset)
    record["hash"] = file_hash(managed)
    replacement = managed.with_name(record["hash"] + managed.suffix)
    managed.rename(replacement)
    record["file"] = replacement.relative_to(project.root).as_posix()
    atomic_write(project.root / "records" / f"{asset}.json", json.dumps(record))
    project.rebuild_index()
    issues = project.validate()
    assert any(i["severity"] == "error" and i["code"] == "invalid_asset" for i in issues)


def test_distributions_and_abnormal_labels(project, tmp_path):
    image = tmp_path / "tiny.png"
    Image.new("RGB", (16, 200), "blue").save(image)
    boxes = "1 .5 .5 .01 .01\n1 .5 .5 .01 .01\n"
    project.import_image(image, labels=boxes)
    stats = project.statistics()
    assert stats["total"] == 1 and stats["boxes"] == 2
    assert stats["classes"][1]["boxes"] == 2 and stats["classes"][1]["images"] == 1
    assert {i["code"] for i in project.validate()} >= {"duplicate_box", "abnormal_size", "tiny_box"}


def test_import_transaction_failure_removes_only_new_managed_bytes(project, tmp_path, monkeypatch):
    existing, _ = project.import_image(sample(tmp_path), confirmed_empty=True)
    kept = project.get_path(existing)

    def fail(_changes):
        raise OSError("full disk")

    monkeypatch.setattr(project.transaction, "write", fail)
    with pytest.raises(OSError, match="full disk"):
        project.import_image(sample(tmp_path, "new.png", "blue"))
    assert project.count_assets() == 1 and list((project.root / "assets").iterdir()) == [kept]


def test_index_rebuild_cancellation_rolls_back_previous_index(project, tmp_path):
    audited_assets(project, tmp_path)
    cancelled = False

    def progress(_event):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(OperationCancelled):
        project.rebuild_index(progress=progress, cancel=lambda: cancelled)
    assert project.count_assets() == 6


def test_snapshot_cancellation_cleans_build_only(project, tmp_path):
    audited_assets(project, tmp_path)
    split = project.split()
    existing = project.snapshot(split, {})
    cancelled = False

    def progress(_event):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(OperationCancelled):
        project.snapshot(split, {}, progress=progress, cancel=lambda: cancelled)
    assert list((project.root / "snapshots").iterdir()) == [existing]


def test_project_move_and_corrupt_index_recovery_do_not_need_original_source(tmp_path):
    source = sample(tmp_path)
    old_root = tmp_path / "old"
    with DatasetService.create(old_root, "可移动", ["目标"]) as service:
        asset, _ = service.import_image(source, confirmed_empty=True)
    source.unlink()
    destination = tmp_path / "moved"
    old_root.rename(destination)
    (destination / "index.sqlite3").write_bytes(b"corrupt derived index")
    with DatasetService(destination) as service:
        assert service.count_assets() == 1 and service.get_path(asset).is_file()
        assert not service.validate()


def test_duplicate_authoritative_records_do_not_silently_replace_index(project, tmp_path):
    asset, _ = project.import_image(sample(tmp_path), confirmed_empty=True)
    record = project.get_asset(asset)
    record["id"] = "differentid"
    atomic_write(project.root / "records/differentid.json", json.dumps(record))
    with pytest.raises(ValueError, match="重复内容哈希"):
        project.rebuild_index()
    assert project.count_assets() == 1


def test_owned_cleanup_rejects_outside_target_and_journal_validates_all_paths(tmp_path):
    parent = tmp_path / "owned"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("safe")
    with pytest.raises(RuntimeError):
        remove_owned_tree(parent, outside, outside.name)
    atomic_write(parent / "file", "before")
    atomic_write(
        parent / ".transaction.json",
        json.dumps({"committed": True, "after": {"file": "changed", "../outside/keep": "bad"}, "before": {}}),
    )
    with pytest.raises(ValueError):
        FileTransaction(parent).recover()
    assert (parent / "file").read_text() == "before"
    assert (outside / "keep").read_text() == "safe"


@pytest.mark.parametrize("point, expected", [("before_metadata", 0), ("after_metadata", 1)])
def test_image_import_process_interruption_recovers_complete_units(tmp_path, point, expected):
    import subprocess
    import sys

    source = sample(tmp_path)
    root = tmp_path / "project"
    with DatasetService.create(root, "恢复", ["目标"]):
        pass
    code = """
import os, sys
from pathlib import Path
from yolo_workbench.dataset import DatasetService
with DatasetService(Path(sys.argv[1])) as service:
    original = service.transaction.write
    def crash(changes):
        if sys.argv[3] == 'before_metadata': os._exit(23)
        original(changes)
        os._exit(23)
    service.transaction.write = crash
    service.import_image(Path(sys.argv[2]), confirmed_empty=True)
"""
    result = subprocess.run([sys.executable, "-c", code, str(root), str(source), point], timeout=15)
    assert result.returncode == 23
    assert (root / ".asset-import.json").exists()
    with DatasetService(root) as service:
        assert service.count_assets() == expected
        assert len(list((root / "assets").iterdir())) == expected
        assert not service.validate()
    assert not (root / ".asset-import.json").exists()


def test_open_removes_only_recognized_interrupted_owned_builds(tmp_path):
    root = tmp_path / "project"
    with DatasetService.create(root, "恢复", ["目标"]):
        pass
    incomplete = root / "snapshots" / ("a" * 32 + ".building")
    incomplete.mkdir()
    (incomplete / "partial").write_text("partial")
    keep = root / "snapshots" / "user-notes.building"
    keep.mkdir()
    temporary = root / "assets" / ("." + "b" * 32 + ".importing")
    temporary.write_text("partial")
    with DatasetService(root):
        assert not incomplete.exists() and not temporary.exists()
        assert keep.exists()


def test_confirmed_negative_missing_label_file_is_detected(project, tmp_path):
    asset, _ = project.import_image(sample(tmp_path), confirmed_empty=True)
    (project.root / "labels" / f"{asset}.txt").unlink()
    assert any(i["severity"] == "error" and "缺失标签" in i["message"] for i in project.validate())


def test_image_decompression_bomb_is_rejected_without_publishing_asset(project, tmp_path, monkeypatch):
    source = sample(tmp_path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)
    with pytest.raises(ValueError, match="像素数量"):
        project.import_image(source)
    assert not list((project.root / "assets").iterdir())


def test_damaged_index_schema_is_rebuilt_from_records(tmp_path):
    source = sample(tmp_path)
    root = tmp_path / "project"
    with DatasetService.create(root, "索引恢复", ["目标"]) as service:
        service.import_image(source, confirmed_empty=True)
        service.db.executescript("DROP TABLE assets; CREATE TABLE assets (unexpected TEXT);")
    with DatasetService(root) as service:
        assert service.count_assets() == 1 and not service.validate()
