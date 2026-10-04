from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from yolo_workbench import importers
from yolo_workbench.dataset import DatasetService, file_hash
from yolo_workbench.importers import (
    apply_imported_config,
    import_dataset,
    inspect_dataset,
    normalize_model_config,
    normalize_training_config,
)
from yolo_workbench.labels import Box


def picture(path: Path, color="red") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 64), color).save(path)
    return path


@pytest.fixture
def service(tmp_path):
    with DatasetService.create(tmp_path / "managed", "导入测试", ["cat", "dog"]) as value:
        yield value


def yolo_directory(tmp_path):
    root = tmp_path / "来源 数据"
    first = picture(root / "images/train/中文.png")
    second = picture(root / "images/val/中文.png", "blue")
    for subset in ("train", "val"):
        path = root / f"labels/{subset}/中文.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1 .5 .5 .25 .25\n", encoding="utf-8")
    (root / "data.yaml").write_text("names:\n  0: cat\n  1: dog\n", encoding="utf-8")
    return root, first, second


def test_directory_import_preserves_originals_same_stem_paths_and_source_splits(service, tmp_path):
    root, first, second = yolo_directory(tmp_path)
    before = {p: file_hash(p) for p in root.rglob("*") if p.is_file()}
    result = import_dataset(service, root)
    assert result["imported"] == 2 and result["labeled"] == 2 and not result["errors"]
    assert len(result["split"]["train"]) == len(result["split"]["val"]) == 1
    assert (service.root / "splits/default.json").is_file()
    assert all(file_hash(path) == digest for path, digest in before.items())
    first.write_bytes(b"source can change independently")
    second.unlink()
    assert not service.validate()


def test_empty_and_missing_labels_have_distinct_review_behavior(service, tmp_path):
    root = tmp_path / "source"
    picture(root / "empty.png")
    (root / "empty.txt").write_text("")
    picture(root / "missing.png", "blue")
    result = import_dataset(service, root, trust_empty_labels=True)
    assert result["empty"] == 1 and result["pending"] == 1
    by_name = {r["name"]: r["status"] for r in service.list_assets()}
    assert by_name == {"empty.png": "empty", "missing.png": "pending"}
    other = tmp_path / "default"
    picture(other / "untrusted.png", "green")
    (other / "untrusted.txt").write_text("")
    assert import_dataset(service, other)["pending"] == 1


def test_existing_classes_are_remapped_by_name_and_duplicates_keep_labels(service, tmp_path):
    source = tmp_path / "source"
    original = picture(source / "a.png")
    (source / "labels.txt").write_text("dog\ncat\n")
    (source / "a.txt").write_text("0 .5 .5 .25 .25\n")
    existing, _ = service.import_image(picture(tmp_path / "existing.png", "blue"), confirmed_empty=True)
    result = import_dataset(service, source)
    assert result["imported"] == 1 and service.project["classes"] == ["cat", "dog"]
    imported = next(r for r in service.list_assets() if r["id"] != existing)
    assert service.load_boxes(imported["id"])[0].class_id == 1
    service.save_boxes(imported["id"], [Box(0, 0.5, 0.5, 0.2, 0.2)])
    result = import_dataset(service, source)
    assert result["duplicates"] == 1 and result["imported"] == 0
    assert service.load_boxes(imported["id"])[0].class_id == 0
    assert original.exists()


def test_incompatible_classes_abort_existing_project_before_mutation(service, tmp_path):
    service.import_image(picture(tmp_path / "existing.png"), confirmed_empty=True)
    root = tmp_path / "new"
    picture(root / "a.png", "blue")
    (root / "labels.txt").write_text("unregistered\n")
    result = import_dataset(service, root)
    assert result["errors"] and result["skipped"] == 1
    assert service.count_assets() == 1 and service.project["classes"] == ["cat", "dog"]


def test_invalid_label_or_image_is_reported_without_partial_asset(service, tmp_path):
    root = tmp_path / "invalid"
    picture(root / "a.png")
    (root / "a.txt").write_text("0 nan .5 .3 .3")
    (root / "b.png").write_bytes(b"not an image")
    result = import_dataset(service, root)
    assert result["failed"] == 2 and len(result["errors"]) == 2
    assert not list((service.root / "assets").iterdir())
    assert not list((service.root / "records").iterdir())


def test_inspection_reads_metadata_without_decoding_images(tmp_path, monkeypatch):
    root, _, _ = yolo_directory(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preview must not decode images")

    monkeypatch.setattr(Image, "open", forbidden)
    preview = inspect_dataset(root)
    assert preview["classes"] == ["cat", "dog"] and preview["image_count"] == 2
    assert preview["label_count"] == 2 and not preview["errors"]


def test_zip_wrapper_import_is_bounded_and_source_reference_survives_cleanup(service, tmp_path, monkeypatch):
    source, _, _ = yolo_directory(tmp_path)
    archive = tmp_path / "数据集.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in source.rglob("*"):
            if path.is_file():
                stream.write(path, "wrapper/" + path.relative_to(source).as_posix())
    staging_parent = tmp_path / "temporary"
    staging_parent.mkdir()
    marker = staging_parent / "keep.txt"
    marker.write_text("unrelated")
    monkeypatch.setattr(importers.tempfile, "gettempdir", lambda: str(staging_parent))
    digest = file_hash(archive)
    preview = inspect_dataset(archive)
    assert preview["classes"] == ["cat", "dog"]
    result = import_dataset(service, archive)
    assert result["imported"] == 2 and not result["errors"]
    assert file_hash(archive) == digest and list(staging_parent.iterdir()) == [marker]
    record = service.get_asset(service.list_assets()[0]["id"])
    assert record["source"].startswith(str(archive.resolve()) + "!/wrapper/")
    assert service.get_path(record["id"]).is_file()


@pytest.mark.parametrize(
    "name",
    [
        "../escape.png",
        "..\\escape.png",
        "/absolute.png",
        "C:/drive.png",
        "C:relative.png",
        "\\\\server\\share\\file.png",
        "file:stream.png",
        "CON.txt",
        "folder./image.png",
        "folder /image.png",
    ],
)
def test_zip_rejects_windows_aliases_and_traversal_before_mutation(service, tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "payload")
    assert inspect_dataset(archive)["errors"]
    result = import_dataset(service, archive)
    assert result["errors"] and service.count_assets() == 0
    assert not (tmp_path / "escape.png").exists()


def test_zip_rejects_symlinks_case_collisions_and_file_directory_conflicts(service, tmp_path):
    for variant in ("link", "case", "collision"):
        archive = tmp_path / f"{variant}.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            if variant == "link":
                info = zipfile.ZipInfo("link.png")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                stream.writestr(info, "../outside.png")
            elif variant == "case":
                stream.writestr("A.png", "1")
                stream.writestr("a.png", "2")
            else:
                stream.writestr("folder", "file")
                stream.writestr("folder/a.png", "file")
        assert import_dataset(service, archive)["errors"]
    assert service.count_assets() == 0


def test_zip_bomb_size_ratio_and_member_limits(service, tmp_path, monkeypatch):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        stream.writestr("large.txt", b"0" * 1024 * 1024)
    assert import_dataset(service, archive)["errors"]
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("large.txt", "x" * 100)
    monkeypatch.setattr(importers, "MAX_FILE_BYTES", 50)
    assert import_dataset(service, archive)["errors"]
    monkeypatch.setattr(importers, "MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(importers, "MAX_FILES", 1)
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("a.txt", "a")
        stream.writestr("b.txt", "b")
    assert import_dataset(service, archive)["errors"]


def test_cancellation_keeps_only_complete_units_and_cleans_own_stage(service, tmp_path, monkeypatch):
    source, _, _ = yolo_directory(tmp_path)
    archive = tmp_path / "cancel.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for path in source.rglob("*"):
            if path.is_file():
                stream.write(path, path.relative_to(source))
    staging_parent = tmp_path / "temp"
    staging_parent.mkdir()
    monkeypatch.setattr(importers.tempfile, "gettempdir", lambda: str(staging_parent))
    cancelled = False

    def progress(event):
        nonlocal cancelled
        if event["phase"] == "import" and event["completed"] == 1:
            cancelled = True

    result = import_dataset(service, archive, progress=progress, cancel=lambda: cancelled)
    assert result["cancelled"] and result["imported"] == 1 and result["skipped"] == 1
    assert service.count_assets() == 1 and not service.validate()
    assert not list(staging_parent.iterdir())
    assert not (service.root / ".transaction.json").exists()


def test_legacy_values_are_typed_preserved_and_merged_without_defaults(service, tmp_path):
    raw = {
        "version": "yolov8",
        "model_size": "n",
        "input_size": "320",
        "batch_size": "-1",
        "epochs": "3000",
        "patience": "100",
        "device": "0",
        "workers": "12",
        "advanced_params": "",
        "项目目录": "D:/yolo/data/mouse",
    }
    root = tmp_path / "旧配置"
    root.mkdir()
    (root / "训练参数.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    (root / "模型配置.json").write_text(
        json.dumps(
            {
                "模型类型": "ncnn",
                "yolo版本": 8,
                "推理尺寸": 320,
                "标签文件": "labels.txt",
                "模型文件": "best.ncnn",
            }
        ),
        encoding="utf-8",
    )
    preview = inspect_dataset(root)
    training = preview["config"]["training"]
    assert not preview["errors"] and training["epochs"] == 3000 and training["batch"] == -1
    assert training["device"] == "0" and training["workers"] == 12 and training["imgsz"] == 320
    result = import_dataset(service, root)
    assert not result["errors"] and service.get_settings()["training"] == training
    assert service.get_settings()["model"]["family"] == "yolov8"
    assert service.get_settings()["legacy"]["训练参数.json"] == raw
    apply_imported_config(service, {"training": {"epochs": 321}, "model": {"confidence": ".7"}})
    assert service.get_settings()["training"]["epochs"] == 321
    assert service.get_settings()["training"]["batch"] == -1
    assert service.get_settings()["model"]["confidence"] == 0.7


@pytest.mark.parametrize(
    "raw",
    [
        {"epochs": True},
        {"epochs": "1.5"},
        {"device": "0,1"},
        {"input_size": 321},
        {"workers": -1},
        {"advanced_params": "data: /outside"},
        {"epochs": float("nan")},
        {"epochs": 10, "训练轮数": 20},
    ],
)
def test_invalid_typed_legacy_values_are_rejected(raw):
    with pytest.raises(ValueError):
        normalize_training_config(raw)


@pytest.mark.parametrize(
    "raw", [{"confidence": "nan"}, {"iou": True}, {"推理尺寸": "321"}, {"yolo版本": 99}, {"模型文件": ""}]
)
def test_invalid_model_config_is_rejected(raw):
    with pytest.raises(ValueError):
        normalize_model_config(raw)


def test_invalid_or_conflicting_class_metadata_is_not_silently_accepted(service, tmp_path):
    root = tmp_path / "source"
    picture(root / "a.png")
    (root / "data.yaml").write_text("names: {1: cat, 2: dog}")
    assert import_dataset(service, root)["errors"]
    (root / "data.yaml").write_text("names: [dog, cat]")
    (root / "labels.txt").write_text("cat\ndog\n")
    assert import_dataset(service, root)["errors"]
    assert service.count_assets() == 0


def test_yaml_file_lists_keep_declared_test_split_and_rebase_old_root(service, tmp_path):
    root = tmp_path / "relocated"
    for index, color in enumerate(("red", "blue", "green", "yellow")):
        picture(root / f"pictures/{index}.png", color)
        (root / f"pictures/{index}.txt").write_text("0 .5 .5 .2 .2")
    (root / "training-list.txt").write_text("./pictures/0.png\n./pictures/1.png\n")
    (root / "data.yaml").write_text(
        "path: D:/old/dataset\ntrain: training-list.txt\nval: pictures/2.png\ntest: D:/old/dataset/pictures/3.png\nnames: [cat, dog]\n"
    )
    result = import_dataset(service, root)
    assert not result["errors"] and result["imported"] == 4
    assert [len(result["split"][key]) for key in ("train", "val", "test")] == [2, 1, 1]
    assert any(w["code"] == "rebased_dataset_path" for w in result["warnings"])
    saved = json.loads((service.root / "splits/default.json").read_text(encoding="utf-8"))
    assert saved["test"] == result["split"]["test"]


def test_yaml_external_and_leaking_file_references_are_rejected(service, tmp_path):
    root = tmp_path / "source"
    picture(root / "a.png")
    (root / "data.yaml").write_text("train: ../outside\nval: a.png\nnames: [cat, dog]")
    assert import_dataset(service, root)["errors"]
    (root / "data.yaml").write_text("train: a.png\nval: a.png\nnames: [cat, dog]")
    assert import_dataset(service, root)["errors"]
    assert service.count_assets() == 0


def test_images_subdirectory_selection_does_not_import_sibling_outputs(service, tmp_path):
    root, _, _ = yolo_directory(tmp_path)
    picture(root / "outputs/prediction.png", "yellow")
    result = import_dataset(service, root / "images")
    assert not result["errors"] and result["imported"] == 2
    assert all(r["name"] != "prediction.png" for r in service.list_assets())


def test_partial_auto_batch_preview_defers_device_context_until_apply(service):
    config = normalize_training_config({"batch_size": "-1"})
    assert config == {"batch": -1}
    with pytest.raises(ValueError, match="CPU"):
        apply_imported_config(service, {"training": config})
    service.update_settings({"training": {"device": "0"}})
    apply_imported_config(service, {"training": config})
    assert service.get_settings()["training"] == {"device": "0", "batch": -1}


def test_single_yolo_image_reads_sibling_labels_without_importing_whole_manifest(service, tmp_path):
    root, first, _ = yolo_directory(tmp_path)
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [cat, dog]")
    result = import_dataset(service, first)
    assert not result["errors"] and result["imported"] == result["labeled"] == 1
    assert service.load_boxes(service.list_assets()[0]["id"])[0].class_id == 1
