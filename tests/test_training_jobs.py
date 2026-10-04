from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from yolo_workbench.dataset import DatasetService
from yolo_workbench.export_worker import _ncnn_names
from yolo_workbench.jobs import JobManager
from yolo_workbench.training import resolve_model, validate_training_request
from yolo_workbench.training_worker import digest, prepare_snapshot, validate_model_family


def fixture_project(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    project = tmp_path / "project"
    with DatasetService.create(project, "CPU production worker test", ["square"]) as dataset:
        for index in range(6):
            image = Image.new("RGB", (64, 64), (20 + index, 20, 20))
            ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill="red")
            source = fixtures / f"{index}.png"
            image.save(source)
            dataset.import_image(source, labels="0 0.5 0.5 0.5 0.5")
        snapshot = dataset.snapshot(dataset.split(grouped=False), {"epochs": 3})
    return project, snapshot


def test_request_preflight_and_snapshot_integrity(tmp_path):
    _, snapshot = fixture_project(tmp_path)
    parameters = {"snapshot": str(snapshot), "model": "yolov8n.yaml", "config": {"epochs": 3}}
    assert validate_training_request(parameters)[1]["epochs"] == 3
    with pytest.raises(ValueError, match="不会下载"):
        resolve_model("https://example.com/weights.pt")
    with pytest.raises(ValueError, match="不会下载"):
        resolve_model("yolov8n.pt")
    with pytest.raises(ValueError, match="不能同时"):
        validate_training_request({**parameters, "finetune": True, "resume_checkpoint": "resume.pt"})
    before = {str(path): digest(path) for path in snapshot.rglob("*") if path.is_file()}
    run = tmp_path / "run"
    run.mkdir()
    manifest, data_path = prepare_snapshot(snapshot, run)
    assert manifest["classes"] == ["square"]
    assert data_path.exists()
    assert before == {str(path): digest(path) for path in snapshot.rglob("*") if path.is_file()}
    label = next(snapshot.glob("labels/train/*.txt"))
    label.write_text("0 0.1 0.1 0.1 0.1\n")
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="发生变化"):
        prepare_snapshot(snapshot, second)


def test_ncnn_rejects_raw_multiheads_before_load(tmp_path):
    param = tmp_path / "model.param"
    param.write_text("7767517\n4 4\nInput input 0 1 in0\nSplit head 1 3 in0 out0 out1 out2\n")
    with pytest.raises(ValueError, match="单解码输出"):
        _ncnn_names(param)


@pytest.mark.parametrize("family", ["yolo11", "yolo26"])
def test_model_architecture_cannot_be_claimed_as_yolov8(family):
    # A renamed user PT file must not override the architecture stored inside it.
    loaded = SimpleNamespace(model=SimpleNamespace(yaml={"yaml_file": f"cfg/models/{family}n.yaml"}))
    with pytest.raises(ValueError, match=f"实际架构为 {family}"):
        validate_model_family(loaded, "yolov8")
    assert validate_model_family(loaded, family) == family


def test_unknown_architecture_cannot_acquire_false_manifest_family():
    loaded = SimpleNamespace(model=SimpleNamespace(yaml={"yaml_file": "custom.yaml"}))
    with pytest.raises(ValueError, match="无法验证"):
        validate_model_family(loaded, "yolov8")


def wait_job(manager, job, *, stop_first_epoch=False, timeout=240):
    events = []
    stopped = False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = manager.poll_events()
        events.extend(batch)
        if (
            stop_first_epoch
            and not stopped
            and any(event["type"] == "progress" and event["data"].get("epoch") == 1 for event in batch)
        ):
            manager.stop(job.id)
            stopped = True
        if not manager.active_jobs():
            events.extend(manager.poll_events())
            if job.state not in {"succeeded", "stopped"}:
                pytest.fail(
                    f"{job.state}: {job.error}\n{(job.run_dir / 'stderr.log').read_text(encoding='utf-8')[-10000:]}"
                )
            return events
        time.sleep(0.05)
    manager.stop(job.id, force=True)
    pytest.fail(f"Worker timeout; see {job.run_dir}")


@pytest.mark.skipif(
    os.environ.get("YOLO_RUN_TRAINING_INTEGRATION") != "1", reason="Explicit heavy runtime check"
)
def test_real_cpu_stop_resume_evaluate_and_exports(tmp_path):
    python = Path(os.environ["YOLO_TRAIN_PYTHON"])
    project, snapshot = fixture_project(tmp_path)
    app = Path(__file__).resolve().parents[1]
    config = {"family": "yolov8", "imgsz": 64, "epochs": 3, "batch": 2, "workers": 0, "device": "cpu"}
    parameters = {
        "snapshot": str(snapshot),
        "model": "yolov8n.yaml",
        "config": config,
        "expert_yaml": "amp: false\noptimizer: SGD\nwarmup_epochs: 1\nclose_mosaic: 0\n",
    }
    before = {str(path): digest(path) for path in snapshot.rglob("*") if path.is_file()}
    with JobManager(project, {"train": python}, app) as manager:
        first = manager.start("train", parameters)
        events = wait_job(manager, first, stop_first_epoch=True)
        assert first.state == "stopped"
        assert first.result["epochs_completed"] < 3
        assert any(event["type"] == "checkpoint" and event["data"]["full_state"] for event in events)
        checkpoint = Path(first.result["resume_checkpoint"])
        original_hash = digest(checkpoint)
        resumed = manager.start("train", {**parameters, "resume_checkpoint": str(checkpoint)})
        events = wait_job(manager, resumed)
        assert resumed.state == "succeeded"
        assert resumed.result["epochs_completed"] == 3
        assert resumed.result["start_epoch"] == first.result["epochs_completed"] + 1
        restoration = next(event["data"] for event in events if event["type"] == "resume")
        assert (
            restoration["optimizer_restored"]
            and restoration["scheduler_restored"]
            and restoration["rng_restored"]
        )
        assert restoration["optimizer_state_entries"] > 0
        assert digest(checkpoint) == original_hash
        model = resumed.result["best"]
        evaluation = manager.start("evaluate", {"model": model, "snapshot": str(snapshot), "imgsz": 64})
        wait_job(manager, evaluation)
        assert evaluation.result["source"] == "ultralytics_standard_evaluation"
        assert "metrics/mAP50(B)" in evaluation.result["metrics"]
        packages = []
        for format_name, profile, size in (
            ("pt", "generic", 64),
            ("onnx", "cq", 320),
            ("ncnn", "ascript_v8", 640),
        ):
            exported = manager.start(
                "export",
                {
                    "model": model,
                    "format": format_name,
                    "profile": profile,
                    "family": "yolov8",
                    "imgsz": size,
                    "output_dir": str(project / "exports"),
                },
            )
            wait_job(manager, exported)
            package = Path(exported.result["package_dir"])
            packages.append(package)
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["validation"]["passed"]
            assert manifest["classes"] == ["square"]
            assert (package / "example.py").exists()
        assert len(set(packages)) == 3
        assert before == {str(path): digest(path) for path in snapshot.rglob("*") if path.is_file()}
