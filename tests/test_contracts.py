from dataclasses import replace

import pytest

from yolo_workbench.models import ModelManifest, validate_contract
from yolo_workbench.training import TrainingConfig


def test_cq_contract_matches_provided_six_class_sample():
    manifest = ModelManifest("yolov8", "onnx", list("abcdef"), [1, 3, 320, 320], [[1, 10, 2100]], "hash")
    validate_contract(manifest, "cq", opset=12)
    for bad in [
        replace(manifest, family="yolo26"),
        replace(manifest, output_shapes=[[1, 2100, 10]]),
        replace(manifest, classes=["one"]),
        replace(manifest, input_shape=[2, 3, 320, 320]),
        replace(manifest, precision="FP16"),
    ]:
        with pytest.raises(ValueError):
            validate_contract(bad, "cq", opset=12)
    with pytest.raises(ValueError):
        validate_contract(manifest, "cq", opset=17)


def test_ascript_contract_requires_decoded_640_single_output():
    manifest = ModelManifest("yolov8", "ncnn", list("abcdef"), [1, 3, 640, 640], [[10, 8400]], "hash")
    validate_contract(manifest, "ascript_v8")
    with pytest.raises(ValueError):
        validate_contract(
            replace(manifest, output_shapes=[[70, 40, 40], [70, 20, 20], [70, 10, 10]]), "ascript_v8"
        )


@pytest.mark.parametrize(
    "configuration",
    ["data: arbitrary", "resume: true", "project: C:/tmp", "[]", "amp: maybe", "lr0: .nan", "fliplr: 1.5"],
)
def test_expert_config_cannot_override_managed_fields(configuration):
    with pytest.raises(ValueError):
        TrainingConfig().effective(configuration)


def test_training_defaults_and_cuda_auto_batch():
    assert TrainingConfig().effective()["batch"] == 4
    assert TrainingConfig().effective()["workers"] == 0
    assert TrainingConfig(device="0", batch=-1, workers=4).effective()["batch"] == -1
    with pytest.raises(ValueError):
        TrainingConfig(batch=-1).effective()
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0).effective()
