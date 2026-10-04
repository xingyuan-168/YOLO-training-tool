import json
import os
import subprocess
from pathlib import Path

import pytest

from yolo_workbench.models import inspect_ncnn, model_classes, resolve_model


def test_model_manifest_preserves_frozen_classes_and_rejects_path_escape(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test")
    metadata = {"schema_version": 1, "model_file": model.name, "classes": ["one", "two"]}
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    selected, frozen = resolve_model(tmp_path, "cq")
    assert selected == model
    assert resolve_model(tmp_path / "manifest.json", "cq")[0] == model
    assert model_classes(selected, metadata=frozen) == ["one", "two"]
    with pytest.raises(ValueError, match="冻结类别"):
        model_classes(selected, ["two", "one"], frozen)
    metadata["model_file"] = "../outside.onnx"
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="越界"):
        resolve_model(tmp_path, "cq")


def test_pt_sidecar_manifest_is_selected_before_folder_manifest(tmp_path):
    model = tmp_path / "last.pt"
    model.write_bytes(b"test")
    (tmp_path / "last.pt.manifest.json").write_text(json.dumps({"classes": ["frozen"]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"classes": ["other"]}))
    selected, metadata = resolve_model(model, "pt")
    assert model_classes(selected, metadata=metadata) == ["frozen"]
    assert resolve_model(tmp_path / "last.pt.manifest.json", "pt")[0] == model


def test_training_sidecar_without_model_file_resolves_exact_checkpoint(tmp_path):
    for name in ("last.pt", "best.pt"):
        (tmp_path / name).write_bytes(b"test")
    manifest = tmp_path / "last.manifest.json"
    manifest.write_text(json.dumps({"classes": ["frozen"], "family": "yolo11"}))
    assert resolve_model(manifest, "pt")[0] == tmp_path / "last.pt"
    assert resolve_model(tmp_path / "last.pt", "pt")[1]["family"] == "yolo11"


def test_ncnn_raw_multi_head_rejected_before_native_load(tmp_path):
    model = tmp_path / "bad.param"
    model.write_text(
        "7767517\n4 4\nInput in0 0 1 in0\n"
        "Convolution a 1 1 in0 out0\nConvolution b 1 1 in0 out1\n"
        "Convolution c 1 1 in0 out2\n"
    )
    with pytest.raises(ValueError, match="多检测头"):
        inspect_ncnn(model, "yolov8", ["one"])


def _numpy():
    return pytest.importorskip("numpy")


def test_letterbox_round_trip_and_class_aware_nms():
    np = _numpy()
    pytest.importorskip("cv2")
    from yolo_workbench.inference import decode_output, letterbox

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, geometry = letterbox(image, 320)
    assert tensor.shape == (3, 320, 320)
    # Overlap in same class suppressed; equal box in another class retained.
    output = np.array(
        [[160, 160, 160], [160, 160, 160], [160, 160, 160], [80, 80, 80], [0.9, 0.8, 0.1], [0.1, 0.1, 0.95]]
    )
    detections = decode_output(output, ["a", "b"], geometry)
    assert [d["class_id"] for d in detections] == [1, 0]
    assert detections[0]["xyxy"] == pytest.approx([50, 25, 150, 75])


def test_end_to_end_preserves_model_output_without_second_nms():
    np = _numpy()
    from yolo_workbench.inference import decode_output

    geometry = {"scale": 1, "left": 0, "top": 0, "width": 100, "height": 100}
    output = np.array([[[0, 0, 50, 50, 0.9, 0], [0, 0, 50, 50, 0.8, 0]]])
    detections = decode_output(output, ["frozen"], geometry, end_to_end=True)
    assert len(detections) == 2
    output[0, 0, 5] = 0.4
    with pytest.raises(ValueError, match="整数"):
        decode_output(output, ["frozen"], geometry, end_to_end=True)


def test_bad_output_and_nan_threshold_rejected():
    np = _numpy()
    from yolo_workbench.inference import decode_output, validate_thresholds

    with pytest.raises(ValueError):
        validate_thresholds(float("nan"))
    with pytest.raises(ValueError, match="冻结类别"):
        decode_output(np.zeros((9, 100)), ["one"], {})


def _repo():
    for path in Path(__file__).resolve().parents:
        if (path / ".runtimes/inference/Scripts/python.exe").is_file() and (path / "input/模型样板").is_dir():
            return path
    pytest.skip("Native integration needs prepared inference runtime and supplied samples")


def _isolated(code, tmp_path):
    root = _repo()
    own_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(root / ".runtimes/inference/Scripts/python.exe"), "-c", code, str(root), str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(own_root / "src"), "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
    )
    assert result.returncode == 0, (
        f"Native process exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    )


def test_cq_native_exact_parity_and_normalized_mapping(tmp_path):
    _isolated(
        """
import sys
from pathlib import Path
from ai_engine import Engine, image_from_numpy, AI_DEVICE_CPU
from yolo_workbench.inference import CqBackend
from yolo_workbench.worker_inference import read_image
root, temporary = Path(sys.argv[1]), Path(sys.argv[2])
path=root/'input/模型样板/best.onnx'
classes=(path.parent/'labels.txt').read_text(encoding='utf-8-sig').splitlines()
image=read_image(root/'input/UI-1.png')
with CqBackend(path,classes,temporary,device=AI_DEVICE_CPU) as backend:
    result=backend.predict(image,0.0)
    dll,config=backend.engine.dll_path,backend.config_path
    try:
        CqBackend(path,classes,temporary,device=AI_DEVICE_CPU)
        raise AssertionError('second CQ engine accepted')
    except RuntimeError:
        pass
with Engine(dll_path=dll) as direct:
    with direct.yolo_model(320,AI_DEVICE_CPU,0,1) as model:
        model.load_model(path,config)
        expected=model.infer(image_from_numpy(image),0.0)
assert result['raw_detections']==expected
assert result['runtime']['active']=='cpu'
assert len(result['detections'])==len(expected)>0
for normalized,raw in zip(result['detections'],expected):
    assert normalized['xyxy']==[raw[k] for k in ('x1','y1','x2','y2')]
    assert normalized['confidence']==raw['score']
    assert normalized['class_name']==classes[raw['class_id']]
print('CQ native parity passed',len(expected))
""",
        tmp_path,
    )


def test_ncnn_actual_zero_and_sample_decode_and_legacy_rejection(tmp_path):
    _isolated(
        """
import sys,numpy as np
from pathlib import Path
from yolo_workbench.inference import create_backend
from yolo_workbench.worker_inference import read_image
root, temporary=Path(sys.argv[1]),Path(sys.argv[2])
params={'backend':'ncnn','model':str(root/'input/模型样板/AScript_专用模型'),'device':'cpu'}
with create_backend(params,temporary) as backend:
    for image in (np.zeros((640,640,3),dtype=np.uint8),read_image(root/'input/UI-1.png')):
        result=backend.predict(image,.5)
        assert isinstance(result['detections'],list)
        assert result['runtime']['active']=='cpu'
    assert backend.manifest.output_shapes==[[10,8400]]
try:
    create_backend({**params,'model':str(root/'input/模型样板/best.ncnn.param')},temporary)
    raise AssertionError('raw NCNN model accepted')
except ValueError as error:
    assert '多检测头' in str(error)
print('NCNN zero and provided image passed')
""",
        tmp_path,
    )


def test_onnx_sample_and_worker_result_png(tmp_path):
    _isolated(
        """
import sys
from pathlib import Path
from yolo_workbench.worker_inference import handle
root,temporary=Path(sys.argv[1]),Path(sys.argv[2])
events=[]
result=handle({'protocol_version':1,'kind':'infer','job_id':'test','run_dir':str(temporary),
'parameters':{'backend':'onnx','model':str(root/'input/模型样板/best.onnx'),
'source':'image','source_path':str(root/'input/UI-1.png'),'device':'cpu'}},lambda t,d:events.append((t,d)))
assert result['frame_count']==1
assert Path(result['last_frame_path']).is_file()
assert result['runtime']['active']=='CPUExecutionProvider'
assert any(t=='frame' for t,d in events)
print('ONNX worker passed')
""",
        tmp_path,
    )
