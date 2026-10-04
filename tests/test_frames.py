import subprocess
import sys

import pytest

from yolo_workbench.frames import SharedFrameReader, SharedFrameWriter


def test_shared_frame_copies_own_pixels_and_expires_overwritten_slot():
    with SharedFrameWriter() as writer, SharedFrameReader() as reader:
        first = writer.write(bytes([1, 2, 3]) * 12, 4, 3)
        image = reader.read(first)
        assert image["data"] == bytes([1, 2, 3]) * 12
        writer.write(bytes([4, 5, 6]) * 12, 4, 3)
        latest = writer.write(bytes([7, 8, 9]) * 12, 4, 3)
        assert reader.read(first) is None
        assert image["data"] == bytes([1, 2, 3]) * 12
        assert reader.read(latest)["data"] == bytes([7, 8, 9]) * 12


def test_resize_keeps_reader_handle_and_returns_new_geometry():
    with SharedFrameWriter() as writer, SharedFrameReader() as reader:
        first = writer.write(bytes(12), 2, 2)
        assert reader.read(first)["width"] == 2
        larger = writer.write(bytes([2]) * 30000, 100, 100)
        assert first["shm_name"] != larger["shm_name"]
        assert reader.read(first)["data"] == bytes(12)
        assert reader.read(larger)["height"] == 100


def test_reader_keeps_mapping_alive_after_writer_closes():
    writer, reader = SharedFrameWriter(), SharedFrameReader()
    reference = writer.write(bytes([2]) * 12, 2, 2)
    assert reader.read(reference)["data"] == bytes([2]) * 12
    writer.close()
    assert reader.read(reference)["data"] == bytes([2]) * 12
    reader.close()
    assert SharedFrameReader().read(reference) is None


def test_bounded_frame_size_and_reference_validation():
    with SharedFrameWriter(max_bytes=12) as writer, SharedFrameReader() as reader:
        with pytest.raises(ValueError):
            writer.write(bytes(300), 10, 10)
        with pytest.raises(ValueError):
            reader.read(
                {"schema_version": 1, "shm_name": "unrelated", "capacity": 10, "slot": 0, "sequence": 1}
            )


def test_import_does_not_load_native_inference_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import yolo_workbench.frames; "
            "assert not any(n in sys.modules for n in "
            "('numpy','cv2','torch','onnxruntime','ncnn','windows_capture'))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cross_process_pixels():
    import json

    with SharedFrameWriter() as writer:
        reference = writer.write(bytes([14, 29, 255]) * 12, 4, 3)
        script = (
            "import json,sys; from yolo_workbench.frames import SharedFrameReader; "
            "r=SharedFrameReader(); im=r.read(json.loads(sys.argv[1])); "
            "assert im['data']==bytes([14,29,255])*12; r.close()"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps(reference)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
