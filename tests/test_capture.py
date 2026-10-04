import pytest

from yolo_workbench.capture import LatestFrameQueue, client_crop
from yolo_workbench.worker_inference import handle, summarize_benchmark


def test_latest_queue_discards_stale_frames_and_close_wakes_consumer():
    queue = LatestFrameQueue()
    queue.put("old")
    queue.put("new")
    assert queue.dropped == 1
    assert queue.get(0) == "new"
    assert queue.get(0) is None
    queue.put("discarded")
    queue.clear()
    assert queue.get(0) is None
    queue.close()
    queue.put("ignored")
    assert queue.get(0) is None


def test_client_crop_physical_coordinates_and_resize_mismatch():
    np = pytest.importorskip("numpy")
    frame = np.zeros((150, 220, 3), dtype=np.uint8)
    frame[30:130, 10:210] = 200
    metadata = {
        "client_rect": [110, 230, 310, 330],
        "capture_rect": [100, 200, 320, 350],
        "window_rect": [93, 200, 327, 357],
        "dpi": 144,
    }
    cropped = client_crop(frame, metadata)
    assert cropped.shape == (100, 200, 3)
    assert (cropped == 200).all()
    with pytest.raises(ValueError, match="尺寸不同步"):
        client_crop(np.zeros((160, 220, 3), dtype=np.uint8), metadata)


def test_capture_file_fallback_and_metadata(tmp_path):
    pytest.importorskip("numpy")
    from PIL import Image

    image = tmp_path / "source.png"
    Image.new("RGB", (20, 12), (200, 50, 30)).save(image)
    result = handle(
        {
            "protocol_version": 1,
            "kind": "capture",
            "run_dir": str(tmp_path / "run"),
            "parameters": {"source": "image", "source_path": str(image)},
        },
        lambda *_: None,
    )
    assert result["saved_count"] == 1
    assert result["source"]["path"] == str(image)
    with Image.open(result["output_path"]) as saved:
        assert saved.getpixel((0, 0)) == (200, 50, 30)


def test_worker_rejects_input_output_and_minimum_benchmark(tmp_path):
    with pytest.raises(ValueError, match="只读"):
        handle({"kind": "capture", "run_dir": str(tmp_path / "input")}, lambda *_: None)
    with pytest.raises(ValueError, match="至少预热"):
        handle(
            {"kind": "benchmark", "run_dir": str(tmp_path), "parameters": {"warmup": 1, "iterations": 2}},
            lambda *_: None,
        )


def test_benchmark_percentiles_and_real_elapsed_throughput():
    result = summarize_benchmark([10, 20, 30, 40, 50], [20, 30, 40, 50, 60], 1.25)
    assert result["p50_ms"] == 30
    assert result["p95_ms"] == 48
    assert result["throughput_fps"] == 4
    assert result["end_to_end_p95_ms"] == 58
