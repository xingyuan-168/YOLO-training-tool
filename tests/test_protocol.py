import io
import json

import pytest

from yolo_workbench.protocol import EventReader, EventWriter


def test_unicode_events_roundtrip_and_sequence():
    stream = io.StringIO()
    writer = EventWriter("任务-1", stream)
    writer.emit("state", {"state": "running"})
    writer.emit("progress", {"epoch": 1, "message": "已保存"})
    reader = EventReader("任务-1")
    events = [reader.parse(line) for line in stream.getvalue().splitlines()]
    assert events[1]["data"]["message"] == "已保存"
    assert [event["sequence"] for event in events] == [1, 2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "other"),
        ("sequence", 2),
        ("sequence", True),
        ("protocol_version", 9),
        ("data", []),
        ("type", None),
    ],
)
def test_mismatched_worker_messages_rejected(field, value):
    stream = io.StringIO()
    event = EventWriter("job", stream).emit("state", {"state": "running"})
    event[field] = value
    with pytest.raises(ValueError):
        EventReader("job").parse(json.dumps(event))


def test_failed_serialization_does_not_break_error_event_sequence():
    stream = io.StringIO()
    writer = EventWriter("job", stream)
    writer.emit("state", {"state": "running"})
    with pytest.raises(ValueError):
        writer.emit("metrics", {"loss": float("nan")})
    writer.emit("error", {"message": "训练返回非有限指标"})
    reader = EventReader("job")
    events = [reader.parse(line) for line in stream.getvalue().splitlines()]
    assert [event["sequence"] for event in events] == [1, 2]
