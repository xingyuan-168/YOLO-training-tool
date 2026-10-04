"""Versioned JSONL event envelope; no framework logging on the protocol stream."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

PROTOCOL_VERSION = 1
STATES = {"preparing", "running", "stopping", "stopped", "succeeded", "failed", "interrupted"}


class EventWriter:
    def __init__(self, job_id, stream):
        self.job_id, self.stream, self.sequence = job_id, stream, 0
        self.lock = threading.Lock()

    def emit(self, event_type: str, data: dict) -> dict:
        with self.lock:
            event = {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": self.job_id,
                "sequence": self.sequence + 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "type": event_type,
                "data": data,
            }
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            self.stream.write(encoded)
            self.stream.flush()
            self.sequence = event["sequence"]
            return event


class EventReader:
    def __init__(self, job_id):
        self.job_id, self.sequence = job_id, 0

    def parse(self, line: str) -> dict:
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("不支持的 Worker 协议版本")
        if event.get("job_id") != self.job_id:
            raise ValueError("任务 ID 不匹配")
        if type(event.get("sequence")) is not int or event["sequence"] != self.sequence + 1:
            raise ValueError("Worker 事件序号中断或乱序")
        if not isinstance(event.get("data"), dict) or not isinstance(event.get("type"), str):
            raise ValueError("非法事件内容")
        if event["type"] == "state" and event["data"].get("state") not in STATES:
            raise ValueError("未知任务状态")
        datetime.fromisoformat(event["timestamp"])
        self.sequence = event["sequence"]
        return event
