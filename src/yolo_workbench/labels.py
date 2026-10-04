"""Strict YOLO detection labels, normalized coordinates, original-pixel conversion."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def validate(self, class_count: int) -> None:
        if type(self.class_id) is not int or not 0 <= self.class_id < class_count:
            raise ValueError("非法类别编号")
        if not all(math.isfinite(v) for v in (self.cx, self.cy, self.width, self.height)):
            raise ValueError("坐标必须为有限数值")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("边框面积必须大于零")
        if not (self.width / 2 - 1e-9 <= self.cx <= 1 - self.width / 2 + 1e-9):
            raise ValueError("边框水平越界")
        if not (self.height / 2 - 1e-9 <= self.cy <= 1 - self.height / 2 + 1e-9):
            raise ValueError("边框垂直越界")

    def xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        return (
            (self.cx - self.width / 2) * width,
            (self.cy - self.height / 2) * height,
            (self.cx + self.width / 2) * width,
            (self.cy + self.height / 2) * height,
        )

    @classmethod
    def from_xyxy(cls, class_id: int, xyxy: tuple, width: int, height: int) -> Box:
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸必须大于零")
        x1, y1, x2, y2 = xyxy
        return cls(
            class_id, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height
        )


def parse_labels(text: str, class_count: int) -> list[Box]:
    boxes = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError("目标检测标签每行必须有五个字段")
            box = Box(int(tokens[0]), *(float(v) for v in tokens[1:]))
            box.validate(class_count)
            boxes.append(box)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"第 {line_number} 行：{exc}") from exc
    return boxes


def format_labels(boxes: list[Box], class_count: int) -> str:
    for box in boxes:
        box.validate(class_count)
    return "".join(f"{b.class_id} {b.cx:.10f} {b.cy:.10f} {b.width:.10f} {b.height:.10f}\n" for b in boxes)


def validate_classes(names: list[str]) -> list[str]:
    if not names or any(not isinstance(n, str) or not n.strip() or "\n" in n or "\r" in n for n in names):
        raise ValueError("至少需要一个有效类别名称，名称不能包含换行")
    normalized = [n.strip() for n in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("类别名称不能重复")
    return normalized
