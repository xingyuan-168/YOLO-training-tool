"""Create a small, clearly synthetic dataset for learning the offline workflow."""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from yolo_workbench.dataset import DatasetService
from yolo_workbench.labels import Box, format_labels


def create_demo(destination: Path, count=36):
    randomizer = random.Random(42)
    with DatasetService.create(destination, "形状识别 · 合成演示", ["红色方块", "蓝色圆形"]) as service:
        with tempfile.TemporaryDirectory(prefix="yolo-demo-") as scratch:
            for index in range(count):
                image = Image.new("RGB", (640, 384), (225 + index % 10, 235, 245))
                draw = ImageDraw.Draw(image)
                for x in range(0, 640, 32):
                    draw.line((x, 0, x, 384), fill=(212, 224, 239))
                for y in range(0, 384, 32):
                    draw.line((0, y, 640, y), fill=(212, 224, 239))
                boxes = []
                if index % 9:
                    for class_id in (0, 1):
                        width = randomizer.randint(50, 115)
                        x = randomizer.randint(30 + 280 * class_id, 170 + 270 * class_id)
                        y = randomizer.randint(35, 230)
                        bounds = (x, y, x + width, y + width)
                        if class_id == 0:
                            draw.rectangle(bounds, fill=(223, 71, 78))
                        else:
                            draw.ellipse(bounds, fill=(53, 123, 227))
                        boxes.append(Box.from_xyxy(class_id, bounds, 640, 384))
                path = Path(scratch) / f"demo-{index + 1:03d}.png"
                image.save(path)
                service.import_image(
                    path,
                    labels=format_labels(boxes, 2),
                    confirmed_empty=not boxes,
                    session=f"synthetic-{index}",
                    source_reference="generated:shape-demo",
                    metadata={"synthetic": True, "seed": 42, "sample": index},
                )
        service.split()
        service.update_settings(
            {
                "training": {
                    "family": "yolov8",
                    "scale": "n",
                    "imgsz": 320,
                    "epochs": 10,
                    "batch": 4,
                    "workers": 0,
                    "device": "cpu",
                }
            }
        )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=36)
    args = parser.parse_args()
    print(create_demo(args.destination, args.count))
