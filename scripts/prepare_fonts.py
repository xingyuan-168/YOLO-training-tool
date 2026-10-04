"""Prepare redistributable fonts explicitly, before offline training or packaging."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    import matplotlib

    root = Path(__file__).resolve().parents[1]
    target = root / "fonts"
    target.mkdir(exist_ok=True)
    original = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name in ("DejaVuSans.ttf", "LICENSE_DEJAVU"):
        shutil.copyfile(original / name, target / name)
    base = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/"
    for remote, local in (("NotoSansSC%5Bwght%5D.ttf", "NotoSansSC.ttf"), ("OFL.txt", "OFL-NotoSansSC.txt")):
        destination = target / local
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".download")
            try:
                with (
                    urllib.request.urlopen(base + remote, timeout=120) as response,
                    temporary.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    records = []
    for path in sorted(target.glob("*")):
        if path.suffix == ".ttf":
            with path.open("rb") as stream:
                records.append(
                    {"file": path.name, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
                )
    (target / "manifest.json").write_text(
        json.dumps({"source": base, "fonts": records}, indent=2), encoding="utf-8"
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
