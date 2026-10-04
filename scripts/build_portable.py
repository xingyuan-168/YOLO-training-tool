"""Build an onedir GUI with independent relocatable worker interpreters.

Unlike copying a venv, each worker includes its Python base, DLLs and stdlib.
No target is recursively deleted; a build requires a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_tree(source, target):
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git", ".pytest_cache"),
    )


def standalone_python(role, destination):
    environment = ROOT / ".runtimes" / role
    interpreter = environment / "Scripts" / "python.exe"
    base = Path(
        subprocess.check_output(
            [str(interpreter), "-c", "import sys; print(sys.base_prefix)"], text=True
        ).strip()
    )
    destination.mkdir(parents=True)
    for path in base.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    for name in ("DLLs", "Lib", "include", "libs"):
        if (base / name).exists():
            copy_tree(base / name, destination / name)
    site = destination / "Lib" / "site-packages"
    copy_tree(environment / "Lib" / "site-packages", site)
    # Editable .pth points at the developer checkout and must never ship.
    for path in site.glob("*.pth"):
        if "yolo_workbench" in path.name or str(ROOT) in path.read_text(encoding="utf-8", errors="replace"):
            path.unlink()
    (destination / "Scripts").mkdir(exist_ok=True)
    for path in (environment / "Scripts").glob("*"):
        if path.name.startswith("pnnx") and path.is_file():
            shutil.copyfile(path, destination / "Scripts" / path.name)
    # Application modules are added through PYTHONPATH by JobManager.
    clean_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    clean_env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1")
    probe = subprocess.check_output(
        [
            str(destination / "python.exe"),
            "-I",
            "-c",
            "import sys,json; print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,'path':sys.path}))",
        ],
        text=True,
        cwd=destination,
        env=clean_env,
        encoding="utf-8",
    )
    info = json.loads(probe)
    if Path(info["prefix"]).resolve() != destination.resolve():
        raise RuntimeError("运行环境仍依赖开发机路径")
    for path in info["path"]:
        if path and not Path(path).resolve().is_relative_to(destination.resolve()):
            raise RuntimeError(f"运行环境包含外部 Python 路径：{path}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "output" / "YOLOWorkbench")
    parser.add_argument(
        "--gui-only",
        action="store_true",
        help="Refresh executable only, retaining prepared worker environments",
    )
    args = parser.parse_args()
    target = args.destination.resolve()
    if not target.is_relative_to((ROOT / "output").resolve()) or target == (ROOT / "output").resolve():
        parser.error("交付目录必须在项目 output 内")
    if target.exists() and not args.gui_only:
        parser.error("交付目录已存在；选择新的空目录，避免覆盖用户项目")
    gui_python = ROOT / ".runtimes" / "gui" / "Scripts" / "python.exe"
    work = ROOT / ".artifacts" / "packaging"
    work.mkdir(parents=True, exist_ok=True)
    command = [
        str(gui_python),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        target.name,
        "--distpath",
        str(target.parent),
        "--workpath",
        str(work / "build"),
        "--specpath",
        str(work),
        "--paths",
        str(ROOT / "src"),
        "--exclude-module",
        "torch",
        "--exclude-module",
        "torchvision",
        "--exclude-module",
        "ultralytics",
        "--exclude-module",
        "onnxruntime",
        "--exclude-module",
        "ncnn",
        "--exclude-module",
        "ai_engine",
        "--exclude-module",
        "pytest",
        str(ROOT / "scripts" / "launch_gui.py"),
    ]
    # PyInstaller deletes its own existing output with --noconfirm. Build to a
    # separate managed staging directory when refreshing the executable.
    if args.gui_only:
        stage = work / "refresh"
        command[command.index("--distpath") + 1] = str(stage)
    build_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    build_env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1")
    # Ambient tool/plugin PATH entries can inject an incompatible ICU DLL into
    # Qt's bundle. Resolve native dependencies only from this Python and Windows.
    python_base = subprocess.check_output(
        [str(gui_python), "-I", "-c", "import sys; print(sys.base_prefix)"],
        text=True,
    ).strip()
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    build_env["PATH"] = os.pathsep.join(
        map(str, (gui_python.parent, Path(python_base), windows / "System32", windows))
    )
    subprocess.run(command, cwd=ROOT, check=True, env=build_env)
    if args.gui_only:
        source = work / "refresh" / target.name
        # A merge would retain DLLs removed from the new build. This directory
        # contains only our generated GUI runtime, never user projects/settings.
        internal = target / "_internal"
        if internal.exists():
            if not (target / "build-manifest.json").is_file() or internal.is_symlink():
                raise RuntimeError("不能刷新未确认来源的 GUI 运行目录")
            if any(p.is_symlink() or p.is_junction() for p in internal.rglob("*")):
                raise RuntimeError("GUI 运行目录包含链接，停止刷新")
            shutil.rmtree(internal)
        for path in source.iterdir():
            if path.is_dir():
                copy_tree(path, target / path.name)
            else:
                shutil.copyfile(path, target / path.name)
    prior_manifest = target / "build-manifest.json"
    runtimes = (
        json.loads(prior_manifest.read_text(encoding="utf-8")).get("runtimes", {})
        if args.gui_only and prior_manifest.is_file()
        else {}
    )
    if not args.gui_only:
        for role in ("train", "inference"):
            runtimes[role] = standalone_python(role, target / "runtime" / role)
    copy_tree(ROOT / "src" / "yolo_workbench", target / "app" / "yolo_workbench")
    for name in ("docs", "scripts", "fonts"):
        copy_tree(ROOT / name, target / name)
    (target / "models").mkdir(exist_ok=True)
    for path in (ROOT / "models").glob("*"):
        if path.is_file():
            shutil.copyfile(path, target / "models" / path.name)
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, target / name)
    licenses = target / "licenses"
    licenses.mkdir(exist_ok=True)
    for role in ("gui", "train", "inference"):
        site = ROOT / ".runtimes" / role / "Lib" / "site-packages"
        for package in site.glob("*.dist-info"):
            for path in package.rglob("*"):
                if path.is_file() and any(
                    word in path.name.lower() for word in ("license", "copying", "notice", "copyright")
                ):
                    out = licenses / role / package.name / path.relative_to(package)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, out)
    # Include all project source and reproducible build metadata alongside binaries.
    copy_tree(ROOT / "src", target / "source" / "src")
    copy_tree(ROOT / "tests", target / "source" / "tests")
    exe = target / (target.name + ".exe")
    with exe.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "executable": exe.name,
        "sha256": digest,
        "runtime_layout": "standalone Python 3.12 + isolated site-packages",
        "runtimes": runtimes,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "source_files": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
    }
    (target / "build-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"target": str(target), "executable": str(exe), "sha256": digest}))


if __name__ == "__main__":
    main()
