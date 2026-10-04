# YOLO 本地工作台

Windows 中文桌面工具：项目与图片导入、矩形标注、截图采集、数据检查与冻结、训练和完整恢复、标准评估、CQ_AI/NCNN/PT/ONNX 部署验证、模型导出及问题样本回流。

## 使用

便携入口：`output/YOLOWorkbench/YOLOWorkbench.exe`。保持整个目录一起移动，不单独复制 EXE。包内已包含独立 CPU 训练/推理环境、CQ_AI 0.14.6、YOLOv8n/11n/26n 官方权重和离线绘图字体。

操作步骤见 [中文使用说明](docs/USER_GUIDE.md)，实际检查、硬件范围和未验项目见 [首版验收报告](docs/reports/V1_ACCEPTANCE.md)。本机已验证 CPU 和 Intel DirectML；CUDA、TensorRT、AScript Android 真机需要目标设备验收。合成数据只证明流程，不表示业务准确率。

## 源码运行

要求 Windows x64、Python 3.12、uv。原始 `input/` 保持只读；Wheel 按 [vendor/README.md](vendor/README.md) 放入。

```powershell
uv sync --locked --python 3.12
uv run --locked python scripts/prepare_runtime.py gui
uv run --locked python scripts/prepare_runtime.py train
uv run --locked python scripts/prepare_runtime.py inference
.\.runtimes\train\Scripts\python.exe scripts/prepare_models.py --destination models
.\.runtimes\train\Scripts\python.exe scripts/prepare_fonts.py
.\.runtimes\gui\Scripts\python.exe -m yolo_workbench
```

准备命令显式联网；已有依赖缓存可给 prepare_runtime 添加 `--offline`。任务执行期间不会自动安装或升级。GUI 不加载 PyTorch、CQ DLL、NCNN 或 ONNX Runtime；后端在独立子进程运行。

## 开发与验证

```powershell
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
.\.runtimes\gui\Scripts\python.exe -m pytest tests/test_desktop.py -q
uv run --locked python scripts/smoke_compat.py
```

GUI 测试需要 GUI 环境中安装锁定的开发测试依赖。真实训练/恢复/导出测试由 `YOLO_RUN_TRAINING_INTEGRATION=1` 和 `YOLO_TRAIN_PYTHON` 显式启用；数值一致性工具见 `scripts/validate_export_parity.py`。构建前在 GUI 环境准备 build 依赖组，再执行 `scripts/build_portable.py`；重建 GUI 使用 `--gui-only`，不会覆盖用户项目。

输入、权重、运行环境、生成数据和打包产物不进入 Git。接口见 [API_SPEC](docs/API_SPEC.md)，结构见 [ARCHITECTURE](docs/ARCHITECTURE.md)，复用及许可证见 [OPEN_SOURCE_RESEARCH](docs/OPEN_SOURCE_RESEARCH.md)。经用户批准的 HTML 只作设计依据，生产程序使用 PySide6。

