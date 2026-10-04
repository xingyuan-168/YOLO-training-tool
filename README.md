# YOLO 本地工作台

面向 Windows 的个人 YOLO 标注、训练与部署验证工作台，采用 Python 3.12、PySide6、Ultralytics 和 CQ_AI 0.14.6。

当前已建立 M0 CPU 兼容基线、M1 四页交互原型和独立数据服务。生产 GUI 尚未实现，需要用户批准当前原型后进入 PySide6 阶段。完整产品的 M2–M6 验收尚未完成，不能把 HTML 原型作为成品。

- input/：用户原始资料，只读。
- output/：最终交付物，只放成品。
- docs/：需求、范围、架构、开源调研与决策记录。

输入材料和模型不纳入 Git；运行环境、下载缓存和临时测试产物也不纳入 Git。

开发分支：feat/yolo-workbench。治理使用已安装的 AI Engineering OS `aios` CLI，实际检查结果保存到阶段报告。

## 查看原型

直接用浏览器打开 `docs/design/PROTOTYPE.html`；该文件自包含，不需要构建或网络。配套交互规范在 `docs/design/UI_SPEC.md`。所有训练、指标和识别结果均为原型演示数据。

也可本地启动：

```powershell
uv run --locked python -m http.server 8765 --bind 127.0.0.1 --directory docs/design
```

浏览器访问 `http://127.0.0.1:8765/PROTOTYPE.html`。

## 开发与运行环境

要求 Windows x64、uv 与 Python 3.12。开发环境和三个角色环境相互独立：

```powershell
uv sync --locked --python 3.12
uv run --locked python scripts/prepare_runtime.py gui
uv run --locked python scripts/prepare_runtime.py train
uv run --locked python scripts/prepare_runtime.py inference
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
```

推理环境准备前，按 `vendor/README.md` 放入用户已有的 CQ_AI 0.14.6 Wheel；脚本先校验 SHA256，再安装到 `.runtimes/inference`，不会误用全局旧版本。该 Wheel 不从未知下载地址自动获取。

首次准备可下载依赖；已缓存后给准备命令添加 `--offline`。训练任务不调用准备命令、不静默升级包。CUDA/TensorRT 环境不属于当前 CPU 锁文件的验证结果。

## 可复现兼容检查

本机已提供的 `input/模型样板` 与 `input/UI-1.png` 保持只读。准备 train/inference 环境后运行：

```powershell
uv run --locked python scripts/smoke_compat.py
# 只重跑某项
uv run --locked python scripts/smoke_compat.py --only cq ncnn
```

每项在独立进程运行，结果写入 `.artifacts/compat`。CQ 回归使用阈值 0 产生非空框，比较适配层与直接引擎输出；此阈值只用于测试，不是产品默认值。三代 Nano 架构随机初始化用于前向/ONNX 验证；CPU 训练使用合成六图，仅检验管线和完整检查点，不衡量业务准确率。

已实现接口见 `docs/API_SPEC.md`，数据协议见 `docs/DATABASE.md`，实际进度和剩余验收见 `docs/reports/M0_M1_PROGRESS.md`。
