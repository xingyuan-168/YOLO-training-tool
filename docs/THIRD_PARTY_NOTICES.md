# 第三方组件与许可

下表来自本次锁定运行环境的发行包元数据，完整许可文件随便携包放在 `licenses/` 和对应 `runtime/*/Lib/site-packages/` 中；Python 自带许可保留在独立运行目录。第三方组件的原有许可、版权和例外条款继续适用。交付物同时包含本工具的 Python 源码与构建配置。

| 组件 | 本次版本 | 发行包声明 |
|---|---|---|
| PySide6 / Qt | 6.10.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only；以各库许可为准 |
| PyQtGraph | 0.13.7 | MIT |
| Ultralytics | 8.4.7 | AGPL-3.0 |
| PyTorch / TorchVision | 2.9.1+cpu / 0.24.1+cpu | BSD 类许可及随包第三方 notices |
| ONNX | 1.19.1 | Apache-2.0 |
| ONNX Runtime | 1.23.2 | MIT |
| NCNN / PNNX | 1.0.20260526 / 20260526 | BSD-3-Clause |
| OpenCV Python | 5.0.0.93 | Apache-2.0 及随包第三方 notices |
| Windows Capture | 2.0.1 | MIT |
| NumPy / psutil | 2.2.6 / 7.0.0 | BSD 类许可 |
| Pillow / PyYAML | 11.3.0 / 6.0.2 | MIT-CMU / MIT |
| PyInstaller（构建工具） | 6.22.3 | GPLv2+，含 bootloader 特别例外 |

模型缓存来自 Ultralytics 官方资产；下载来源与 SHA-256 记录在 `models/cache-manifest.json`。CQ_AI 0.14.6 使用用户提供且哈希已核验的 Wheel，封装与配套 DLL 原样保留，不将其重新声明为以上开源许可。其内置 ONNX Runtime / DirectML 版本与独立 Python ORT 版本分别管理。

中文显示使用 Google Fonts 的 Noto Sans SC，适用 SIL Open Font License 1.1，许可位于 `fonts/OFL-NotoSansSC.txt`。训练图表兼容字体 DejaVu Sans 的许可位于 `fonts/LICENSE_DEJAVU`；字体下载与校验记录在 `fonts/manifest.json`。未复制 YOLO Annotator Desktop 的源码。

使用独立进程是运行时隔离措施，不改变各组件原有许可。此文档记录实际使用的组件及随附资料，不替代任何原始许可文本。
