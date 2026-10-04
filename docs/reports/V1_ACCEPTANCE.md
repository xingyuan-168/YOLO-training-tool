# YOLO 工作台桌面版验收记录

日期：2026-10-04。交付入口：`output/YOLOWorkbench/YOLOWorkbench.exe`。完整目录包含独立训练/推理 Python、已锁定依赖、官方三代 n 权重、中文字体、源码、许可和使用说明。输入材料 `input/` 只读使用。

## 已完成的产品流程

四个真实 PySide6 页面已接通项目、图片/YOLO/ZIP 导入、矩形标注与撤销重做、显式负样本、类别迁移/恢复、回收区、检查、分组划分和不可变快照。窗口/桌面采集、快捷键和定时采集可将图片入库；标注不加载 PyTorch 或原生推理引擎。

训练、标准评估、PT/ONNX/NCNN 导出、CQ_AI/NCNN/PT/ONNX 部署验证、基准测试和问题样本回流使用独立任务进程。安全停止、完整状态恢复、强制结束所属进程树、错误隔离和任务记录恢复均有对应测试。新训练保留优化器等完整状态，普通 stripped last.pt 使用权重续训。

首版任务索引由 `jobs/<id>/job.json` 与事件事实文件恢复；SQLite 用于图片索引。通用 TensorRT `.engine` 转换尚未开放，CQ TensorRT 适配保留为可选路径，待 NVIDIA 环境验收。

## 实际验证证据

| 项目 | 结果与范围 |
|---|---|
| 自动化回归 | GUI 环境执行全部当前测试：158 passed、4 skipped；跳过项需要独立重型环境，具体见下文 |
| Qt 标注 | 实际鼠标绘制、移动、缩放、类别切换、删除、撤销重做、负样本撤销、自动保存和项目重开通过 |
| 大数据浏览 | 10,000 条索引分页测试通过；初始不解码图片，页缓存/缩略图数量有界；不代表万张全量导入性能承诺 |
| 三代官方权重 | v8n/11n/26n 真正加载，正确系列通过，8 项错误系列检查拒绝；旧版 v8 缺失 yaml_file 的结构兼容已修复 |
| CPU 停止与恢复 | 真实小数据训练第 1 轮停止，使用完整 resume.pt 恢复到原计划第 3 轮，验证优化器等状态恢复 |
| 便携 GUI 闭环 | EXE 整目录移至含中文和空格的新路径；实际 CPU 训练 1 轮→标准评估→CQ ONNX 导出→CQ 识别通过。各 Worker 的 Python 网络守卫生效，GUI 未加载重型库；见 portable-report.json |
| CQ 回归 | 同模型、同配置、同图像，与直接调用 CQ_AI 0.14.6 的类别、坐标、分数及状态对齐；IoU 固定 0.45 |
| 数值转换 | v8/11/26 PT→ONNX 与 v8 PT→NCNN 四组真实检测回归通过；见 [export-parity.json](export-parity.json) |
| 长稳 | CQ CPU 真实连续推理 1,800.07 秒 / 38,314 帧；队列/共享内存有界，工作集未持续增长；见 [INFERENCE_ACCEPTANCE.md](INFERENCE_ACCEPTANCE.md) |
| 窗口与设备 | 自有 WGC 窗口客户区 344×201、96 DPI、最小化和关闭状态通过；Intel UHD DirectML 实际运行通过，部分节点由 CPU 执行 |
| 界面 | 1366×768、1920×1080，以及 Qt 150% 缩放离屏渲染；修复窄高度验证面板挤压，控件可滚动；此项不代替物理混合 DPI 多屏测试 |

轻量测试的 4 个跳过项是：真实训练集成、真实导出集成、官方权重检查，以及 GUI 环境缺少 OpenCV 的 letterbox 测试。前三项已在独立训练环境运行过；推理环境完成了 OpenCV/CQ 原生检查。各环境独立执行，未把 skip 算为 pass。

数值比较使用同一预处理和后处理，非空真实图片最差匹配 IoU 约 0.9999986、最大分数差约 9.24e-7，优于初始 0.99 / 0.001 门槛。合成图两边都无框的样本标为 vacuous，不计入有效通过。合成演示一轮训练的 mAP 可以为 0，这只验证工作流，不声明业务准确率。

## 环境与边界

本机：Windows 11 x64、Intel i7-13620H / Intel UHD、约 16 GB RAM。基础环境为 Python 3.12、PySide6 6.10.2、Ultralytics 8.4.7、PyTorch 2.9.1+cpu、CQ_AI 0.14.6；完整版本由 `uv.lock` 与包内 build-manifest.json 记录。

- CUDA 训练和 TensorRT 执行：缺少 NVIDIA 设备，待验。
- AScript Android 使用端：插件扫描因缺少 psutil 失败，当前没有连接目标设备；设备数量未知，真机加载/识别待验。桌面 NCNN 结构及真实图片数值测试已通过。
- Windows 10、全新电脑、实体混合 DPI 多屏、受保护游戏/模拟器窗口以及 WGC 连续 30 分钟：待对应环境验收。本次长稳输入是受控动态图像流。
- n/s/m/l/x 可选择；实测完整兼容矩阵集中于 n，其他规模需要准备权重并补充性能测量。
- 离线测试禁止 Worker 的 Python 网络连接，并验证完整解释器属于迁移后的包；没有将其表述为物理断网或全新操作系统测试。

## 重现与交付文件

```powershell
.runtimes/gui/Scripts/python.exe -m pytest tests -q
.runtimes/train/Scripts/python.exe scripts/validate_export_parity.py --help
.venv/Scripts/python.exe scripts/smoke_portable.py --bundle 'output/YOLOWorkbench' --output '.artifacts/portable-check'
.runtimes/gui/Scripts/python.exe scripts/review_desktop.py --project '<项目>' --output '.artifacts/ui-review'
```

主要证据：`portable-report.json`、`export-parity.json`、`inference-soak.json`、`test-results.xml`、`frontend-gate.json` 和最终 `finish-gate.json`。构建清单记录 EXE SHA-256 与源文件哈希。界面证据见 [标注](screenshots/annotate-1366.png)、[验证](screenshots/verify-1366.png)、[日志](screenshots/logs-1366.png)、[设置](screenshots/settings-1366.png)。

使用步骤与快捷键见 [USER_GUIDE.md](../USER_GUIDE.md)，第三方声明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。最终 Gate 以机器执行的 JSON 结果为准。
