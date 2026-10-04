# 推理、采集与帧传输

部署模型和采集库只在任务子进程加载。GUI 可导入 `frames` 与 `capture.enumerate_windows/window_status`；这些导入不加载 NumPy、PyTorch、ONNX Runtime、NCNN 或 windows-capture。PT 使用训练环境；CQ_AI、NCNN、ONNX 和采集使用推理环境。

## Worker 接口

`worker_inference.handle(request, emit)` 接收协议 v1 的 `{job_id, kind, parameters, run_dir, protocol_version}`，返回一个最终结果字典。外层 Worker 负责 `running/result/succeeded/stopped/failed` 事件。取消使用 `run_dir/stop.flag`；本模块不终止其他进程。

任务种类：`infer`、`infer_stream`、`capture`、`capture_stream`、`benchmark`。

| 参数 | 内容 |
|---|---|
| `model` | 绝对模型路径、包目录或 JSON 清单；采集任务不需要 |
| `backend` | `cq / ncnn / onnx / pt` |
| `family` | `yolov8 / yolo11 / yolo26`；包中的冻结 family 优先 |
| `classes` | 可选冻结名称列表；与清单/labels/模型内名称冲突时拒绝 |
| `device` | `cpu / auto`；PT/ONNX 可用 CUDA 编号；CQ 可选 `directml/tensorrt` |
| `input_size` | 默认 640，32 的倍数；固定模型输入由实际模型/清单决定 |
| `confidence / iou` | 默认 0.5 / 0.45；CQ 拒绝更改 0.45 |
| `source` | `image / folder / video / window / desktop` |
| `source_path` | 图片、目录或视频路径 |
| `hwnd / monitor_index` | 显式目标窗口句柄 / 显示器编号（默认 1） |
| `client_only` | 默认 true，窗口仅客户区 |
| `max_fps` | 帧事件和 WGC 最大刷新率，默认 10，上限 120 |
| `save_frames / interval_seconds` | 连续采集定时保存，默认 false / 5；单张采集始终保存 |
| `max_frames / duration_seconds` | 可选正数运行上限，0 表示不设置 |
| `source_timeout_seconds` | 持续无有效新帧的停止时限，默认 10，允许 1–120 |
| `warmup / iterations` | benchmark 至少 20 / 200 |
| `stability_seconds` | benchmark 可选实际持续时间；来源提前结束不报完成 |

`infer` 对窗口/桌面处理一帧，对图片/目录/视频处理所有给定内容；`infer_stream` 对窗口/桌面持续处理。所有连续来源使用最新帧，允许丢弃过期画面。图片目录不会自动变成无限循环。

事件：

- `model`：`manifest` 和 `capabilities`，UI 据此设置阈值控件。
- `frame`：`frame` 为共享内存引用；其余字段含 `source/detections/runtime/elapsed_ms/end_to_end_ms`。推理还带 `model_sha256/model_path/backend/family/classes/confidence/iou`。
- `source_status`：`ready/minimized/hidden/empty/no_new_frame/closed/unavailable`，窗口信息含 HWND、PID、标题、物理坐标和 DPI。
- `capture_saved`：`output_path/source/saved_count`，可导入为待标注资产，不能把预测自动当真值。
- `progress/metrics`：性能阶段进度与实际测量统计。

检测结果统一为 `{xyxy:[x1,y1,x2,y2], class_id, class_name, confidence}`，坐标是原图像素。每帧结果逐条写入 `detections.jsonl` 或 `capture.jsonl`，最终结果含 `results_path/frame_count/saved_count/cancelled`。有效最终图像另存 `last-frame.png` 与来源/模型信息 `last-frame.json`；单张截图返回 `output_path`，连续定时截图使用 `capture-000001.png` 及同名 JSON。`input/` 不能作为输出目录。

## 模型契约与后处理

清单可使用 `model_file` 或 `model_path`，必须位于包内。支持 `manifest.json/model_manifest.json`、`last.pt.manifest.json` 与训练产生的 `last.manifest.json`；明确选择 sidecar 时定位对应 checkpoint。包中冻结类别优先；不从当前项目类别猜测模型输出的含义。

| 后端 | 检查与语义 |
|---|---|
| CQ_AI 0.14.6 | v8/11、opset12、FP32、batch1、320/640、`[1,4+C,N]`；原生最近邻 letterbox、跨类别 NMS、IoU=0.45；不追加 Python NMS |
| AScript v8 NCNN | 640、`[4+C,8400]`；输入 RGB/CHW/FP32、letterbox，按类别 NMS；原始三头样板在原生加载前拒绝 |
| 通用 NCNN | 单解码输出；v8/11 保留 DFL 结构检查，YOLO26 无 DFL/Softmax，允许清单声明的 `[4+C,N]` 并隔离确认实际输出 |
| 通用 ONNX | 一个 NCHW 输入与一个检测输出，FP32/FP16；`[1,4+C,N]` 走按类别 NMS，`[1,N,6]` 端到端结果不追加 NMS |
| PT | Ultralytics 原生预测与冻结模型 names；返回实际参数设备、精度及耗时 |

Ultralytics 8.4.7 导出 YOLO26 NCNN 时会关闭不受支持的 TopK 端到端分支，因此实测通用 NCNN 输出 `[84,2100]`（320、80 类），`capabilities.iou=true/end_to_end=false`。YOLO26 PT 和 ONNX 端到端模型则为 `iou=false/end_to_end=true`。UI 必须使用运行时 capabilities，不能只按 family 禁用 IoU。

CQ 原生数据另保留在 `CqBackend.predict()["raw_detections"]` 供精确回归；Worker 不传播该重复字段。一个进程只拥有一个 CQ Engine，关闭会释放全局原生资源。NCNN 的 NumPy 输入和 Mat 必须存活至 `extract()` 完成，输出再独立复制。CPU NCNN 明确关闭 fp16/bf16 运算与存储选项；未将 CPU 检查当作 Android AScript 验收。

## 采集与共享帧生命周期

窗口以 HWND 选取，同时记录 PID，防止窗口句柄复用。Win32 线程临时使用 Per-Monitor-V2 坐标上下文后恢复；客户区和 DWM 扩展窗口框转换为物理像素。WGC 根据 HWND 捕获并复制原生映射帧，桌面使用 DXGI duplication。捕获尺寸与当前客户区不同步时丢弃该帧；最小化、隐藏、空客户区、cloaked 窗口或全透明帧不能保存。关闭/不可用立即失败，无新帧显示状态并超时停止。完全黑但不透明的合法画面不被武断判为失效。

`LatestFrameQueue` 的容量为 1。`SharedFrameWriter` 每个映射有两个槽；改变尺寸时最多保留当前和上一个映射，单槽上限 64 MiB。帧事件只传引用和元数据。Windows 附加共享内存时返回页对齐后的实际容量，Reader 做边界检查而不要求容量数值完全相等。

```python
reader = SharedFrameReader()
copied = reader.read(event["data"]["frame"])
if copied is not None:
    # QImage(..., Format_BGR888).copy() before releasing this byte dictionary.
    display(copied["data"], copied["width"], copied["height"], copied["stride"])
reader.close()
```

Reader 在复制前后校验序号，过期或已关闭引用返回 None；返回的 bytes 不依赖写入器后续操作。Reader 自有 OS 句柄可维持映射直到关闭。最终 PNG 保证进程退出后仍可显示单张/最后一帧；不能假设尚未附加的 shm 引用在生产者退出后仍存在。

## 验证与限制

针对性测试：`tests/test_inference.py`、`tests/test_frames.py`、`tests/test_capture.py` 和原有 `test_contracts.py`。原生测试各自在推理子进程运行，覆盖 CQ 精确 native parity、标准化类别映射、NCNN 真正前向/解码、普通多头拒绝、ONNX Worker 图像产物，以及跨进程共享内存与 resize/退出生命周期。

`tests/capture_probe.py <output.png>` 创建并只采集自己的临时窗口，验证客户区像素和 ready/minimized/closed 状态。2026-10-04 在本机 Windows、DPI96 实测客户区 344×201 通过。其他 DPI、多个屏幕、受保护窗口和 DXGI 桌面实际场景仍需按使用环境验收。

v8、11、26 的 PT/ONNX 适配器已在本机 CPU 运行已有随机初始化架构模型；YOLO26 端到端阈值能力正确。CQ CPU 与 AUTO 已实跑，AUTO 报告 Intel UHD Graphics DirectML 并有 mixed CPU fallback；这不是 NVIDIA/CUDA/TensorRT 验收。一次真实 CQ CPU benchmark 完成 20 次预热和 200 次测量；数值受同时运行负载影响，只作为当前机器诊断，不是泛化性能承诺。适配器 P50/P95 包含预处理/执行/后处理，端到端另含读源，吞吐量按真实测量墙钟时间计算。

`scripts/soak_inference.py --repo-root <root> --seconds 1800` 使用自己生成的连续变化图像，执行真实 CQ CPU 推理、容量 1 队列和共享帧读写。预热后记录 RSS/句柄起点、峰值与结束值，检查 30 分钟实耗、帧数、映射数量、RSS 增量 ≤64 MiB 和句柄增量 ≤32。每 60 秒原子更新报告；只有 `state=passed` 才表示完成，`running/stopped/failed` 均不算通过。报告位于 `.artifacts/inference-soak/report.json`。此稳定性检查不代表 WGC 30 分钟或业务准确率验证。
