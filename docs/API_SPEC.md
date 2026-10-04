# API Spec

生产接口均由本仓库实现；重型训练、转换和推理库仅在 Worker 中加载。

| 模块 | 接口与责任 | 详细契约 |
|---|---|---|
| dataset / importers | DatasetService、受管理导入、标签、类别迁移、三集合划分、校验、快照、恢复 | [DATA_WORKFLOWS.md](DATA_WORKFLOWS.md) |
| jobs / worker | JobManager.start/poll_events/stop/shutdown、持久事件、进程归属、资源冲突 | [JOBS.md](JOBS.md) |
| training / training_worker | TrainingConfig、架构与参数校验、训练、完整恢复、标准评估 | [JOBS.md](JOBS.md) |
| export_worker / models | 通用 PT/ONNX/NCNN、CQ、AScript 包及结构/执行检查 | [JOBS.md](JOBS.md) |
| inference / worker_inference | CQ、NCNN、PT、ONNX，原图 xyxy 结果、基准和来源追踪 | [INFERENCE_CAPTURE.md](INFERENCE_CAPTURE.md) |
| capture / frames | WGC/DXGI、窗口状态、容量1最新帧队列、双槽共享内存 | [INFERENCE_CAPTURE.md](INFERENCE_CAPTURE.md) |
| desktop | PySide6 四页面、画布撤销栈、懒加载缩略图、异步数据任务、实时指标 | [USER_GUIDE.md](USER_GUIDE.md) |
| runtime | 便携路径、独立环境、PyInstaller 子进程 DLL 搜索路径隔离 | [ARCHITECTURE.md](ARCHITECTURE.md) |

数据、契约、参数错误使用 ValueError；项目锁冲突使用 RuntimeError；存储异常保留 OSError；取消使用 OperationCancelled。GUI 显示中文错误并记录完整日志。子进程原生崩溃产生 failed/interrupted 记录，不在 GUI 里重试加载原生库。

协议、项目、模型清单 schema 为 v1，SQLite user_version=1；遇到更高版本拒绝写入。stdout 专用 JSONL：protocol_version、job_id、sequence、timestamp、type、data。序列化失败不消耗序号，后续 error 仍可解析。第三方 Python/原生日志进入任务 stderr.log。

模型类别由该模型的清单、权重或 ONNX 元数据确定。当前项目类别不能覆盖历史模型含义。CQ 固定 IoU=0.45；其余控件按 Worker capabilities 决定，YOLO26 NCNN 与 PT/ONNX 的 NMS 能力不同。
