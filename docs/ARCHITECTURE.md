# Architecture

## 定位

Python 3.12 x64。GUI 只负责控件和轻量服务，不加载 PyTorch 或原生推理 DLL。训练/评估/导出、CQ_AI、NCNN 分别使用任务子进程与独立环境。

## 组件

以下为完整目标架构。当前批次实现数据服务、参数/模型契约、CQ 适配和事件编解码；生产 Worker/JobManager、采集、导出管线及桌面界面仍待后续阶段。实际已实现边界见 API_SPEC.md。

| 层 | 责任 |
|---|---|
| domain | 标签、项目、任务事件、模型契约、配置校验；与 Qt 无关 |
| services | DatasetService、类别迁移、划分、快照、原子写入、索引、回收 |
| jobs | JobManager、JSONL、取消、拥有的子进程、异常恢复和资源占用 |
| workers | train/evaluate/export、CQ_AI/NCNN/generic、capture，第三方日志转 stderr |
| desktop | PySide6 Widgets，QGraphicsView，虚拟列表，监控和设置（批准后实施） |
| runtime | 环境准备/校验、固定 Wheel 与哈希、离线缓存、打包 |

项目事实文件与 SQLite 索引分离；标签 UTF-8 YOLO TXT；图片入库后不可变；快照标签独立复制、图片可硬链接。同一项目只允许一个写者。

CQ_AI 默认 AUTO / 单会话 / FP32，UI 展示真实后端。保持最近邻 letterbox 和跨类别 NMS；不默认追加改变语义的 Python NMS。IoU 固定 0.45。运行时所有资源路径解析为绝对路径。

## 数据流

导入/截图 → 原始资产 + 来源 → 当前标签/审核状态 → 检查/划分 → 不可变快照 → 独立训练任务 → 模型 manifest → 标准评估或部署适配 → 验证后的导出包 → 困难样本回流。

控制采用带协议版本/任务 ID/序号/UTC 时间/类型/data 的 JSONL；stdout 专用于协议，stderr 存日志。连续帧使用有界共享内存，事件仅包含引用。只终止本工具拥有的进程树；任务冲突按设备和类型校验。

导出契约：CQ_AI v8/11 batch1、320/640、FP32、opset12、无 NMS、[1,4+C,N]；AScript v8 640、[4+C,8400]、param/bin/classes/example；YOLO26 通用路径。
