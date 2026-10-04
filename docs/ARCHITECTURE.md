# Architecture

Python 3.12 x64、PySide6 Widgets。桌面入口 `python -m yolo_workbench`，便携入口 `YOLOWorkbench.exe`。GUI 只加载控件和轻量服务；训练、转换、评估和部署推理位于独立任务进程。

| 层 | 实现 |
|---|---|
| domain | 标签、模型契约、训练参数、协议；与 Qt 无关 |
| services | DatasetService、YOLO/ZIP 导入、类别迁移、划分、不可变快照、事务恢复 |
| jobs | JSONL、取消、进程归属、任务恢复、设备占用 |
| workers | Ultralytics train/evaluate/export；CQ、NCNN、PT、ONNX；WGC/DXGI capture |
| desktop | 四页面、QGraphicsView 八手柄画布、QUndoStack、分页缩略图、PyQtGraph |
| runtime | 显式环境准备、固定 Wheel/hash、字体/权重缓存、PyInstaller onedir |

## 数据流与隔离

导入/截图 → 托管原始资产和来源 → 当前标签和审核状态 → 检查/划分 → 冻结快照 → 训练任务 → 模型清单 → 标准评估或部署验证 → 通过验证的导出包 → 问题样本回流。

GUI 线程拥有自己的 SQLite 连接。导入、校验、类别迁移和冻结数据先保存并释放服务；后台线程独立打开服务，完成后 GUI 轻量重开。首次索引重建在后台进行；`index_prepared=True` 仅用于已经准备好的有效索引，恢复事件会强制重建。缩略图最多2线程、200项缓存；SQL页缓存最多12页，每页100项。

当前 SQLite 索引图片；任务与产物使用 jobs/<job_id>/job.json、JSONL 和 manifest 事实文件恢复内存索引。图片/标签/快照独立存储；图片只对项目内不可变资产使用硬链接。来源目录不被链接或覆盖。同一项目限制一个写者。

Windows Job Object 只拥有本工具创建的进程树。启动门闩确保 Worker 归属确认后才加载原生库。AUTO 尚未确定设备时保守阻止其他计算密集任务；采集与标注保持可用。JSONL stdout 只传协议，第三方日志转 stderr。连续画面采用容量1最新帧队列、双槽共享内存，最多保留两个尺寸映射。

数据后台操作期间的截图入库路径排队，重开后补入库。问题样本保存干净原图及执行时模型/阈值/来源；预测不自动成为人工真值。完整恢复检查点在权重 strip 之前保存，恢复创建新任务目录。

## 模型契约

CQ 默认 AUTO/单会话/FP32，展示真实设备；保持最近邻 letterbox、跨类别 NMS、IoU=0.45，不追加改变语义的后处理。AScript 使用独立 NCNN 适配。CQ v8/11 固定320/640、opset12、[1,4+C,N]；AScript v8 固定640、[4+C,8400]。YOLO26 走通用路径，NCNN 导出的 NMS 能力依据实际计算图确定。

导出新目录后执行结构与独立运行检查，成功才生成可用 manifest。具体契约见 JOBS.md 与 INFERENCE_CAPTURE.md；数值回归报告不混同不同部署预处理算法。

## 便携布局

目录包包含 GUI 的 _internal、runtime/train、runtime/inference、app/yolo_workbench、models、fonts、docs、source 和许可证。两个 Worker 环境包含完整 Python 基础解释器、标准库、各自依赖；移除指向开发目录的 editable .pth，不依赖开发机 venv。

启动 Worker 时清理 PYTHONHOME/user-site，并临时恢复标准 Windows DLL 搜索路径，避免继承 PyInstaller GUI 的 DLL 目录。包带 DejaVu 与 OFL Noto 字体，离线绘图不下载字体。依赖只由显式准备命令安装。CPU 锁文件固定运行组合；CUDA、TensorRT、Android 使用端分别验收，基础包不带整套 NVIDIA 运行库。

