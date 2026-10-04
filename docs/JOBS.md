# 任务、训练和导出

生产接口由 `jobs.py`、`worker.py`、`training_worker.py` 和 `export_worker.py` 提供。GUI 环境不导入 PyTorch、ONNX Runtime 或 NCNN。运行环境必须事先显式准备，任务不会安装依赖或下载模型。

## JobManager

```python
manager = JobManager(project_root, {"train": train_python, "inference": inference_python}, app_root)
job = manager.start("train", parameters, runtime="train")
events = manager.poll_events()  # GUI 定时器调用；所有返回值是普通字典
manager.stop(job.id)           # 下一安全边界停止
manager.stop(job.id, force=True)
manager.shutdown()
```

`jobs` 是按 ID 索引的 Job 字典，包含已恢复的历史任务；`active_jobs()` 返回仍在执行或清理的任务。Job 公共字段：`id / kind / state / run_dir / parameters / runtime / result / error / pid / creation_identity / returncode`。`run_dir` 为 Path；其余字段可 JSON 序列化。

状态：preparing → running → succeeded/failed；停止时 stopping → stopped；强制结束或监督器崩溃恢复时 interrupted。一次最多一个训练；同设备的 train/evaluate/export/infer/infer_stream/benchmark 互斥，capture/capture_stream 不占计算设备。标注与数据服务保持独立。

开发布局的源码路径是 `app_root/src`，便携布局是 `app_root/app`；Python 路径由调用者提供。训练参数引用本地 PT 或受支持的 `yolov8[n/s/m/l/x].yaml`、`yolo11*.yaml`、`yolo26*.yaml` 架构。无路径的 PT 名会先查 `app_root/models`，缺失时明确失败。

可分发字体从 `app_root/fonts/DejaVuSans.ttf` 与 `app_root/fonts/NotoSansSC.ttf` 读取，分别通过 YOLO_WORKBENCH_FONT / YOLO_WORKBENCH_FONT_CJK 传给 Worker；不下载字体，也不把系统 Arial 复制进包。

每个任务保存 `request.json / job.json / events.jsonl / stderr.log / result.json`。重开监督器会重放完整事件，忽略崩溃留下的截断尾行，并把未完成任务标记 interrupted。损坏的任务文件保留供诊断。恢复不会根据旧 PID 终止进程。

Windows 使用 Job Object 的 KILL_ON_JOB_CLOSE；Worker 启动后必须等 launch.flag，确保分配到所属 Job 后才可导入原生库、产生后代进程。监督器只通过自己持有的 Job/进程句柄终止任务，并记录进程创建时间用于审计。不会按进程名、旧 PID、用户全部 Python 进程进行清理。关闭管理器先请求停止，短暂等待后终止尚未结束的所属树；保留最近完整检查点。

## Worker 协议 v1

调用：`python -m yolo_workbench.worker --request FILE`。请求包含 `protocol_version=1, job_id, kind, parameters, run_dir`。

stdout 专用于 `EventWriter` JSONL；Python stdout、CRT fd1 和 Windows STD_OUTPUT_HANDLE 都被重定向到 stderr，防止原生库日志污染协议。每个事件包含 `protocol_version / job_id / sequence / timestamp / type / data`，序号从 1 连续递增。协议损坏或原生崩溃会生成明确 failed 记录。

| type | data |
|---|---|
| state | state、可选 reason |
| config | effective、family、snapshot、classes、mode=train/finetune/resume |
| progress | epoch/epochs、batch/batches、losses、elapsed_seconds、eta_seconds、resources；导出包含 phase |
| checkpoint | epoch、resume_checkpoint、best、last、full_state |
| metrics | epoch/epochs、losses、metrics、learning_rates、resources |
| resume | checkpoint、start_epoch、optimizer_restored、scheduler_restored、rng_restored、model_restored |
| result | 与 `job.result` 和 result.json 相同的结果字典 |
| error | code、message、log |

损失键使用 Ultralytics 实际 `label_loss_items` 输出，不把所有系列硬编码为三个 loss。推理和采集由 `worker_inference.handle(request, emit)` 延迟导入，事件可扩展为 frame/source_status；外层 Worker 统一发送 running、result 和终态。停止信号是 run_dir/stop.flag。

## 训练与完整恢复

```python
parameters = {
    "model": "D:/models/yolov8n.pt",
    "snapshot": "D:/project/snapshots/ID",
    "config": {"family": "yolov8", "scale": "n", "imgsz": 640,
               "epochs": 100, "batch": 4, "device": "cpu", "workers": 0, "patience": 100},
    "expert_yaml": "optimizer: SGD\namp: false\n",
    "finetune": False,
}
```

TrainingConfig.effective 校验基础字段和专家白名单。快照必须包含 snapshot.json、data.yaml、非空 train/val、冻结类别和图像/标签哈希。每次训练/评估读取并验证快照，再为任务创建私有图像链接、独立标签及 data.yaml；Ultralytics cache 不写入原始快照。

加载 PT/YAML 后，Worker 从模型内部的 `model.yaml.yaml_file` 验证实际架构系列；与配置不同或架构元数据无法识别时，在训练前拒绝执行。文件改名和界面选择不能把 YOLO11/26 权重标记为 YOLOv8，避免产生错误的训练 manifest 与恢复契约。

每轮 `on_model_save` 在 Ultralytics strip_optimizer 前原子生成 `train/weights/resume.pt`。该检查点保存未平均的原始训练权重、FP32 EMA、完整精度优化器、scaler、scheduler、早停状态、Python/NumPy/Torch/CUDA RNG、DataLoader generator、梯度。best.pt/last.pt 仍遵循 Ultralytics 推理权重输出习惯。

安全停止保留本轮完整 checkpoint，再完成最后验证和清理；强停仅保证此前已原子保存的轮次。`resume_checkpoint` 与 `finetune=True` 互斥。完整恢复必须选本工具 resume.pt、同一快照和类别、同一系列，并继续原来的总轮数计划；完成计划后增加训练或更换数据应使用 finetune。普通 best.pt、已 strip 的 last.pt 不冒充完整恢复。

恢复创建新的任务目录，不覆盖历史 CSV、权重和配置。恢复事件报告真实起始轮数及状态装载；实际 CPU 冒烟验证 epoch1 停止后继续到原计划 epoch3。保存恢复状态不承诺 CUDA 多线程数据增强、浮点或跨版本训练逐位相同；首版固定 Ultralytics 8.4.7 并且仅单设备。

训练结果：`best / last / resume_checkpoint / epochs_completed / start_epoch / resumed / metrics / results_csv / plots / classes / stopped`。权重旁 `.manifest.json` 保存冻结训练类别、系列、快照哈希及权重哈希，后续类别重排不改变历史模型。

## 标准评估

参数：`model, snapshot, imgsz=640, batch=4, device="cpu", workers=0, split="val"`。split 可选显式 test；缺少该划分拒绝执行。模型类别顺序必须与冻结快照相同。结果标记 `source=ultralytics_standard_evaluation`，包含标准 metrics、speed、plots，与部署后端预测结果分开。

## 导出与验收

参数：`model, format=pt|onnx|ncnn, profile=generic|cq|ascript_v8, family, imgsz, device="cpu", output_dir`。output_dir 是包的父目录，默认 run_dir/exports；每次生成时间和随机后缀的唯一目录，不覆盖来源权重或旧包。input/ 不接受为导出目的地。

| profile | 格式与约束 |
|---|---|
| generic | PT/ONNX/NCNN；v8/11/26；固定尺寸、Batch1、FP32 |
| cq | ONNX；v8/11；320 或 640；opset12；无 NMS；[1,4+C,N] |
| ascript_v8 | NCNN；v8；640；无 NMS；单输出 [4+C,8400] |

YOLO26 的 ONNX/PT 通用包保留端到端 `[1,N,6]` 输出，manifest 标记 `output_layout=xyxy_score_class, nms_required=false`。Ultralytics 8.4.7 的 NCNN 导出不支持 topk，会关闭端到端分支，实际为 `[4+C,N]`；包标记 `xywh_class_scores, nms_required=true`。不能根据 YOLO26 系列名称直接跳过 NCNN 的 NMS。实际 CPU 跨系列测试通过 YOLO11 CQ320 `[1,84,2100]`、YOLO26 ONNX320 `[1,300,6]`、YOLO26 NCNN320 `[84,2100]`。

包包含模型、manifest.json、labels.txt、classes.txt、example.py、README.txt、validation.json 和验证日志。AScript 包额外包含可作为 Android 项目入口的 `__init__.py`，使用官方 Yolov8Ncnn:1.3 的 load/detect/free API（设备先显式准备插件，模型保持与入口同目录）。manifest 使用 ModelManifest 字段，附 profile、opset、model_file、文件哈希、验证证据；NCNN 包为 model.ncnn.param / model.ncnn.bin。

Windows 的 TorchScript/PNNX 原生路径接口对中文绝对路径不可靠。Worker 在本次唯一包目录内转换，给转换器传 ASCII 相对文件名；Python 仍保留用户选择的中文目标目录。此路径已由实际 NCNN 转换与独立执行测试覆盖。

结构检查之后，独立 Python 子进程使用 PyTorch / ONNX Runtime / NCNN CPU 执行确定性图案输入，检查输出布局和有限数值。只有结构与执行都通过才写成功 manifest。任何失败保留 failure.json 和日志，任务为 failed。PNNX/NCNN/ONNX 等工具链缺失时明确指出，禁止任务内安装。AScript Android 真机、CQ_AI 实际使用端和业务准确率仍须单独验收；CPU 结构执行通过不替代这些验收。

## 验证入口

- `tests/test_jobs.py`：协议、持久化/重开、同设备冲突、协作停止、强停所属 Windows 子进程树、原生异常及 stdout 隔离。
- `tests/test_training_jobs.py`：参数拒绝、快照完整性和只读、NCNN 多头拒绝；显式设置 `YOLO_RUN_TRAINING_INTEGRATION=1` 与 `YOLO_TRAIN_PYTHON` 后运行实际 CPU train/stop/resume/evaluate/PT/ONNX/NCNN 链路。
- `tests/test_export_jobs_integration.py`：同样显式启用后验证 YOLO11/26 原生导出和实际输出布局；架构随机初始化仅验证转换/执行链路，不表示准确率。

## FP32 跨格式数值验收

`scripts/validate_export_parity.py` 使用已显式准备的真实预训练 Nano 权重，通过生产 JobManager 重新导出模型，再在独立运行环境比较。默认读取 Ultralytics 随包 `bus.jpg`、`zidane.jpg`，加一张确定性的合成渐变图。原始权重、导出文件、图片和相同输入 Tensor 的 SHA256 全部写入报告；禁止下载模型。

同一比较共享 RGB、INTER_LINEAR letterbox、114 padding、NCHW FP32 / 255 和 640 输入。v8/11 共享 confidence=0.25、按类别 NMS IoU=0.45；YOLO26 共享端到端最终行解码，不另加 NMS。按类别以最大 IoU 做一对一匹配，要求框数/类别完全相同、每框 IoU ≥ 0.99、置信度差 ≤ 0.001。两边都是空检测标记 `vacuous_no_detections`，不算通过证据；每组至少需要一张真实图片的非空检测通过。

2026-10-04 CPU 验收通过，完整哈希、版本与逐图结果见 `docs/reports/export-parity.json`：

| 比较（640 / FP32） | 非空图片 | 最低匹配 IoU | 最大置信度差 |
|---|---:|---:|---:|
| v8n PT → CQ ONNX | 2 | 0.999999227 | 0.000000924 |
| 11n PT → CQ ONNX | 2 | 0.999999125 | 0.000000596 |
| 26n PT → 通用 ONNX | 2 | 0.999998789 | 0.000000656 |
| v8n PT → AScript NCNN | 2 | 0.999998610 | 0.000000715 |

每组照片分别包含 5 与 3 个匹配检测；四个合成图比较均为空，单独记录且排除通过计数。结果证明这些模型的 CPU 导出数值一致性，不代表每个用户模型、业务准确率或 Android 真机验收。

在已准备的项目运行 `python scripts/validate_export_parity.py`；可传 `--train-python / --models / --images / --artifacts / --report`，便携包自动识别 runtime/train/python.exe。`--self-test` 在训练环境执行七个验收规则检查，覆盖类别不匹配、框差异、置信度差异、空结果和顺序无关性。
