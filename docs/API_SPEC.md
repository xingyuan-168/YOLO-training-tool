# API Spec

## 接口

当前已实现（不依赖 Qt）：

- `DatasetService.create/open`：取得独占写锁，恢复中断的文件事务，重建派生索引。
- `import_image`：解码检查、内容去重、复制原始资产，显式传入标签或负样本确认。
- `list_assets(status, search, limit, offset)`：分页索引查询，不解码图片。
- `load_boxes/save_boxes`：标准 YOLO TXT，空标签默认未审核；标签与记录事务提交。
- `recycle/restore`：逻辑回收，原始资产仍可供历史快照引用。
- `migrate_classes/restore_class_migration`：完整编号映射、删除确认、事务记录；拒绝覆盖迁移后的新修改。
- `split`：固定 seed，按会话或图片划分 train/val，未审核必须明确排除。
- `snapshot`：冻结图片、标签、类别、划分和参数；原始图片硬链接失败则复制。
- `validate`：损坏/变更图片、非法标签、审核状态、重复框与孤立标签。
- `TrainingConfig.effective`：校验设备、轮数、输入尺寸及专家白名单；禁止覆盖应用管理字段。
- `ModelManifest / inspect_onnx / validate_contract`：输入输出结构、类别、精度、系列、opset 契约。
- `CqBackend`：配套 Wheel 显式 DLL、单会话加载/识别/状态/释放。仅在推理环境的独立进程中使用。
- `EventWriter / EventReader`：协议 v1、任务 ID、递增序号、UTC 时间、类型和数据。

待后续阶段实现：JobManager、生产 TrainerBackend、NCNN 预测适配器、CaptureBackend、共享帧缓冲、ZIP/完整数据集导入、显式测试集、导出管线、生产 GUI。M0 的脚本探针不是这些生产服务的替代品。

## 错误

数据/契约/参数错误抛出 ValueError，项目锁冲突抛出 RuntimeError，存储异常保留 OSError。UI 层在后续实现中转换为中文错误与恢复操作。原生进程异常由 supervisor 识别，不能在 GUI 进程试加载。

JSONL stdout 只传事件；第三方日志进入 stderr。当前只实现编解码，生产 Worker 和进程状态机尚未交付。

## 兼容

协议 schema v1、项目 schema v1、索引 user_version=1。遇到更高版本拒绝写入。CQ_AI=0.14.6，IoU=0.45；AScript v8 需单输出 [4+C,8400]；YOLO26 走通用模型路径。
