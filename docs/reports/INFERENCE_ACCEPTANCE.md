# 推理稳定性与 AScript 设备检查

记录时间：2026-10-04 13:49:51 UTC。

## 30 分钟实际运行

[`inference-soak.json`](inference-soak.json) 是 `.artifacts/inference-soak/report.json` 完成产物的逐字节副本，SHA-256 为 `f18e8e11484013044608c6fdedaffbaa7adbe6c19dcb105d205d5573bdf840e9`。来源文件完成时间为 2026-10-04 13:40:48 UTC。所有预设检查均为 true，最终状态 `passed`。

| 观测 | 实际结果 |
|---|---:|
| 实际持续时间 | 1,800.074588 秒 |
| 已推理并校验共享帧 | 38,314 |
| 最新帧队列主动丢弃过期帧 | 192 |
| 无帧等待超时计数 | 0 |
| 实际吞吐量 | 21.2851 帧/秒 |
| 平均适配器耗时 | 17.9840 毫秒 |
| 平均读源至显示副本耗时 | 19.1661 毫秒 |
| RSS 起点 / 峰值 / 结束 | 114.76 / 118.80 / 32.76 MiB |
| 句柄起点 / 峰值 / 结束 | 281 / 287 / 282 |
| 观测到的共享内存映射数 | 1（允许上限 2） |

覆盖范围为 CQ_AI 0.14.6、CPU、FP32、单会话、320 输入，以及自有生成图像流、容量 1 最新帧队列、双槽共享内存写入与复制校验。RSS 是 Windows 工作集观测，随系统内存管理变化；本结果通过既定增长阈值，不是分配器级无泄漏证明。192 个过期帧由有界队列策略主动丢弃，不表示无效画面被保存。`rss_current_bytes` 保留最后一次周期采样，完成值以 `rss_final_bytes` 为准。

本次运行不覆盖 WGC 连续 30 分钟、Android AScript 使用端、NVIDIA/CUDA/TensorRT 或业务准确率；独立 WGC 自有窗口检查的结果见 `../INFERENCE_CAPTURE.md`。

## AScript 只读检查

已读取 AScript 1.7.0 技能，确认本次会话有 `scan_devices` 和 `get_device_status` 工具。当前工作树没有保存 AScript 目标的 `.vscode/settings.json`。

- `scan_devices({port:9096})` 实际返回 `ModuleNotFoundError: No module named 'psutil'`，未得到设备列表。
- `get_device_status({})` 实际返回 `RuntimeError: 尚未连接设备。请先使用 connect_device 工具连接设备。`
- 结论：设备可用数量未知，扫描被插件运行环境缺少依赖阻塞，当前 AScript 连接尚未建立；Android 目标端验收保持待验。

本轮未安装依赖、选择或连接设备、读取屏幕、上传文件、运行脚本或操作其他项目。没有将扫描错误表述为零设备，也没有将桌面 CPU 验证表述为目标端验收。
