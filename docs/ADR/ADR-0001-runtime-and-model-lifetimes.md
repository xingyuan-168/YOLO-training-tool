# 独立运行环境与原生模型生命周期

2026-10-04，已采用。

GUI、训练、推理分别使用项目内环境；GUI 不包含 Torch、CQ_AI 或 NCNN。训练固定 Ultralytics 8.4.7 / Torch 2.9.1+cpu，原生推理固定 CQ_AI 0.14.6。uv.lock 保留解析结果；CQ Wheel 另用 SHA256 校验。后续 NVIDIA 支持单独锁定和验收。

CQ_AI 的 Engine.close 释放进程级资源，不能在同一进程中随意关闭另一个 Engine 并期望之前的句柄继续有效。默认一个进程一个 Engine、一模型一会话；直接回归与适配回归使用顺序生命周期。

NCNN 的 Mat 可引用 NumPy 数组内存。探针把临时 NumPy 数组直接传入 input 后销毁，再 extract 曾导致访问冲突。保留数组和 Mat 到 extract 完成，两套输入样板均通过。普通样板输出是 [40,40,70]/[20,20,70]/[10,10,70]，仍不能宣称符合 AScript v8 解码后的单输出契约。

这些结论来自本项目独立 CPU 进程实测，不替代 DirectML、NVIDIA 或 Android 真机验证。
