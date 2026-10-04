# Open Source Research

## Requirement

requirement_id: YOLO-V1
summary: Windows 本地 YOLO 标注、采集、训练、评估、部署验证和导出工作台。
scope:
  - desktop annotation and managed datasets
  - capture and isolated job workers
  - training, evaluation and export
  - CQ_AI and NCNN inference adapters
updated_at: 2026-10-04

## Candidates

### Ultralytics

- URL: https://github.com/ultralytics/ultralytics ; https://docs.ultralytics.com/usage/python/
- License: AGPL-3.0，保留许可证；进程隔离不改变许可证义务。
- 解决什么：YOLOv8 / YOLO11 / YOLO26 训练、标准评估、通用导出。
- 可直接复用：Python API、callback、官方转换链路。
- 可二开：应用适配层，不维护框架分叉。
- 值得学习：训练参数、检查点生命周期、导出元数据。
- 风险：训练结束移除优化器，须另存完整恢复检查点；固定版本且禁用运行时安装；YOLO26 不宣称兼容旧解码器。

### CQ_AI 0.14.6（用户现有工程）

- URL: https://github.com/xingyuan-168/CV_OCR
- License: 使用用户提供的 Wheel 和随包许可证，未假定公共开源授权。
- 解决什么：既有部署语义、CPU / DirectML / 可选 TensorRT。
- 可直接复用：Engine、yolo_model、infer、runtime_status、release 和配套 x64 DLL。
- 可二开：独立进程与能力适配层。
- 值得学习：Letterbox、跨类别 NMS、会话管理。
- 风险：固定 IoU=0.45；资源相对路径以可执行文件目录解析，应用传绝对路径；原生崩溃必须隔离。Python 封装绑定 OCR/CV 符号，因此首版复用完整 Wheel。

### PySide6 / Qt

- URL: https://doc.qt.io/qtforpython-6/ ; https://doc.qt.io/qtforpython-6/licenses.html
- License: LGPL-3.0 / GPL / commercial；选择公开 LGPL 组件，保留动态库与许可证。
- 解决什么：桌面控件、虚拟列表、QGraphicsView 标注。
- 可直接复用：Widgets、模型视图、信号槽。
- 可二开：应用控件。
- 值得学习：DPI、键盘操作、无障碍。
- 风险：GUI 线程不能加载模型或批量解码图片。

### Windows Capture

- URL: https://github.com/NiiightmareXD/windows-capture
- License: MIT。
- 解决什么：WGC 窗口采集、DXGI 桌面采集。
- 可直接复用：HWND 绑定、Frame、后台 capture control。
- 可二开：采集状态和有界队列。
- 值得学习：关闭回调及来源生命周期。
- 风险：DPI / 客户区需实机验证，不保证最小化持续采集。

### YOLO Annotator Desktop

- URL: https://github.com/sicaizhuang/yolo-annotator-desktop ; https://raw.githubusercontent.com/sicaizhuang/yolo-annotator-desktop/main/pyproject.toml
- License: MIT。
- 解决什么：标签读写、原子保存、类别迁移与检查参考。
- 可直接复用：审查后可独立的纯数据模块，提取时保留来源与版权。
- 可二开：Tkinter 整体不直接嵌入 Qt。
- 值得学习：安全写入及状态建模。
- 风险：不能带入隐式 Tk 状态；当前未复制第三方源码。

### NCNN / AScript v8

- URL: https://github.com/Tencent/ncnn ; https://docs.ultralytics.com/integrations/ncnn/ ; https://ascript.cn/docs/android/api/screen/yolo/v8/yolo/
- License: NCNN BSD-3-Clause；AScript 插件依其发布条款，不改包。
- 解决什么：param/bin 运行时、AScript v8 契约。
- 可直接复用：NCNN Python 运行时、官方导出器。
- 可二开：固定 640、单输出 [4+C,8400] 检查和示例。
- 值得学习：in0 / out0 和加载释放。
- 风险：旧多输出样板本机提取崩溃，必须独立进程试运行；零输入成功不等于 Android 真机验收。

### Datumaro / CVAT / Label Studio / Optuna / FiftyOne

- URL: https://github.com/open-edge-platform/datumaro ; https://github.com/cvat-ai/cvat ; https://github.com/HumanSignal/label-studio ; https://github.com/optuna/optuna ; https://github.com/voxel51/fiftyone
- License: 实际引入版本时分别核查；首版未引入代码或依赖。
- 解决什么：多格式、团队协作、调参与分析。
- 可直接复用：V2/V3 再评估。
- 可二开：首版不 fork 大型平台。
- 值得学习：数据集与实验工作流。
- 风险：依赖、维护与部署成本超出首版范围。

## Decision

decision: build
reason: 自建项目数据、任务协议和 Qt 工作流；use Ultralytics、PySide6、Windows Capture、CQ_AI Wheel、NCNN；selective extract 仅在独立模块审查后实施。不重造训练框架、不分叉完整标注平台。

输入只读，用户样本和 C++ 源码不自动上传 GitHub。CQ_AI Wheel SHA256：d47d071820eac8aa40c1b9c05fee247a2a94bc6c76c78476892f5e6a38e8e193。依赖在 M0 实测后锁定。
