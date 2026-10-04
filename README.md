# YOLO 本地工作台

面向 Windows 的个人 YOLO 标注、训练与部署验证工作台，采用 Python 3.12、PySide6、Ultralytics 和 CQ_AI 0.14.6。

当前处于 M0 工程基线与 M1 交互原型阶段，尚未交付生产 GUI。前端在用户批准具体原型后实现。详细范围与阶段见 docs/SCOPE.md。

- input/：用户原始资料，只读。
- output/：最终交付物，只放成品。
- docs/：需求、范围、架构、开源调研与决策记录。

输入材料和模型不纳入 Git；运行环境、下载缓存和临时测试产物也不纳入 Git。

开发分支：feat/yolo-workbench。治理使用已安装的 AI Engineering OS `aios` CLI，实际检查结果保存到阶段报告。
