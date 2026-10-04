# Project Instructions（宪法）

1. 不改变 DSH 原生工程方式：AIOS 只治理"能否做、何时做、做完留什么"。
2. 正式编码前必须通过 Code Start Gate：GitHub remote 可达、开源调研已记录、仓库无副本式脏乱。
3. input/ 只读：不得修改、重命名或删除其中内容。
4. 禁止复制式版本管理：不创建 src_v2/、backup/、copy/ 等副本；历史由 Git 保存。
5. 受影响的项目文档必须随本次变更同步更新。
6. 前端实现前必须先有 docs/design/PROTOTYPE.html 与 docs/design/UI_SPEC.md，并获得用户批准。
7. 复杂并行任务使用 DSH 原生子 Agent + .worktrees/ 隔离，完成后清理。
8. 一次性文件（临时脚本、缓存、调试产物）任务结束删除，不进入 Git。
9. 有价值的决策、Bug 根因与可复用经验写入 docs/memory/memory.jsonl。
10. 危险 Git/删除操作必须保护用户资产：主工作区禁 force push、禁删远端 ref、禁递归强删。
