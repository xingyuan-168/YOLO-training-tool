# Database

## Schema

`project.json` 为项目事实：schema_version、UUID、name、classes、class_revision、created_at，以及校验后的项目 settings（training/model/legacy）。训练高级控件和原始专家 YAML 分开保存，避免重新打开后相互覆盖。

`records/<asset_id>.json`：内容哈希、原始文件相对路径、来源引用（仅审计，不作为读取依赖）、原名称、宽高、采集会话、审核状态、deleted、revision。metadata 保存截图来源或问题样本对应的模型和阈值。`labels/<asset_id>.txt` 为当前 YOLO 标签。`labels.txt` 为 UTF-8 类别。

`assets/<sha256>.<extension>`：导入时复制而不是链接外部源文件，入库后不原位覆盖。

`index.sqlite3` 是派生索引，assets 表：id 主键、hash 唯一、name、status、width、height、session、deleted；支持分页查询。大图不存数据库。

`splits/default.json` 保存 seed、分组策略和固定清单。`snapshots/<id>` 保存独立标签、图片链接/副本、data.yaml、snapshot.json；元数据含标签哈希、图片哈希、类别、参数和审核状态。

`history/classes-<id>.json` 保存类别迁移前后文本。撤回前检查迁移后的文件未再编辑，避免覆盖新工作。

`jobs/<job_id>/job.json` 保存任务索引与最新状态，`request.json/events.jsonl/stderr.log/result.json` 保存输入、事件、第三方输出和结果。首版任务索引从这些文件读取，不另外维护 SQLite 任务表，避免两套事实源。完整 checkpoint 位于任务训练目录中；导出到每个任务独立目录并通过 manifest 声明可用性。

## 迁移

项目 schema v1，SQLite user_version=1；新库建表，版本高于 1 拒绝打开。后续迁移必须显式实现和测试，不通过重新建空表吞掉事实。

## 恢复

操作系统文件锁 `.writer.lock` 限制单写者，进程异常退出后锁自动释放。

`.transaction.json` 记录提交前/后文本和 committed 标记。中断在标记前回滚，在标记后补写最终值；文件通过同目录临时文件、fsync、os.replace 写入。项目打开时先恢复事务，再重建 SQLite 索引。

首次打开和导入在后台线程建立索引；返回界面时通过 `index_prepared=True` 复用刚检查的数据库。同一连接只在所属线程使用；有恢复标记、数据库损坏或不兼容 schema 时仍执行完整检查。分页模型每页 100 条、最多缓存 12 页和 200 张缩略图。

快照先写入 `<id>.building`，成功后整体重命名。失败只清理此次创建且检查过路径范围的构建目录。工作标签改变不影响快照。
