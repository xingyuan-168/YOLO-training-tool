# 数据管理与导入接口

本文描述 `dataset.py`、`importers.py`、`labels.py`、`storage.py` 的生产数据接口。GUI 只通过服务访问事实文件；SQLite 属于派生索引，图片读取使用项目内相对路径。

## 导入与配置

```python
from pathlib import Path
from yolo_workbench.dataset import DatasetService
from yolo_workbench.importers import inspect_dataset, import_dataset

preview = inspect_dataset(Path("D:/datasets/example.zip"))
with DatasetService(Path("D:/projects/example")) as service:
    result = import_dataset(service, Path(preview["source"]),
                            trust_empty_labels=False,
                            progress=lambda event: print(event),
                            cancel=lambda: False)
```

`inspect_dataset(source, *, progress=None, cancel=None)` 只扫描路径并读取有界元数据，不解码图片。结果包含 `source`、`kind`（directory/file/zip/unknown）、`classes`（列表或 None）、`image_count`、`label_count`、`config`、`errors`、`warnings`。取消时返回 `cancelled=True`。预览不能代替逐图完整解码校验。

`import_dataset(service, source, trust_empty_labels=False, progress=None, cancel=None)` 支持图片目录、单张图片、标准 YOLO 目录、YAML/配置文件所在目录和 ZIP。结果包含：

| 字段 | 含义 |
|---|---|
| total / imported / duplicates / failed / skipped | 候选图片总量、新入库、重复内容、逐图失败、尚未处理或前置检查阻止的数量 |
| pending / labeled / empty | 本次新入库图片的未审核／已标注／确认无目标数量 |
| cancelled | 用户在处理单元之间取消；已提交样本仍然有效 |
| errors / warnings | `{path, code, message}` 结构化问题；调用者应展示错误，不能只检查 imported |
| config | `{training: {}, model: {}, legacy: {}}` 校验后的配置预览 |
| split | `{train: [id], val: [id], test: [id]}` 来源划分映射 |

支持 `images/train/a.png ↔ labels/train/a.txt`、`train/images/a.png ↔ train/labels/a.txt` 及同目录图片/TXT。多个候选标签视为错误。YAML `train/val/test` 可为图片路径、目录、路径列表或图片清单 TXT；清单路径只能落在所选来源范围内。存在显式清单时只导入清单成员。旧绝对 `path` 在导入目录内重新定位并报告警告，不读取旧机器上的外部路径。YAML `download` 永不执行。

类别来自浅层 `labels.txt`、`classes.txt`、`.names`、YAML `names`；字典编号必须从 0 连续递增，`nc` 必须匹配，多份类别定义必须一致。空项目可采用来源类别；已有项目按名称映射编号，未知类别要求先新增，绝不默默改变已有编号。重复 SHA-256 内容保留现有标签、审核和回收状态。缺失标签始终 pending；空 TXT 默认 pending，只有显式 `trust_empty_labels=True` 才视为确认负样本。

图片逐一复制、完整解码并事务提交，原图和来源目录保持不变，不硬链接外部原图。ZIP 原始审计来源保存为 `archive.zip!/relative/image.png`，不会记录即将删除的临时文件路径。每图失败可继续，源元数据/类别失败在入库前终止。空项目且 train/val 有效时保存来源划分；未审核、重复内容或其他泄漏会阻止保存并报告警告。

ZIP 在展开前校验全部路径：拒绝绝对路径、磁盘/UNC/ADS、`..`、Windows 保留名、符号链接/特殊文件、大小写碰撞、文件/目录冲突和加密条目。仅接受 stored/deflate；上限为 100,000 条目、单文件 1 GiB、总展开 20 GiB、压缩比 200。元数据上限 4 MiB，单标签 16 MiB。每次仅展开正在导入的图片到独立临时目录，校验实际展开长度；成功、失败和取消都会检查绝对边界后清理自己的暂存目录。图片解码同时采用 Pillow 像素数量限制。超限来源需由用户在工具外拆分整理。

`normalize_training_config(raw, *, existing=None)` / `normalize_model_config(raw)` 将旧字段转为规范类型，不注入未提供的默认值。例如样板的 320、3000 epochs、batch=-1、device="0"、workers=12 被原样保留。支持 `训练参数.json` / `模型配置.json` 及英文同义文件；原始字典存入 legacy，旧项目目录/模型路径只作审计引用，不执行命令、不加载模型。

`apply_imported_config(service, config)` 合并前再次校验；`import_dataset` 在未取消的导入结束时自动应用有效配置。`service.update_settings({training, model, legacy})` 按部分合并已校验值，未提供的已有值保留；`get_settings()` 返回深复制。GUI 若自行编辑配置，先通过相应 normalize 函数校验。配置保存在可选的 `project.json.settings`，project schema 仍为 1。

## 浏览、标注与类别

| 接口 | 契约 |
|---|---|
| `DatasetService.create(root, name, classes)` / `DatasetService(root)` | 新建/打开并取得独占写锁；服务及其 SQLite 连接只能由所属线程使用 |
| `get_asset(id)` / `get_path(id)` | 前者返回完整记录，后者返回托管图片绝对 Path；原始 source 不参与读取 |
| `count_assets(status=None, search='', deleted=False)` | 查询数量，不解码图片 |
| `list_assets(*, status=None, search='', limit=100, offset=0, deleted=False)` | 保留分页接口，按 name/id 排序；limit 1..1000；行含 id/hash/name/status/width/height/session/deleted |
| `iter_asset_pages(..., page_size=200)` / `iter_assets(...)` | 流式分页／逐记录迭代；避免批量图片解码和重复 OFFSET 扫描 |
| `next_unreviewed(current_id=None, *, search='', wrap=True)` | 当前排序下下一个 pending 行，无结果返回 None，默认环回 |
| `import_image(source, *, labels=None, confirmed_empty=False, session=None, source_reference=None, metadata=None)` | 返回 `(asset_id, created)`，附加采集会话和 JSON 来源元数据 |
| `load_boxes(id)` / `save_boxes(id, boxes, *, confirmed_empty=False)` | YOLO 检测框；标签和审核记录原子更新 |
| `update_status(id, status)` | 审核状态须与标签一致；置 pending 保留现有框；empty 是显式负样本确认 |
| `recycle(id, *, restore=False)` | 逻辑回收/恢复，保留图片及快照引用 |

`preview_class_migration(names, mapping, *, progress=None, cancel=None)` 必须覆盖每个旧编号，值为新编号或 None；返回 `classes/mapping/dropped/affected_images/total_boxes/class_counts/requires_confirmation`。预览不写入；范围包含回收区标签。

`migrate_classes(names, mapping, *, allow_drop=False, progress=None, cancel=None)` 复用预览校验，返回上述字段及 `migration_id`。删除现有框须显式 `allow_drop=True`；某图最后一个目标被删除后回到 pending，不自动变成负样本。新增和重命名仍使用同一受控映射接口。`restore_class_migration(migration_id)` 仅在迁移后全部记录、标签和相关项目元数据保持原状态时恢复；新导入、回收、编辑或外部变化都会阻止覆盖。历史快照类别映射独立保存。

## 检查、划分与快照

`statistics(*, progress=None, cancel=None)` 返回 total、deleted、statuses、boxes、sizes（`"WxH": count`）、classes（每类 id/name/boxes/images）和读取问题 issues。只读标签与尺寸记录，不解码图片。

`validate(*, split=None, progress=None, cancel=None)` 返回 `{id, severity, code, message}` 列表：完整像素解码、外部内容变化、尺寸不一致、缺失/非法/孤立标签、未审核、重复框、低于 32 像素或宽高比超过 10 的图片、边长不足 2 像素的框，以及划分重复图片和跨会话泄漏。不传 split 时检查现有 default.json；传入 `{}` 也会按无效划分报告。重复导入图片在导入结果中报告；唯一哈希索引避免重复入库。

`split(*, seed=42, train_ratio=.8, test_ratio=0., grouped=True, exclude_pending=False, progress=None, cancel=None)` 的验证比例为 `1 - train_ratio - test_ratio`。默认 80/20；测试比例非零时至少需三个独立组。分组时同一 session 的图不跨集合，无会话按单图分组；比例按组数分配，因此图片数量可能偏离目标比例。pending 必须审核或显式排除。返回并保存 seed/grouped/train_ratio/test_ratio/train/val/test。

`validate_split(split)` 对清单做只读检查。`snapshot(split, parameters, *, progress=None, cancel=None)` 仅接受非空 train/val、无重复、无会话泄漏且已审核的图片；参数必须可序列化为有限 JSON 值。快照冻结图片哈希、标签哈希、独立标签、类别、划分和参数；图片仅对项目内不可变资产使用硬链接，失败则复制。非空测试集写入 `data.yaml.test`。构建先进入随机 ID 的 `.building` 目录，完成后整体重命名；取消/失败只清理当前构建。返回最终快照 Path。

所有长操作的 `progress` 接收 `{phase, completed, total, path}`，`cancel` 可为无参数布尔回调或带 `is_set()` 的 Event。服务长操作取消时抛 `storage.OperationCancelled`，导入器转换成结果的 cancelled 字段。逐图解码是最小取消单位；GUI 应将扫描、解码、迁移、检查、划分与快照放入工作线程，在该线程内打开服务并保持单线程连接所有权。

## 恢复与迁移

JSON/TXT 是事实，索引通过 `rebuild_index(*, progress=None, cancel=None)` 事务重建，返回记录数量；中途取消保留之前索引，冲突哈希/坏记录不会默默丢掉。打开时损坏的 SQLite 派生文件会重建，未来版本索引拒绝写入。新增索引仅优化分页，assets 表及 user_version 仍为 1。

`.transaction.json` 按提交标记回滚/补写标签和记录，恢复前先校验所有目标路径。`.asset-import.json` 记录刚发布的资产：重开时保留已提交记录，清除未提交且哈希仍匹配的独占资产；若外部已修改则拒绝删除。已完成事务不依赖索引成功与否，下次重建修复派生状态。打开后仅清理符合工具 UUID 命名规则的残留 `.importing` 和 `.building`；用户命名目录保持不动。

项目可整体搬迁。asset.file 使用相对路径；原始 source、旧配置路径为审计信息，删除原始来源也不影响标注、检查和生成新快照。已有快照自己的 data.yaml 使用相对图片路径；上层训练运行器负责将快照根作为数据根。

## 验证范围

`tests/test_dataset.py` 与 `tests/test_importers.py` 覆盖路径攻击、链接/容量 ZIP、元数据冲突、跨集合泄漏、测试集 YAML、内容去重、源文件独立性、取消、进程中断前后恢复、迁移撤回保护、坏像素、坏索引、项目目录搬迁、类别分布与原配置值保留。NVIDIA 训练、真实设备推理和导出不属于这些数据测试的结论。
