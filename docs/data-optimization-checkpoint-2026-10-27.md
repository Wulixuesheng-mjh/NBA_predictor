# 数据优化暂停检查点（2026-10-27）

> 已于 2026-10-28 恢复工作；最新进度见 `data-optimization-checkpoint-2026-10-28.md`。下文保留为暂停时快照。

用户要求先保存进度，后续再执行未完成工作。本文件是最新恢复入口。

## 暂停状态

- 本轮 `nba-stat-backfill`、`nba-injury-pilot`、`nba-fetch-kaggle-v525` 均已结束，无本轮采集容器继续运行。
- backend、frontend、postgres 保持运行，原有应用调度未停止。
- 代码已保存在工作区，未提交 Git。
- 未执行新的特征缓存构建、E3 消融、模型重训或晋升。
- 活动模型仍按上一轮的 `20261022_050952`；候选 `20261027_213v7` 没有被本轮替换。
- **已采集到可信快照，但尚未接入特征读取。不能声称当前训练已使用这些新伤病信息。**

## 已完成的数据与工程修改

### 1. 数据库备份和迁移

- 补采前 PostgreSQL 自定义格式备份：`reports/data_quality_20261027/db_before_backfill.dump`。
- 已执行 `006_snapshot_prospective` 迁移，数据库为此 head。
- `team_game_stats` 新增 `(game_id, team_id)` 唯一约束，迁移前确认不存在重复键。
- `injury_reports` 新增来源 URL、文件 SHA256、不可变球员状态 payload、解析问题和发布时间验证标记。
- 新增 `prospective_runs`、`prospective_forecasts` 表及 ORM 模型，但尚无前瞻采集服务、固定协议或定时任务。
- 006 迁移文件已复制进当前 backend 容器执行；镜像尚未重建。后续重建时必须保留该迁移。

### 2. Kaggle 源文件已保存

- 当前配置数据集仍是 `eoinamoore/historical-nba-data-and-player-box-scores`。
- SDK 最新请求指向 527 版并返回 404；526 版也返回 404。确认 525 版可公开下载后，使用同一数据源的按文件下载入口。
- 已下载 `Games.csv`、`Players.csv`、`TeamStatistics.csv`、`PlayerStatistics.csv`，避免下载整个约 1.1GB 包。
- Docker 持久卷路径：`/app/ml/saved_models/data_sources/kaggle/v525/`。
- `source_manifest.json` 保存版本、URL、文件大小、列名、哈希和真实下载时间；`current.json` 指向当前文件目录。
- 已查看的文件首行比赛日为 2026-10-23；后续应完整扫描日期范围后报告源数据真实截止日，不把版本号当作今日数据证明。
- 新入口：`scripts/fetch_kaggle_files.py`。`KaggleCollector._ensure_dataset` 已增加显式源快照读取。

### 3. 统计补采已执行一次

新增 `backend/app/collector/stat_backfill.py`，`KaggleCollector.sync_game_stats` 改为调用它。修复旧逻辑只处理“完全没有球队统计”、跳过缺失球员统计的问题。

- 待修复 3,863 场；与源数据唯一匹配 3,823 场。
- 写入/更新 7,646 条球队统计、95,707 条球员统计。
- 所有统计列参与 upsert，不再仅更新得分、篮板、助攻。
- 分钟保留六位小数；真实进攻篮板和比赛时长用于效率/节奏计算。
- 补充 3,823 场真实来源开球时间、NBA 比赛日；本次也将匹配比赛的 `game_date` 校正为 UTC 开球时间。
- 未匹配球员 0、歧义比赛 0、比分冲突 0。
- 完整性重新验收，修订号提升。**写入成功不等于所有比赛已经完整。**

暂停时已结束比赛覆盖：

| 赛季 | 比赛数 | 球队完整 | 球员完整 | 有精确时间 |
|---|---:|---:|---:|---:|
| 2023-24 | 1383 | 1383 | 0 | 1383 |
| 2024-25 | 1385 | 1383 | 542 | 1385 |
| 2025-26 | 1387 | 1387 | 548 | 1055 |
| 2026-27 | 40 | 0 | 0 | 0 |

补采前球员完整数为 2024-25 的 0、2025-26 的 332；其余历史缺口仍需诊断。最新 40 场仍只有比分，不能称为球队状态已更新。

报告：`reports/data_quality_20261027/box_score_backfill.json`。

### 4. 官方伤病 PDF 试运行已完成

- 新脚本：`scripts/import_verified_injuries.py`；新增 Compose `data-worker` profile，复用现有 Java/nbainjuries 镜像。
- 新摄取服务：`backend/app/services/report_snapshot_service.py`。
- 试运行目标：2026-01-05 至 2026-01-15，每日 11:30、17:30 ET 报告。
- 成功解析 20 份 PDF，新增 608 条比赛/球队快照，其中 270 条完整、338 条部分完整。
- 608 条均验证发布时间；实际已入库发布时间范围为 2026-01-06 16:30 UTC 至 2026-01-15 22:30 UTC。
- 8 个报告球队/日期未能唯一绑定比赛，未强行关联。
- 1 月 5 日两份 PDF 发生 SSL EOF 错误，尚未重试。
- 发布时间取自 PDF 第一页标题，采集时间保存实际采集时刻，两者分开。
- `NOT YET SUBMITTED`、未知状态、未解析球员标记为部分报告，不写成健康或完整报告。
- 原始 PDF 和解析 JSON 位于 `/app/ml/saved_models/data_sources/official_injuries/`。
- 旧 `historical_player_injuries` 的 56,563 条记录完全保留，`game_id`、`published_at` 仍为空，没有伪造时间或批量改写。

报告：`reports/data_quality_20261027/injury_pilot.json`。源文件清单也已复制到该目录。

### 5. 比赛时间读取修复已写代码，尚未运行验证

`BallDontLieCollector._upsert_game` 改为读取 API 的真实 `datetime` 字段，保留来源 `date` 作为 NBA 比赛日；无有效时间不伪造。

已用 API 只读核实 2026-10-27 四场确实返回真实 `datetime`，但**尚未执行修改后的同步**，所以数据库 2026-27 仍无精确时间。后续需要验证并修复实际同步，不得把代码完成当作数据完成。

## 下一次继续的起点（按顺序）

1. 先检查本轮代码和迁移，增加真正的幂等、完整性和时间截断测试。目前新模块只做了 AST 解析；摄取与补采已实际运行，但尚未完成全部回归测试。
2. 诊断统计补采后仍不完整的比赛：区分源数据缺失、球员得分合计、分钟合计、加时和其他问题。定位最新 40 场为何未与 525 源数据匹配/补齐，不凭空补技术统计。
3. 验证 BallDontLie 时间修复，同步今天与必要的下一比赛日；处理 source NBA 日期和 UTC 日期跨日匹配，避免重复比赛。
4. **接入新的不可变 `InjuryReport.entries_json` 读取路径。** 当前 `PlayerFeatures` 仍读取旧历史表；`InjuryNetFeatures` 的严格 observed_at 条件也会把今天采集的历史回建快照排除。应明确区分：
   - 历史重建可使用真实发布时间早于 cutoff 的已验证快照，保留采集时间为今天，标明重建；
   - 真正线上赛前留档必须同时满足发布时间、实际采集时间早于同一个 as_of；
   - 选最新快照的完整性，不能用旧完整报告掩盖最新部分报告；Available 状态应覆盖旧 Out；
   - 旧缺时记录继续保留，不伪装成可验证赛前信息。
5. 对 270 条完整快照计算净补偿的门槛漏斗：比赛绑定、真实发布时间、至少五场完整参考、参考后的统计缺口、240 分钟容量、80% 历史产出覆盖。报告真实非零率再扩大伤病采集范围。
6. 再重建新语义版本的特征缓存，完成 E3 无伤病/聚合伤病/净补偿消融。旧候选使用本轮补采前的数据，不能据新覆盖率宣称它已经学到伤病。
7. 注意本轮校正了开球时间：评估切分和 bootstrap 应按 `source_game_date` 分组，而非直接截 UTC 日期；背靠背也要复核 NBA 比赛日；历史日期精度回放不得认证为真实赛前预测。
8. 实现前瞻服务：锁定协议及模型清单 hash，用同一个 as_of、同一个有序特征快照保存生产基线和最终候选；赛后不能补写为赛前记录。每 50 场输出进度，200 场前不依据成绩反复调参。
9. 重建/验证 backend 镜像、应用服务与后续采集调度。目前仍运行旧进程，文件挂载不代表旧进程已经重新导入新代码。
10. 未满足既定准入前，不执行任何模型晋升。

## 恢复命令参考（现在不执行）

```powershell
# 修复 1 月 5 日失败的真实 PDF 采集；已存在 payload 会按来源哈希去重
docker compose --profile data run --rm data-worker `
  /app/scripts/import_verified_injuries.py --start 2026-01-05 --end 2026-01-05 `
  --output /app/ml/saved_models/data_quality/injury_retry_jan05.json

# 统计补采入口；重新运行前先诊断现有不完整原因
docker compose --profile trainer run --rm -e PYTHONPATH=/app:/app/backend `
  --entrypoint python trainer /app/scripts/backfill_box_scores.py
```

**不要运行旧 `scripts/import_historical_injuries.py` 来“补齐时间”。** 旧脚本仍含 `observed_at=published_at` 以及过宽的完整报告判断，尚未改为新摄取逻辑。
