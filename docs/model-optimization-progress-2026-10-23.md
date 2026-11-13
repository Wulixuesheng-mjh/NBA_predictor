# 模型优化实施进度（2026-10-23）

> 后续进度已更新到 `docs/model-optimization-progress-2026-10-27.md`：213 维模型已重训为影子候选 `20261027_213v7`，训练与审计报告已保存；活动模型尚未晋升。下文保留为 10 月 23 日的历史检查点。

## 当前状态

- 工作区修改均已保存在本地，尚未提交 Git。
- PostgreSQL、backend、frontend 容器已启动；数据库迁移为 `005_data_quality_ledger (head)`。
- 当前线上活动模型仍为旧包 `20261022_050952`（166 维），没有被本轮代码自动替换。
- 新训练流程默认生成影子候选；未通过实验和前瞻门槛前不写 `active.json`。
- 当前新特征维度为 213。

## 已完成实现

### 数据、时间和账本

- 比分、球队统计、球员统计分别保存完整性状态；Final 不再等同于统计完整。
- 增加来源比赛日、UTC 开球时间、时间精度、比赛类型、统计更新时间、加时分钟和回合估算标记。
- 增加数据修订号；统计、名单和伤病修改会让特征缓存失效。
- 新增伤病报告完整性快照和有时效的名单快照模型。
- 正式赛前、假设预览、重建回放三种模式分开；只有精确开球时间之前的正式赛前预测可写账本。
- 账本保存 XGB、MLP、基线、市场输入、伤病快照、数据质量、有序特征向量、语义版本和 hash。

### 基础错误和特征

- 背靠背改为比赛日间隔 1 天。
- top5 查询限定球队并按数值分钟排序。
- `expected_total` 的 `/200` 量纲错误改为 `/100`。
- 名单缺失不再编码为完全连续，增加缺失 mask。
- 球队状态特征只读取通过质量验收的统计；最近一场统计缺口进入预测警告。
- 增加 3PA/FGA、当季比赛数、统计覆盖率。
- 增加上季攻防节奏先验和当季 K=10 平滑值。
- 新节奏效率基线使用平滑后的 pace/off_rtg/def_rtg，新增 `shrunk_expected_total`。

### 时间权重和训练

- 实现 T0—T5 窗口与连续日期半衰期权重，窗口内权重均值为 1，并报告每季样本、总权重和有效样本量。
- XGBoost 胜负、分差、总分可分别配置训练策略；MLP 使用一套共享策略。
- XGBoost 早停和全量重拟合均使用对应目标权重，轮数为准确的 `best_iteration + 1`。
- MLP 使用逐样本加权损失和加权标准化；记录 best epoch 与学习率序列，再重新初始化并全量拟合。
- 总分同时训练直接回归和“收缩基线＋残差”候选，在相同验证样本上按 MAE 选择。
- `run_model_experiments.py` 使用共同完整比赛日验证集比较 T0—T5，重复候选只运行一次；可用 `--include-mlp` 比较 MLP。

### 伤病净补偿

- 参考分钟取最近十场完整球员统计，0.85 衰减，每场归一为 240 分钟，DNP 记 0。
- Out/Doubtful/Questionable/Probable 使用 0/0.25/0.60/0.95 可出场概率；二分法分配全队 240 分钟。
- 角色分钟优先当前队 180 天内最近十场，不足五场时向最近二十次出场收缩；不使用全生涯均值。
- 球员得分、抢断和盖帽每分钟产出向上季同位置均值加入 200 分钟先验收缩。
- 输出净进攻变化、防守事件变化、替补接管分钟、新增损失分钟、后卫/前锋/中锋缺口、转队角色占比和估算覆盖率。
- 长期缺阵参考分钟接近零，不重复扣减；复出可以产生正向净变化。
- 历史伤病导入器已改为保存真实报告发布时间、目标比赛、球队和完整报告快照；未知报告不会被写成健康。

### 区间、盘口和发布

- 真实赛前样本外残差最多保存 1,000 条；满 200 条才返回经验 80%/90% 区间。
- 盘口概率由回归预测分布和用户输入阈值计算；整数盘口返回赢、走、输概率。
- 新字段 `market_home_handicap=-6.5` 表示主让 6.5；与旧 `spread_line` 同时出现会报错。
- 训练默认写 `candidates/<version>.json`；`promote_model_release.py` 校验文件 hash、备份 `rollback.json` 后原子晋升。
- 旧 166 维活动包继续按前缀加载；对 v5 的错误 `expected_total` 尺度做兼容转换，并返回升级警告。

## 已完成验证

- 全部 Python 文件 AST 解析通过；`git diff --check` 通过（仅 CRLF 提示）。
- 优化不变量检查通过：时间权重、完整比赛日切分、总分量纲、名单缺失、开季平滑、240 分钟、长期缺阵/复出、盘口单调和整数走盘。
- XGBoost 合成冒烟通过：213 维、加权训练、全量重拟合、残差总分、保存/加载一致；示例总分 MAE 2.2994。
- MLP 合成冒烟通过：213 维、加权训练、重新初始化全量拟合、ONNX 导出/加载和 PyTorch/ONNX 误差门槛；best epoch 8。
- backend 和 frontend 生产镜像构建通过；TypeScript/Vite 构建通过。
- backend 健康检查通过；Alembic 为 head。
- 湖人主场对勇士非持久化预览通过：活动旧模型 XGB 总分 217.5，MLP 218.5，新的收缩节奏效率基线 230.0；相对 234.5 盘口仍偏低，页面会标记模型与基线相差 -12.5。
- 预览前后预测账本均为 51 条，证明假设预览未持久化。
- 日期精度比赛的正式赛前持久化请求返回 400；重建回放持久化请求返回 400。
- 数据质量批量刷新结果：30,791 场 Final，30,690 场比分完整，4,153 场球队统计完整，332 场球员统计完整。历史数据的主要瓶颈是真实 box score 缺失或分钟/球员得分不一致。

## 下一次继续的起点

1. 运行 `scripts/verify_feature_build.py`，用 2025-26 真实数据库构建一次 213 维特征集并查看伤病净补偿覆盖率。该容器运行本次未获执行授权，因此尚未执行。
2. 根据真实构建结果修复任何 SQL/覆盖率问题；特别检查历史完整伤病报告与至少五场完整球员统计的交集。
3. 全量构建一次新缓存，然后运行 T0—T5 XGB；胜出候选再运行 `--include-mlp`。
4. 按 E0—E4 完成开季先验、伤病组和直接/残差总分消融，并生成按比赛日 bootstrap 区间。
5. 只把通过开发期和 2025-26 历史审计的模型保留为影子候选；不晋升活动模型。
6. 建立连续 200 场真实赛前对照；每 50 场只报告进度，200 场前不改规则。

## 常用命令

```powershell
# 真实单赛季特征构建验证（不训练、不发布）
docker compose --profile trainer run --rm `
  -e PYTHONPATH=/app:/app/backend `
  --entrypoint python trainer /app/scripts/verify_feature_build.py

# T0—T5 XGB 共同验证集对照
docker compose --profile trainer run --rm `
  -e PYTHONPATH=/app:/app/backend `
  --entrypoint python trainer /app/scripts/run_model_experiments.py

# 加上 MLP 对照
docker compose --profile trainer run --rm `
  -e PYTHONPATH=/app:/app/backend `
  --entrypoint python trainer /app/scripts/run_model_experiments.py --include-mlp

# 显式晋升已通过门槛的候选（现在不要执行）
python scripts/promote_model_release.py <version>
```
