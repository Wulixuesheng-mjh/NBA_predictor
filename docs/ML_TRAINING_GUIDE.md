# ML 模型训练指南

本文档详细说明如何训练、评估和部署 NBA 预测平台的机器学习模型。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    训练流程                               │
│                                                         │
│  PostgreSQL ──→ FeatureBuilder ──→ Dataset ──┬──→ XGBoost Trainer
│                                              ├──→ MLP Trainer
│                                              └──→ Calibrator
│                                                         │
│  输出: saved_models/xgboost/ + saved_models/mlp/        │
└─────────────────────────────────────────────────────────┘
```

### 双模型策略

| 模型 | 角色 | 预测目标 | 输出 |
|------|------|---------|------|
| **XGBoost × 3** | 主力 | 胜负 (分类)、让分 (回归)、总分 (回归) | `win_prob_home`, `predicted_spread`, `predicted_total` |
| **MLP 三头** | 辅助对比 | 同上，单模型三输出头 | 同上 |

两个模型独立训练、独立推理，前端并列展示。当两者预测方向一致时，信号置信度更高。

---

## 前置条件

### 环境

- Python 3.12+
- PostgreSQL 16（需有足够历史比赛数据）
- 8 GB+ 内存（MLP 训练需要）
- （可选）NVIDIA GPU + CUDA 12.x（加速 MLP 训练）

### 依赖安装

```bash
# 方式 1: Docker（推荐，无需本地 Python 环境）
docker compose --profile trainer build

# 方式 2: 本地安装
cd ml
pip install -e ".[dev]"
```

---

## 第一步：数据准备

模型训练需要数据库中已有比赛数据。训练器从 PostgreSQL 读取数据，通过特征工程构建训练样本。

### 1.1 确保数据库有数据

```bash
# 检查数据库状态
docker compose exec backend python -c "
import asyncio
from app.database import async_session_factory
from sqlalchemy import text

async def check():
    async with async_session_factory() as sess:
        r = await sess.execute(text('SELECT COUNT(*) FROM games'))
        print(f'Games: {r.scalar()}')
        r = await sess.execute(text('SELECT COUNT(*) FROM team_game_stats'))
        print(f'Team stats: {r.scalar()}')
        r = await sess.execute(text('SELECT COUNT(*) FROM game_player_stats'))
        print(f'Player stats: {r.scalar()}')

asyncio.run(check())
"
```

### 1.2 填充数据（首次使用）

```bash
# 导入 2025-26 赛季数据
docker compose exec backend python scripts/seed_data.py --season 2025-26

# 导入多个赛季（建议至少 2 个赛季以获得足够训练样本）
docker compose exec backend python scripts/seed_data.py --season 2024-25
docker compose exec backend python scripts/seed_data.py --season 2025-26
```

### 1.3 数据量要求

| 数据量 | 可行性 | 建议 |
|--------|--------|------|
| < 200 场 | 不可训练 | 至少导入 1 个完整赛季 |
| 200-500 场 | 勉强可用 | 仅用 XGBoost，跳过 MLP |
| 500-2000 场 | 基本够用 | XGBoost + MLP，强正则化 |
| 2000+ 场 | 理想 | 双模型均可充分训练 |

> **注意**：每支球队单赛季 82 场，30 支球队共 1230 场比赛。2 个赛季约 2460 场，是理想的训练数据量。

---

## 第二步：理解特征工程

训练器通过 `FeatureBuilder` 从数据库构建约 70 维特征。特征分为四大类：

### A. 球队近期表现 (~50 维)

```
对每支球队，回溯 N 场 (5-15，默认 10)：
  - 胜率
  - 场均得分 / 场均失分 / 场均净胜分
  - 场均篮板 / 助攻 / 抢断 / 盖帽 / 失误
  - 投篮命中率 (FG%) / 三分命中率 (3P%) / 罚球命中率 (FT%)
  - 真实命中率 (TS%) / 有效命中率 (eFG%)
  - 进攻效率 (Off RTG) / 防守效率 (Def RTG) / Pace
  - 近 5 场趋势（线性回归斜率）
  - 近 5 场 vs 近 10 场差值（状态信号）
```

实现文件：`ml/src/features/team_features.py`

### B. 球员可用性 (~10 维)

```
基于伤病状态计算缺阵火力损失：

  缺阵火力 = Σ(缺阵球员场均得分 × 场均上场分钟 / 48)

  - 该公式用上场时间占比加权，真实反映球员在体系中的权重
  - 场均 20 分打 20 分钟的第六人 ≠ 场均 20 分打 36 分钟的核心

辅助指标：
  - 缺阵球员总上场时间占比之和
  - 缺阵关键球员标记 (加权损失 > 10)
  - 防守核心缺阵标记 (场均抢断+盖帽 > 2.5)
```

实现文件：`ml/src/features/player_features.py`

### C. 比赛情境 (~8 维)

```
  - 主队休息天数 / 客队休息天数
  - 主队是否背靠背 / 客队是否背靠背
  - 主队主场胜率 / 客队客场胜率
  - 主队近 5 主场胜率 / 客队近 5 客场胜率
  - 主队连胜/连败场数 / 客队连胜/连败场数
```

实现文件：`ml/src/features/context_features.py`

### D. 历史对战 (~3 维)

```
最近 3 次交手：
  - 主队胜场数
  - 场均分差
  - 场均总得分
```

实现文件：`ml/src/features/h2h_features.py`

### 自适应回溯窗口

赛季初比赛不足时的降级策略：

```
可用场数 N = min(已打场数, 用户选择 [5-15])

N >= 5  → 正常使用最近 N 场
N = 3-4 → 使用全部可用场数，降权处理
N = 1-2 → 补充上赛季球队级数据 (Net Rating / Off RTG / Def RTG / Pace)
          标注"含上季数据 · 阵容可能变化"
N = 0   → 纯上赛季球队级数据 + "阵容可能已变化"醒目提示
```

实现文件：`ml/src/features/feature_builder.py`

---

## 第三步：训练模型

### 3.1 Docker 方式（推荐）

```bash
# 基础训练：使用最近 10 场回溯窗口
docker compose --profile trainer run --rm trainer --lookback 10

# 指定赛季
docker compose --profile trainer run --rm trainer --seasons 2024-25 2025-26 --lookback 10

# 自定义参数
docker compose --profile trainer run --rm trainer \
  --seasons 2024-25 2025-26 \
  --lookback 12 \
  --xgboost-n-estimators 500 \
  --xgboost-max-depth 4 \
  --mlp-epochs 150
```

### 3.2 本地方式

```bash
cd ml
python scripts/train_models.py --seasons 2025-26 --lookback 10
```

### 3.3 训练参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `--seasons` | 自动检测 | - | 训练数据赛季，如 `2024-25 2025-26` |
| `--lookback` | 10 | 5-15 | 回溯比赛场数 |
| `--xgboost-n-estimators` | 300 | 100-1000 | XGBoost 树数量 |
| `--xgboost-max-depth` | 5 | 3-8 | 最大树深度（小数据用浅树） |
| `--xgboost-learning-rate` | 0.05 | 0.01-0.3 | 学习率 |
| `--mlp-epochs` | 200 | 50-500 | MLP 训练轮数 |
| `--mlp-batch-size` | 64 | 16-256 | MLP 批量大小 |
| `--mlp-learning-rate` | 0.001 | 0.0001-0.01 | MLP 学习率 |
| `--no-calibrate` | false | - | 跳过校准系数计算 |

### 3.4 训练过程详解

训练分 5 步执行（见 `ml/src/trainer.py`）：

```
Step 1/5: 构建训练数据集
  ├── 从 PostgreSQL 读取所有 Final 状态比赛
  ├── 为每场比赛构建 70 维特征向量
  ├── 过滤：每队需要至少 5 场前置比赛（否则无法提取回溯特征）
  └── 输出：X (N×70), y_win (N,), y_spread (N,), y_total (N,)

Step 2/5: 时序划分
  ├── 前 85% 作为训练集，后 15% 作为验证集
  └── 使用时序划分而非随机划分（防止未来信息泄露）

Step 3/5: 训练 XGBoost
  ├── 三个独立模型：XGBoost-Win, XGBoost-Spread, XGBoost-Total
  ├── 每个模型独立训练 + Early Stopping (patience=50)
  └── 保存至 ml/saved_models/xgboost/

Step 4/5: 训练 MLP
  ├── 单模型三输出头：Win Head, Spread Head, Total Head
  ├── 架构: Linear(70→128→64→32) + BatchNorm + Dropout(0.3)
  ├── 优化器: AdamW + ReduceLROnPlateau + EarlyStopping(patience=30)
  └── 保存至 ml/saved_models/mlp/

Step 5/5: 计算校准系数
  ├── 在验证集上对比预测值 vs 实际值
  ├── 计算系统性偏差（如模型是否系统性高估主队）
  └── 保存校准系数供推理时使用
```

---

## 第四步：超参调优 (Optuna)

训练脚本内置了 Optuna 超参搜索（可选启用）：

```bash
# XGBoost 超参搜索
docker compose --profile trainer run --rm trainer \
  --tune-xgboost \
  --tune-trials 100 \
  --seasons 2024-25 2025-26

# MLP 超参搜索
docker compose --profile trainer run --rm trainer \
  --tune-mlp \
  --tune-trials 50
```

Optuna 搜索空间：

| 超参 | 搜索范围 | 说明 |
|------|---------|------|
| `max_depth` | 3-6 | 越深越容易过拟合 |
| `n_estimators` | 100-500 | 树越多越稳定但越慢 |
| `learning_rate` | 0.01-0.1 | 与 n_estimators 成反比 |
| `subsample` | 0.7-1.0 | 行采样率 |
| `colsample_bytree` | 0.7-1.0 | 列采样率 |
| `reg_alpha` | 1e-8 - 1.0 | L1 正则化 |
| `reg_lambda` | 1e-8 - 1.0 | L2 正则化 |

搜索目标：最小化验证集 `spread_mae`（让分平均绝对误差）。

---

## 第五步：回测评估

训练完成后，运行回测评估模型在历史数据上的表现：

```bash
# Docker 方式
docker compose --profile trainer run --rm trainer \
  --backtest \
  --seasons 2025-26

# 本地方式
cd ml
python scripts/train_models.py --backtest --seasons 2025-26
```

### 回测指标

```
Win Accuracy:     0.683   # 胜负预测准确率 (目标 > 60%)
Spread MAE:       7.23    # 让分平均绝对误差 (目标 < 8.0)
Spread RMSE:      9.12    # 让分均方根误差
Spread ±4:        42.1%   # 让分误差在 ±4 分内的比例 (目标 > 40%)
Spread ±6:        58.3%   # 让分误差在 ±6 分内的比例 (目标 > 55%)
Total MAE:        11.45   # 总分平均绝对误差 (目标 < 12.0)
Total RMSE:       14.67   # 总分均方根误差
Total ±8:         39.8%   # 总分误差在 ±8 分内的比例 (目标 > 35%)
Total ±10:        51.2%   # 总分误差在 ±10 分内的比例 (目标 > 50%)
```

### 验证标准（来自项目计划）

| 指标 | 目标 | 说明 |
|------|------|------|
| 让分 ±4 分准确率 | > 55% | 这是核心指标 |
| 总分 ±8 分准确率 | > 55% | 另一个核心指标 |
| 胜负准确率 | > 60% | 基准要求 |

如未达标，请参考下方"调优建议"。

### 回测方法说明

回测采用 **walk-forward（无未来信息泄露）** 方式：

1. 按比赛日期升序排列
2. 对每场比赛，仅使用该日期之前的数据构建特征
3. 用训练好的模型（在回测开始日期之前的数据上训练）进行预测
4. 对比预测值 vs 实际值

> **重要**：基础回测不会在 walk-forward 过程中重训练模型。如需完整的每日重训练回测，请使用 feedback loop（见第七步）。

---

## 第六步：模型文件结构

训练完成后，检查模型文件是否生成：

```bash
ls -la ml/saved_models/xgboost/
# xgboost_win_20260730_120000.json    # XGBoost 胜负模型
# xgboost_spread_20260730_120000.json # XGBoost 让分模型
# xgboost_total_20260730_120000.json  # XGBoost 总分模型
# metadata.json                        # 训练元数据（版本、指标、时间戳）

ls -la ml/saved_models/mlp/
# mlp_20260730_120000.pt              # MLP 模型权重
# mlp_config.json                      # MLP 配置（输入维度、隐藏层等）
# metadata.json                        # 训练元数据

ls -la ml/saved_models/
# calibration.json                     # 校准系数
```

### 模型版本管理

- 版本号格式：`YYYYMMDD_HHMMSS`（训练时间戳）
- `metadata.json` 记录训练指标，用于追踪模型演进
- 后端通过 `ACTIVE_MODEL_VERSION` 环境变量指定使用哪个版本
- 留空则自动使用最新版本

---

## 第七步：每日闭环优化

这是整个系统最核心的自动化流程。配置后无需人工干预。

### 工作流

```
采集器检测到 ≥3 场新 Final 比赛
              │
              ▼
    ① 对比预测 vs 实际结果
       存入 prediction_results 表
              │
              ▼
    ② XGBoost 全量重训练 (~30-60秒)
       (加入新赛果数据)
              │
              ▼
    ③ 更新 30 天滑动窗口校准系数
              │
              ▼
    ④ 新模型自动激活
       次日预测自动使用最新版本
```

### 触发条件

- 自动：`data_collection_log` 检测到新 Final 比赛 ≥ 3 场
- 手动：`POST /api/v1/admin/train-model`
- Docker：`docker compose --profile trainer run --rm trainer`

### 配置文件

相关设置在 `ml/src/config.py`：

```python
min_new_games_for_retrain: int = 3   # 触发重训练的最小新比赛数
auto_retrain: bool = True            # 是否自动重训练
calibration_window_days: int = 30    # 校准滑动窗口天数
min_games_for_calibration: int = 10  # 校准所需最小比赛数
```

---

## 调优建议

### 模型不收敛 / 准确率低

1. **检查数据量**：至少需要 500+ 场比赛才能期待合理结果
2. **检查特征质量**：确保 `team_game_stats` 表有完整的进阶数据（Off RTG, Def RTG, Pace, TS%, eFG%）
3. **减小 max_depth**：数据少时用 depth=3 或 4，避免过拟合
4. **增大正则化**：增加 `reg_alpha` 和 `reg_lambda`
5. **检查数据泄露**：确保验证集时序上在训练集之后

### 让分预测偏差大

1. 检查主客场特征是否正常（`context_features.py`）
2. 增加 `xgboost_n_estimators` 到 500
3. 尝试更大的回溯窗口（12-15 场）

### 总分预测偏差大

1. 检查 Pace 和 Off RTG/Def RTG 是否正确采集
2. 总分天然方差大，±8 分准确率 > 55% 即为合格

### MLP 效果不如 XGBoost

这是预期现象。数据量在几千条级别时，XGBoost 通常优于神经网络。MLP 作为辅助参考，当两者一致时信号更强。

---

## 故障排查

### "No training samples found"

**原因**：数据库中没有足够的已完成比赛。

**解决**：
```bash
# 检查比赛数量
docker compose exec postgres psql -U nba_admin -d nba_predictor \
  -c "SELECT status, COUNT(*) FROM games GROUP BY status;"

# 填充历史数据
docker compose exec backend python scripts/seed_data.py --season 2024-25
docker compose exec backend python scripts/seed_data.py --season 2025-26
```

### "Cannot connect to database"

**原因**：ML 容器无法访问 PostgreSQL。

**解决**：
```bash
# 检查 PostgreSQL 状态
docker compose ps postgres

# 确认网络配置
docker compose exec trainer ping postgres
```

### "CUDA out of memory" (MLP 训练)

**解决**：
```bash
# 减小批量大小
docker compose --profile trainer run --rm trainer --mlp-batch-size 16

# 或在 CPU 上训练（默认）
```

### 模型文件未生成

```bash
# 检查目录权限
ls -la ml/saved_models/

# 确保 volume 正确挂载
docker compose run --rm -v ./ml/saved_models:/app/ml/saved_models trainer
```

---

## 下一步

训练完成并验证回测指标达标后：

1. **启动全平台**：`docker compose up -d`
2. **验证推理**：通过前端或 API 生成预测，确认模型正常加载
3. **监控准确率**：在 HistoryPage 查看近 7/14/30 天准确率仪表盘
4. **定期检查**：每周查看回测报告，必要时重新调参

---

## 相关文件索引

| 文件 | 用途 |
|------|------|
| `ml/src/trainer.py` | 训练流程编排 |
| `ml/src/predictor.py` | 双模型推理 + 一致性分析 |
| `ml/src/backtest.py` | Walk-forward 回测评估 |
| `ml/src/feedback.py` | 每日闭环（评估→重训练→校准） |
| `ml/src/config.py` | ML 配置（超参、路径、阈值） |
| `ml/src/database.py` | 独立数据库连接 |
| `ml/src/features/feature_builder.py` | 特征编排 + 自适应窗口 |
| `ml/src/features/team_features.py` | 球队表现特征 |
| `ml/src/features/player_features.py` | 伤病影响指数 |
| `ml/src/features/h2h_features.py` | 历史对战特征 |
| `ml/src/features/context_features.py` | 比赛情境特征 |
| `ml/src/models/xgboost_model.py` | XGBoost 三模型训练/推理 |
| `ml/src/models/mlp_model.py` | PyTorch MLP 三头模型 |
| `ml/src/models/calibration.py` | 滑动窗口校准 |
| `ml/Dockerfile` | 训练容器 |
| `scripts/train_models.py` | CLI 训练入口脚本 |
