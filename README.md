# 🏀 NBA Predictor — 体育彩票NBA竞猜分析平台

基于机器学习的NBA赛事预测工具，提供**胜负**、**让分胜负**和**总得分**三项预测，辅助中国体育彩票NBA竞猜投注决策。

当前生产已启用 1022 MLP 固定输入修复，XGB 仍为主模型；213 维研究候选未晋升。生产启停使用 `./scripts/production_stack.ps1 ps` / `up`，以保留旧影子隔离配置。部署版本、真实留档验证和回滚命令见 [2026-11-12 部署记录](docs/fixed-mlp-production-deployment-2026-11-12.md)。

## 功能特性

- 🔮 **双模型预测** — XGBoost（主力）+ MLP神经网络（辅助），两个模型独立预测、对比验证
- 📊 **213维特征工程** — 近期状态、开季先验、伤病净补偿、历史对战、比赛情境与攻防节奏交互
- 🔄 **可审计闭环** — 赛果入库 → 样本外评估 → 候选重训练 → 发布门槛 → 原子晋升
- 🏥 **伤病审核工作台** — NBA API自动拉取伤病建议稿，支持逐条确认/修改/删除/一键确认
- 🎚️ **自适应回溯窗口** — 5-15场可调，赛季初自动降权并补充上赛季数据
- 📈 **ECharts可视化** — 雷达对比图、准确率仪表盘、近期战绩走势
- 🐳 **Docker一键部署** — docker-compose编排，3个服务一键启动

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| **后端** | Python FastAPI + SQLAlchemy 2.0 | 异步API，APScheduler定时采集 |
| **前端** | React 18 + TypeScript + Vite | SPA，ECharts可视化，Tailwind CSS |
| **ML** | XGBoost + PyTorch MLP | 双模型，Optuna超参调优，30天滑动窗口校准 |
| **数据库** | PostgreSQL 16 | 比赛、统计、快照、预测账本与修订号，SQLAlchemy ORM |
| **部署** | Docker Compose | 3个常驻服务（PostgreSQL + Backend + Frontend）+ ML Trainer（按需启动） |
| **数据源** | nba_api（官方） + 手动录入伤病 | 零爬虫，零反爬风险 |

## 快速开始

### 前置要求

- Docker & Docker Compose v2
- （可选，本地开发）Python 3.12+ & Node.js 20+

### 启动步骤

```bash
# 1. 克隆项目
git clone <repo-url>
cd NBA_predictor

# 2. 配置环境变量（可选，.env 中有默认值）
# 编辑 .env 修改数据库密码等配置

# 3. 启动所有服务
docker compose up -d

# 4. 初始化实时数据（nba_api，需先训练模型容器才有 scripts）
docker compose --profile trainer run --rm --entrypoint python trainer scripts/seed_data.py --season 2025-26

# 5. (可选) 导入 Kaggle 历史数据集（7赛季完整数据）
docker compose --profile trainer run --rm --entrypoint python trainer scripts/import_kaggle_data.py

# 6. 训练预测模型
docker compose --profile trainer run --rm trainer \
    --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
    --lookback 15
```

启动后访问：
- **前端界面**: http://localhost:3000
- **API文档**: http://localhost:8000/api/docs
- **健康检查**: http://localhost:8000/health

### 本机已验收的发布运行方式

当前应用发布镜像为 `nba-predictor-backend:feedback-fixed-20261030`。在已经保存此镜像的本机运行：

```bash
docker compose -f docker-compose.yml -f docker-compose.release.yml up -d --no-build --pull never
```

该配置只挂载模型数据卷，后端和 ML 推理代码固定在镜像中，不自动加载开发配置的源码挂载。发布配置不负责重新构建镜像；代码更新应先构建、隔离验收，再更新固定镜像版本。默认 `docker compose up` 仍是开发模式。

自动采集和夜间任务只结算预测，不在 API 进程中训练或更改校准。候选训练使用独立 trainer 环境；训练成功也不会自动晋升。主模型仍为 `20261022_050952`，213 维总分候选仅做影子观察。详见 [运行修复与发布记录](docs/model-runtime-release-2026-10-30.md)。

### 训练预测模型

详细训练指南（数据准备、特征工程、超参调优、回测评估）请参阅 **[ML 模型训练指南](docs/ML_TRAINING_GUIDE.md)**。

```bash
# 使用 Docker（推荐）
docker compose --profile trainer run --rm trainer --lookback 10 --seasons 2025-26

# 或本地运行
cd ml && python ../scripts/train_models.py --seasons 2025-26 --lookback 10
```

#### 特征集缓存

全量训练里 ~95% 的时间花在逐场构建特征上（约 1 分钟 / 1000 场）。构建结果按
「特征结构 + 训练参数 + 数据库指纹」做 hash，缓存为 `ml/saved_models/feature_cache/train_<key>.npz`：

- **命中条件**：特征名列表、seasons/lookback/min_games_before、以及 games / 技术统计 /
  球员 / 伤病表的行数与最大更新时间都没变。任何一场新比赛、比分修正、交易换队、
  伤病录入都会改变指纹，缓存自动失效，不会读到旧数据。
- **典型收益**：MLP 训练失败重跑、超参搜索、回测复用同一份特征时，从约 50 分钟降到几秒。
- **开关**：`--no-cache`（跳过读写）、`--clear-cache`（训练前清空）、
  `FEATURE_CACHE_DIR`（自定义目录，默认 `ml/saved_models/feature_cache`）。
- **注意**：新比赛入库后仍会全量重建（保证特征正确）；缓存只消除"数据没变还重算"的浪费。
  查看/清理：`DatasetCache().info()` / `.clear()`（`ml/src/features/dataset_cache.py`）。

#### 特征截断与 MLP 目标标准化

- **静态截断**（[feature_builder.py](ml/src/features/feature_builder.py:125)）：`rest_days`/`days_since_*`
  截到 0–14（休赛期 295 天不是"休息"）、`*_ratio` 截到 ±50（分母趋零时会冲到 392）、
  `*_pct` 截到 ±1。规则写死在代码里，训练和推理走同一条路径，不会出现 train/serve 偏移。
- **目标标准化**（[mlp_model.py](ml/src/models/mlp_model.py:230)）：spread（±10）和 total（约 220）
  量级差两个数量级，混在一个 loss 里会让总分头主导梯度。现在用 z-score 训练、推理时反归一化；
  旧模型没有目标统计量时自动按原样处理。另外加了梯度裁剪（max_norm=1.0）。
- **合法性过滤**：总分 <120 或 >400 的 Final 比赛（历史导入产生的 0-0 等脏数据）不参与训练。
- **统一时间策略**：特征缓存保存全部因果样本；T0—T5 的训练窗口与半衰期权重在缓存之后应用。
  XGBoost 可按胜负、分差、总分分别选择策略，MLP 用同一窗口和逐样本加权损失，验证指标始终等权。

#### 校准、发布与可复现

- **校准默认关闭**（`calibration_enabled=False`）：原先「按实际赛果标准差放大预测」的做法在验证集上
  让分差 MAE 从 11.45 → 11.78、总分 15.06 → 15.76，属于纯放大误差；系数还只按 XGBoost 拟合，
  却被同时套到 MLP 上（旧 MLP 的总分因此被夹到 180 下限）。现在系数照常计算落盘，但默认不套用，
  且**永不套用到 MLP**。要启用必须先在滚动样本外窗口证明有改善。
- **输出合理性校验**：总分不在 150–300、|让分| > 45、胜率越界，一律判定模型异常 → 该模型置空并在
  `data_quality.warnings` 里说明原因（页面会显示「MLP 输出异常…已忽略该模型」），不再把裁剪值伪装成预测。
- **最终模型全量重训**（`refit_on_full`）：训练/验证切分只用来确定早停轮次，随后用全部可用数据重新拟合
  （此前线上模型实际只见过 2022 年以前的数据）。
- **候选与原子发布**：训练默认写入 `ml/saved_models/candidates/<version>.json`，不会改变线上版本。
  达到历史审计和前瞻门槛后，用 `scripts/promote_model_release.py <version>` 校验文件 hash 并原子替换
  `active.json`；上一份清单保存为 `rollback.json`。`--activate` 仅供显式受控发布。
- **预测可回放**：每条预测保存原始特征快照。`python scripts/replay_prediction.py --prediction-id 26`
  可重放任一条历史预测，比对「当时展示值 vs 现在重算值」。
- **界面显示发布信息**：预测结果标题下方展示模型版本与「训练数据截止 YYYY-MM-DD」，MLP 不可用时明确标注。

#### 揭幕战与伤病建模（第三阶段）

- **长休赛期单独表示**：`rest_days` 仍截断在 14 天，另增 `layoff_days`（0–120 天）与
  `is_season_opener`（间隔 >45 天记 1），揭幕战不再被当成"休息了两周"。
- **开季先验**：对手校正缓存开季回退上赛季终值；近 10 场窗口在揭幕战时本身就是上赛季末段数据，
  再叠加阵容连续性特征（`core_retention` 等）描述阵容变化。
- **伤病按概率与近期表现**：`Out 1.0 / Doubtful 0.75 / Questionable 0.4 / Probable 0.05` 加权，
  球员产出用最近 45 天（无近期比赛则回退赛季均值）估算；新增 `injury_defense_lost`（抢断+盖帽损失）
  与 `injury_probable_count`，攻防分开计。
- **节奏 × 效率基线**：`pace × (攻防效率均值)` 独立估算期望总分，随预测一起返回
  （`baseline.status = divergent` 表示与模型分歧超过 8 分），页面直接显示「基线 222.1（模型 218.5，差 -3.6）」。

#### 数据新鲜度、净补偿与不确定性（第四阶段）

- **数据完整性**：比分、球队统计、球员统计分别验收；Final 不再等同于统计完整。缺统计的比赛进入补采，
  加时分钟折算、比赛类型、来源比赛日、精确开球时间和时间精度均单独保存。
- **时间衰减**：`scripts/run_model_experiments.py` 在同一完整比赛日验证区间比较 T0—T5，报告每季样本、
  权重和有效样本量；`--include-mlp` 同时执行共享窗口的 MLP 对照。
- **伤病净补偿**：参考轮换按 0.85 衰减并归一为 240 分钟，Out 分钟强制为 0，其余状态按出场概率分配；
  球员产出向上季同位置均值加入 200 分钟先验收缩，输出替补接管、位置缺口、转队角色和复出净变化。
- **总分残差模型**：收缩后的节奏、进攻效率和对手防守效率生成结构基线；XGBoost 在相同验证样本上比较
  直接总分与“基线＋残差”，只保留 MAE 更低者。
- **区间与盘口概率**：只使用最多 1,000 条真实赛前样本外残差；满 200 条后返回 80%/90% 区间。
  让分与大小分概率直接响应输入盘口，整数盘返回赢、走、输概率。


#### 训练/推理一致性（未来信息泄漏修复）

- **对手校正**：`OpponentStatsCache` 按日期累积，`get(team, season, date)` 只看到该日期之前的比赛；
  开季没有本赛季样本时回退到上赛季终值（顺带成为揭幕战的先验）。在线单场预测走同一套语义。
- **阵容连续性**：当季出场分钟只统计该日期之前的比赛（`before_date` 过滤），不再读取赛季后半段。
- 验证脚本（容器内）：对任意 (队, 赛季, 日期)，缓存结果与「仅取该日期前比赛」的直接聚合逐位一致。


## 项目结构

```
NBA_predictor/
├── docker-compose.yml          # 服务编排
├── .env.example                # 环境变量模板
├── .dockerignore
│
├── backend/                    # FastAPI 后端
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py             # 应用入口 + 中间件
│   │   ├── config.py           # 配置管理（环境变量）
│   │   ├── database.py         # 异步引擎 + Session
│   │   ├── models/             # 9张表 ORM 模型
│   │   ├── schemas/            # Pydantic 请求/响应模型
│   │   ├── api/                # REST 端点（teams/players/games/injuries/predictions/admin）
│   │   ├── services/           # 业务逻辑层
│   │   ├── collector/          # nba_api 数据采集器
│   │   ├── scheduler/          # APScheduler 定时任务
│   │   └── utils/              # 结构化日志等工具
│   └── tests/                  # pytest 测试套件
│
├── ml/                         # 机器学习模块
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       ├── config.py           # ML 配置
│       ├── database.py         # 独立数据库连接
│       ├── features/           # 特征工程（team/player/h2h/context）
│       ├── models/             # XGBoost + MLP + 校准 + 超参调优
│       ├── trainer.py          # 训练流程编排
│       ├── predictor.py        # 推理服务（双模型+一致性）
│       ├── backtest.py         # 历史回测评估
│       └── feedback.py         # 每日闭环优化
│
├── frontend/                   # React SPA 前端
│   ├── Dockerfile
│   ├── nginx.conf              # Nginx 配置（含反向代理）
│   ├── package.json
│   └── src/
│       ├── api/                # Axios API 客户端（6模块）
│       ├── hooks/              # 自定义 React Hooks（6个）
│       ├── components/         # 可复用组件
│       ├── pages/              # 8个页面
│       └── types/              # TypeScript 类型定义
│
└── scripts/
    ├── init_db.sh              # PostgreSQL 初始化
    ├── seed_data.py            # 数据初始化脚本
    └── train_models.py         # CLI 训练脚本
```

## API 端点概览

所有端点前缀 `/api/v1`：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/teams` | 球队列表（支持conference筛选） |
| `GET` | `/teams/{id}` | 球队详情 + 近期战绩 |
| `GET` | `/teams/{id}/players` | 球队阵容 |
| `GET` | `/players` | 球员列表（支持team_id/search/position筛选） |
| `GET` | `/players/{id}` | 球员详情 + 赛季场均 + 伤病 |
| `GET` | `/games` | 比赛列表（支持date/status/team/season筛选） |
| `GET` | `/games/upcoming` | 未来7天赛程 |
| `GET` | `/games/{id}` | 比赛详情 + Box Score |
| `GET` | `/injuries` | 伤病列表（支持team_id/confirmation筛选） |
| `POST` | `/injuries` | 手动录入伤病（自动确认） |
| `PUT` | `/injuries/{id}` | 更新伤病（标记为modified） |
| `DELETE` | `/injuries/{id}` | 删除伤病（软删除） |
| `POST` | `/injuries/confirm-all` | 一键确认全部待审核 |
| `POST` | `/predictions` | **生成预测**（核心端点） |
| `GET` | `/predictions` | 预测历史列表 |
| `GET` | `/predictions/{id}` | 预测详情 + 实际结果对比 |
| `POST` | `/admin/collect` | 手动触发数据采集 |
| `GET` | `/admin/collect/status` | 采集日志 |
| `GET` | `/admin/stats` | 系统统计 |
| `POST` | `/admin/train-model` | 手动触发模型训练 |

完整API文档访问：`http://localhost:8000/api/docs`

## 预测请求示例

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 1,
    "away_team_id": 2,
    "lookback_games": 10
  }'
```

响应示例：

```json
{
  "home_team_name": "Los Angeles Lakers",
  "away_team_name": "Boston Celtics",
  "lookback_games": 10,
  "xgboost": {
    "win_prob_home": 0.62,
    "win_prob_away": 0.38,
    "predicted_spread": -3.5,
    "predicted_total": 218.5,
    "model_version": "xgboost_20260729_120000"
  },
  "mlp": {
    "win_prob_home": 0.58,
    "win_prob_away": 0.42,
    "predicted_spread": -2.0,
    "predicted_total": 215.0
  },
  "consensus": {
    "win_agree": true,
    "spread_direction_agree": true,
    "total_agree": false,
    "agreement_score": 0.67
  },
  "recommended": "xgboost",
  "data_quality": {
    "home_games_available": 10,
    "away_games_available": 10,
    "injuries_reviewed": true,
    "warnings": []
  },
  "calibration_applied": true
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_NAME` | `nba_predictor` | 数据库名 |
| `DB_USER` | `nba_admin` | 数据库用户 |
| `DB_PASSWORD` | `nba_predictor_2024` | 数据库密码 |
| `DB_PORT` | `5432` | 数据库端口 |
| `BACKEND_PORT` | `8000` | 后端API端口 |
| `FRONTEND_PORT` | `3000` | 前端Nginx端口 |
| `SCHEDULER_ENABLED` | `true` | 是否启用自动采集 |
| `COLLECTION_INTERVAL_MINUTES` | `5` | 采集间隔（分钟） |
| `NBA_API_TIMEOUT` | `30` | NBA API 超时（秒） |
| `DEBUG` | `false` | 调试模式 |
| `TRAINER_ARGS` | `--lookback 10` | ML训练容器参数 |

## 伤病审核工作流

```
nba_api 自动拉取 ──→ 状态: "待确认" ──→ 用户审核 ──┬── 一键确认 → "已确认"
                                                    ├── 修改后确认 → "已修改"
                                                    └── 删除 → 移除

预测时的优先级:
  已确认 (confirmed) = 已修改 (modified) > 待确认 (pending)
  手动录入 (manual) > nba_api 自动采集
```

## 模型闭环优化

```
比赛结果入库 ──→ 检测≥3场新Final比赛
                      │
                      ▼
              预测vs实际对比（存入prediction_results）
                      │
                      ▼
              XGBoost/MLP候选全量重训练
                      │
                      ▼
              更新30天滑动窗口校准系数
                      │
                      ▼
              保存影子候选 → 达到准入门槛后原子晋升
```

## 本地开发

### Docker 开发模式（推荐）

`docker-compose.override.yml` 已配置源码热挂载，修改 Python 代码即时生效：

```bash
# 启动服务
docker compose up -d

# 查看日志
docker compose logs -f backend

# 运行测试（SQLite in-memory，无需 PostgreSQL）
docker compose exec backend python -m pytest tests/ -v
```

### 后端本地开发

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

### 前端本地开发

```bash
cd frontend
npm install
npm run dev    # Vite dev server on :5173, 自动代理 /api → localhost:8000
```

### 运行测试

```bash
# Docker 方式（推荐，无需本地 PostgreSQL）
docker compose exec backend python -m pytest backend/tests/ -v

# 本地方式
cd backend
python -m pytest tests/ -v
```

### 模型回测

```bash
docker compose --profile trainer run --rm trainer \
    --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
    --lookback 15
# 输出中包含 Walk-forward Backtest 完整评估指标
```

## 许可

MIT License — 个人分析工具，仅供学习参考。
