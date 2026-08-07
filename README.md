# 🏀 NBA Predictor — 体育彩票NBA竞猜分析平台

基于机器学习的NBA赛事预测工具，提供**胜负**、**让分胜负**和**总得分**三项预测，辅助中国体育彩票NBA竞猜投注决策。

## 功能特性

- 🔮 **双模型预测** — XGBoost（主力）+ MLP神经网络（辅助），两个模型独立预测、对比验证
- 📊 **126维特征工程** — 球队近期表现、伤病火力损失、球员状态、历史对战、比赛情境、pace/off-def匹配交互六大类特征
- 🔄 **每日自动闭环** — 赛果入库 → 预测vs实际对比 → XGBoost全量重训练 → 校准系数更新
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
| **数据库** | PostgreSQL 16 | 9张核心表，SQLAlchemy ORM |
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

### 训练预测模型

详细训练指南（数据准备、特征工程、超参调优、回测评估）请参阅 **[ML 模型训练指南](docs/ML_TRAINING_GUIDE.md)**。

```bash
# 使用 Docker（推荐）
docker compose --profile trainer run --rm trainer --lookback 10 --seasons 2025-26

# 或本地运行
cd ml && python ../scripts/train_models.py --seasons 2025-26 --lookback 10
```

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
              XGBoost全量重训练（~30秒）
                      │
                      ▼
              更新30天滑动窗口校准系数
                      │
                      ▼
              新模型自动激活 → 次日预测使用最新版本
```

## 本地开发

### Docker 开发模式（推荐）

`docker-compose.override.yml` 已配置源码热挂载，修改 Python 代码即时生效：

```bash
# 启动服务
docker compose up -d

# 查看日志
docker compose logs -f backend

# 运行测试（39 tests, SQLite in-memory, 无需 PostgreSQL）
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
