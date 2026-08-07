# Phase 0 — 基础设施架构文档

> 版本: 0.1.0 | 日期: 2026-07-29 | 状态: ✅ 已完成

---

## 目录

1. [整体架构](#1-整体架构)
2. [技术选型决策](#2-技术选型决策)
3. [数据库设计](#3-数据库设计)
4. [数据采集管道](#4-数据采集管道)
5. [伤病审核工作流](#5-伤病审核工作流)
6. [休赛期与空数据兼容](#6-休赛期与空数据兼容)
7. [目录结构规范](#7-目录结构规范)
8. [设计原则总结](#8-设计原则总结)

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose                         │
│                                                          │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │   Frontend    │  │   Backend     │  │  PostgreSQL  │ │
│  │  React 18     │  │  FastAPI      │  │     16       │ │
│  │  TypeScript   │──│  async        │──│  asyncpg     │ │
│  │  Vite + Nginx │  │  APScheduler  │  │              │ │
│  │  :3000        │  │  :8000        │  │  :5432       │ │
│  └───────────────┘  └───────────────┘  └──────────────┘ │
│                          │                                │
│                     [ML Trainer]                          │
│                  XGBoost + PyTorch                        │
│                  (Phase 1 实现)                            │
└──────────────────────────────────────────────────────────┘
```

### 容器分工

| 服务 | 镜像 | 职责 | 端口 |
|------|------|------|------|
| `postgres` | `postgres:16-alpine` | 持久化所有数据，健康检查 pg_isready | 5432 |
| `backend` | `python:3.12-slim` | REST API + APScheduler 定时采集 | 8000 |
| `frontend` | `node:20-alpine` → `nginx:alpine` | SPA 静态服务 + API 反向代理 | 3000 (→80) |

**网络拓扑**: frontend 通过 nginx `proxy_pass` 将 `/api/*` 请求转发到 backend:8000，backend 通过 asyncpg 连接 postgres:5432。三个容器在同一 Docker network 内通信，只有 frontend:3000 对外暴露。

---

## 2. 技术选型决策

### 2.1 Backend: FastAPI (async)

**选择理由**:
- 数据采集是 I/O 密集型操作（等待 nba_api HTTP 响应），异步模型不会阻塞其他请求
- APScheduler 的 `AsyncIOScheduler` 与 FastAPI 共用同一个 event loop，无需额外进程
- Pydantic v2 提供请求/响应验证，与 SQLAlchemy 2.0 的类型注解体系一致

**不选 Flask 的原因**: Flask 默认同步模型，虽然可以用 `quart` 或 `asyncio` 包装，但 FastAPI 原生 async 更简洁，且自动生成 OpenAPI 文档。

### 2.2 ORM: SQLAlchemy 2.0 (async)

**选择理由**:
- 2.0 版本原生支持 `async_sessionmaker` + `create_async_engine`
- 声明式映射 (`Mapped` + `mapped_column`) 提供完整类型检查
- Alembic 无缝集成，支持 async migration

**为什么不用 Tortoise ORM / SQLModel**: Tortoise 生态较小，SQLModel 与 Pydantic 深度耦合但迁移支持不如 Alembic 成熟。SQLAlchemy 是 Python 生态中最稳定、文档最丰富的 ORM。

### 2.3 调度器: APScheduler

**选择理由**:
- 个人分析工具不需要 Celery + Redis 的重量级方案
- `AsyncIOScheduler` 内嵌在 FastAPI 进程中，零额外依赖
- 支持 cron 和 interval 触发器，满足 "每 5 分钟采集一次" 的需求

### 2.4 数据采集: 仅 nba_api

**关键决策**: 不爬虎扑、不爬体彩。原因：

| 考虑因素 | 决策 |
|----------|------|
| 反爬风险 | 虎扑和体彩有强反爬措施，个人 IP 易被封 |
| 需求相关性 | 赔率数据不影响比赛结果预测 |
| 维护成本 | 爬虫需要持续维护解析逻辑和反爬策略 |
| 数据可靠性 | nba_api 是官方数据，权威性高于第三方 |

### 2.5 Frontend: React 18 + Vite + TypeScript + Tailwind

**选择理由**:
- **Vite** 替代 CRA：开发服务器秒启动，HMR 极快，生产构建基于 Rollup
- **TypeScript**：类型安全，API 响应类型与后端 Pydantic schema 对应
- **Tailwind CSS**：原子化 CSS，无需维护独立样式文件，深色模式天然支持
- **ECharts** (via echarts-for-react)：NBA 数据可视化需求（走势图、仪表盘、对比图）

---

## 3. 数据库设计

### 3.1 表结构全景

```
                    ┌──────────┐
                    │  teams   │
                    └────┬─────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴─────┐   ┌────┴─────┐   ┌─────┴──────┐
    │ players  │   │  games   │   │   teams    │
    └────┬─────┘   └────┬─────┘   │  (via FK)  │
         │               │         └────────────┘
    ┌────┴─────┐   ┌────┴─────────┐
    │ injuries │   │ team_game_    │
    └──────────┘   │ stats         │
                   │ game_player_  │
                   │ stats         │
                   └───────────────┘

    ┌──────────────┐    ┌──────────────────┐
    │ predictions  │───▶│ prediction_      │
    └──────────────┘    │ results          │
                        └──────────────────┘
```

### 3.2 中心事实表: `games`

`games` 是数据模型的中心，几乎所有其他表都通过 `game_id` 外键关联：

- `team_game_stats`：一场比赛两条记录（主客队各一）
- `game_player_stats`：一场比赛 N 条记录（上场球员每人一条）
- `predictions`：一场比赛可以有多条预测（不同模型版本）
- `prediction_results`：一条预测对应一条结果

### 3.3 伤病表设计: 审核工作流

```sql
-- 核心字段
source       VARCHAR(10)  -- 'nba_api' | 'manual'
confirmation VARCHAR(15)  -- 'pending' | 'confirmed' | 'modified'
injury_status VARCHAR(20) -- 'Out' | 'Doubtful' | 'Questionable' | 'Probable' | 'Available'
is_active    BOOLEAN      -- 当前是否生效
```

**设计原理**: 详见 [第 5 节 — 伤病审核工作流](#5-伤病审核工作流)。

### 3.4 预测表: 支持双模型 + 校准

```sql
model_name    VARCHAR(20)  -- 'xgboost' | 'mlp'
model_version VARCHAR(20)  -- 'v1.0.0'
calibrated    BOOLEAN      -- 是否已应用校准系数
calibration_factors JSONB  -- 校准参数快照
features_json JSONB        -- 输入特征快照（可复现性）
lookback_games INTEGER     -- 使用的回溯场数
```

**设计原理**: `features_json` 存储预测时的完整特征向量，允许事后分析"模型为什么做出这个预测"。`calibration_factors` 记录校准时使用的系数，支持校准效果的独立评估。

### 3.5 索引策略

```sql
-- 高频查询优化
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_odds_game_time ON odds_history(game_id, collected_at DESC);
CREATE INDEX idx_injuries_player ON player_injuries(player_id, is_active);
CREATE INDEX idx_injuries_active ON player_injuries(is_active) WHERE is_active = TRUE;
```

---

## 4. 数据采集管道

### 4.1 采集流程

```
APScheduler (每5分钟)
    │
    ▼
NBACollector.collect()
    │
    ├── 1. sync_teams()     → teams 表 (upsert)
    ├── 2. sync_players()   → players 表 (upsert, 全30队)
    ├── 3. sync_schedule()  → games 表 (upsert, 当前赛季)
    ├── 4. sync_game_results() → team_game_stats + game_player_stats
    │       │
    │       └── 检测 ≥3 场新 Final → _trigger_model_retraining()
    │
    └── 5. sync_injuries()  → player_injuries (confirmation='pending')
```

### 4.2 Upsert 策略

所有数据写入使用 PostgreSQL 的 `INSERT ... ON CONFLICT DO UPDATE`（通过 SQLAlchemy 的 `pg_insert().on_conflict_do_update()`）：

```python
# 示例：球队同步
stmt = pg_insert(Team).values(...).on_conflict_do_update(
    index_elements=["nba_team_id"],  # 唯一约束
    set_={...}                        # 冲突时更新这些字段
)
```

**为什么用 upsert 而不是 delete + insert**:
1. 避免外键级联问题（game 引用 team，删 team 会导致 game 孤行）
2. 保留手动录入的数据（如伤病的 confirmation 状态）
3. 原子操作，无竞态条件

### 4.3 nba_api 同步调用封装

nba_api 是同步库，通过 `asyncio.to_thread()` 包装避免阻塞 event loop：

```python
# 不会阻塞其他 FastAPI 请求
teams_data = await asyncio.to_thread(self._fetch_teams)
```

### 4.4 采集日志

每次采集结果写入 `data_collection_log` 表：

```
source        | nba_api
task_name     | full_sync
status        | success | failed
records_collected | 整数
error_message | 异常信息（nullable）
started_at / completed_at | 时间戳
```

用于监控采集健康状态、排查问题。

---

## 5. 伤病审核工作流

### 5.1 设计动机

NBA 官方伤病报告的延迟问题：
- 球队有时在赛前几小时才提交伤病报告
- nba_api 可能滞后于实时状态
- 中文媒体的伤病信息（如虎扑）更及时但不能爬取

### 5.2 状态机

```
                nba_api 自动拉取
                      │
                      ▼
               ┌─────────────┐
               │   pending    │  ← 系统建议稿
               │  (待确认)    │
               └──────┬──────┘
                      │ 用户在 AdminPage 审核
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────┐
    │confirmed │ │ modified │ │ 删除  │
    │ (已确认) │ │ (已修改) │ │      │
    └──────────┘ └──────────┘ └──────┘
          │           │
          └─────┬─────┘
                │
                ▼
         预测时优先使用

手动新增 ──→ confirmed ──→ 最高优先级
```

### 5.3 预测时的优先级逻辑

```
优先级排序（高→低）:
1. source='manual' (用户手动新增)
2. confirmation='modified' (用户审核并修改)
3. confirmation='confirmed' (用户一键确认)
4. confirmation='pending' (nba_api 自动拉取，未经审核)
```

如果用户某天忘记审核：
- 系统仍使用 `pending` 数据参与预测
- `data_quality.injuries_reviewed = false`
- `data_quality.injuries_pending_count = N`
- 前端显示 ⚠️ 警告

### 5.4 API 支持

```
POST /api/v1/injuries            # 手动新增（自动 confirmed）
PUT  /api/v1/injuries/{id}       # 修改伤病（confirmation → modified）
DELETE /api/v1/injuries/{id}     # 删除
POST /api/v1/injuries/confirm-all # 一键确认全部 pending
GET  /api/v1/injuries/?confirmation=pending  # 待审核列表
```

---

## 6. 休赛期与空数据兼容

### 6.1 赛季检测

```python
@staticmethod
def _determine_season() -> str:
    today = datetime.now()
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[-2:]}"  # "2025-26"
    else:
        return f"{today.year - 1}-{str(today.year)[-2:]}"   # "2024-25"
```

### 6.2 空数据处理策略

| 场景 | nba_api 行为 | 系统行为 |
|------|-------------|---------|
| 休赛期（7-9月） | 返回空列表或无进行中赛季 | `sync_schedule()` 返回 0，不报错 |
| 赛季中无比赛日 | 返回空列表 | `sync_game_results()` 返回 0 |
| nba_api 超时/不可用 | 抛出异常 | 捕获异常，记录 error_message，不影响其他子任务 |
| 赛季初 < 5 场 | 返回实际数据 | 前端自适应窗口（Phase 3 实现） |

### 6.3 优雅降级原则

> 每个采集方法都遵循 "try + log warning + return 0" 模式，确保单个子任务失败不影响其他子任务。

```python
async def sync_teams(self) -> int:
    try:
        teams_data = await asyncio.to_thread(self._fetch_teams)
    except Exception as e:
        logger.warning("Failed to fetch teams: %s", e)
        return 0  # 不抛异常，继续下一个子任务

    if not teams_data:
        logger.info("No teams returned (off-season?)")
        return 0  # 空数据不是错误
    ...
```

---

## 7. 目录结构规范

### 7.1 组织原则

```
backend/app/
├── models/       → 数据层：ORM 模型，一对一映射数据库表
├── schemas/      → 接口层：Pydantic 请求/响应模型
├── api/          → 路由层：HTTP 端点，薄层，只做参数解析和响应组装
├── services/     → 业务层：业务逻辑，调用 models 和 collectors
├── collector/    → 采集层：外部数据源采集，独立于业务逻辑
├── scheduler/    → 调度层：定时任务配置
└── utils/        → 工具层：跨模块共享的工具函数
```

**分层约束**:
- `api/` 不直接访问数据库 → 通过 `services/`
- `collector/` 独立于 `api/` → 可单独测试
- `models/` 只定义结构，不含业务逻辑

### 7.2 Frontend 组件分层

```
frontend/src/
├── api/          → API 客户端（axios 实例 + 各域端点）
├── types/        → TypeScript 类型定义（与后端 schema 对应）
├── hooks/        → 自定义 Hooks（数据获取、轮询、状态管理）
├── components/
│   ├── layout/   → 布局组件（Header, Sidebar, Container）
│   ├── common/   → 通用组件（Selector, Slider, Badge）
│   ├── prediction/ → 预测相关组件（ResultCard, Gauge）
│   ├── charts/   → ECharts 封装组件
│   └── teams/    → 球队相关组件
├── pages/        → 页面组件（一个页面一个文件）
└── styles/       → 全局样式 + Tailwind 配置
```

---

## 8. 设计原则总结

### 8.1 已确认的硬性约束

| 约束 | 来源 | 影响 |
|------|------|------|
| 零爬虫 | 用户反馈 | 简化架构，无反爬负担 |
| 不采集赔率 | 用户反馈 | 去掉 odds_history 表和相关采集逻辑 |
| XGBoost + MLP | 数据量限制 | Transformer 不适合 ~2000 场的数据集 |
| 手动录入伤病优先 | 数据准确性 | 设计审核工作流而非纯自动采集 |
| 5-15 场可调回溯 | 用户反馈 | 前端滑块 + 后端自适应逻辑 |

### 8.2 架构原则

1. **优雅降级 > 严格失败**: 子任务失败不应阻止其他子任务，空数据不是错误
2. **Upsert 而非 Replace**: 保留用户手动数据，避免外键级联问题
3. **数据可复现**: 预测时保存完整特征快照 (`features_json`)，事后可分析
4. **薄 API 层**: 路由只做参数解析和响应组装，业务逻辑在 services/
5. **独立可测试**: collector 可脱离 API 单独运行（seed_data.py 直接调用）

### 8.3 Phase 1 衔接点

Phase 1 将基于本 Phase 构建：
- 特征工程读取 `team_game_stats` + `game_player_stats` + `player_injuries`
- 模型训练使用 `games` 表的历史数据
- 推理结果写入 `predictions` 表
- 每日闭环重训练由 collector 的 `_trigger_model_retraining()` 触发
