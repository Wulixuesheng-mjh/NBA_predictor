# 下阶段改进计划 — 第 4 次会话

> 目标会话: 2026-08-04
> 前次会话产出: 特征去重完成 (121→110维), 训练验证通过, 相关对-68%
> 当前状态: 休赛期 (2026-27 赛季预计 10月中旬开打), 模型可训练但无实时比赛可预测

---

## 本次会话目标

完成 4 个 P2 级改进 + 1 个 P1 修复，把项目从"能用"推至"好用且可解释"。

---

## 任务清单（按执行顺序）

### 🔴 先决条件 — 启动 Docker 环境

```bash
cd C:\Users\马俊浩\PycharmProjects\NBA_predictor
docker compose up -d
docker compose ps  # 确认 postgres + backend + frontend 都在运行
```

---

### Task 1: SHAP 模型可解释性 (P2 #12) ⭐ 推荐首先做

**为什么先做这个**: 这是用户最感兴趣的功能 — 预测结果旁边展示"为什么给出这个预测"。

#### 1a. 创建分析脚本 `ml/src/shap_analysis.py`

```
功能:
  - 加载 XGBoost 模型
  - 创建 TreeExplainer
  - 抽样 100 场训练数据计算 SHAP values
  - 生成两张图:
    a) summary_plot (全局特征重要性, 用 SHAP 而非 gain)
    b) waterfall_plot (单场预测解释示例)
  - 保存到 /app/ml/outputs/shap/
```

核心代码框架:
```python
import shap
import matplotlib.pyplot as plt
from ml.src.predictor import load_predictor
from ml.src.features.feature_builder import FEATURE_NAMES

predictor = load_predictor()
explainer = shap.TreeExplainer(predictor.xgb_trainer.win_model)

# 抽样数据
X_sample = shap.sample(X_train, 100)

# 全局 summary
shap.summary_plot(explainer(X_sample), X_sample, feature_names=FEATURE_NAMES)

# 单场 waterfall
shap.waterfall_plot(explainer(X_sample[0]))
```

#### 1b. 在 `predictor.py` 中增加 SHAP 解释方法

```python
def explain_prediction(self, X: np.ndarray) -> dict:
    """返回 Top 5 影响最大的特征及贡献值"""
    explainer = self._get_shap_explainer()
    shap_values = explainer(X)  # 单行
    # 按绝对值排序取 Top 5
    ...
    return {
        "top_features": [
            {"feature": "home_avg_net_rtg", "contribution": +0.08, "direction": "home_win"},
            ...
        ],
        "base_value": 0.50
    }
```

#### 1c. 后端 API 扩展

在 `POST /api/v1/predictions` 响应中增加 `explanation` 字段:

```json
{
  "explanation": {
    "top_factors": [
      {"feature": "home_win_rate_last5", "display": "主队近5场胜率 80%", "impact": +0.06, "direction": "home"},
      {"feature": "away_avg_def_rtg", "display": "客队防守效率联盟第2", "impact": -0.04, "direction": "away"},
      {"feature": "rest_days_diff", "display": "主队多休息2天", "impact": +0.03, "direction": "home"}
    ],
    "base_probability": 0.50
  }
}
```

需要创建中文 display name 映射表 `FEATURE_DISPLAY_NAMES`。

#### 1d. 前端 PredictionCard 组件更新

在预测结果卡片上增加一个可折叠的 "为什么？" 区域，用简单的 ↑↓ 箭头展示 Top 3 影响因素:

```
为什么湖人 62% 胜率？
  ↑ +6%  主队近5场胜率 80%
  ↓ -4%  客队防守效率联盟第2
  ↑ +3%  主队多休息2天
```

#### 预估工作量: 1-1.5 小时

---

### Task 2: 球员级别特征扩展 (P2 #10)

**目标**: 在现有 7 个球员特征基础上，增加位置匹配、板凳深度等维度。

#### 2a. 位置匹配特征 (`position_match_features.py`)

```
新增特征 (约 8 个):
  - home_frontcourt_ppg / away_frontcourt_ppg  (前场得分: SF+PF+C)
  - home_backcourt_ppg / away_backcourt_ppg    (后场得分: PG+SG)
  - frontcourt_scoring_diff                     (前场得分差)
  - backcourt_scoring_diff                      (后场得分差)
  - home_size_advantage                         (主队前场身高优势)
```

实现要点:
- `nba_api` 的 `players` 表中有 `position` 字段
- 前场: SF, PF, C
- 后场: PG, SG
- 需要取最近 N 场活跃球员的数据

#### 2b. 板凳深度特征

```
新增特征 (约 4 个):
  - home_bench_scoring_share / away_bench_scoring_share  (替补得分占比)
  - home_bench_net_rtg / away_bench_net_rtg              (替补阵容净效率)
```

实现要点:
- 从 `game_player_stats` 中筛选非首发球员（`starter_position` IS NULL 或 `minutes_played < 25`）
- 计算替补球员的得分占比和简单的 +/- 效率

#### 2c. 阵容稳定性特征

```
新增特征 (约 2 个):
  - home_lineup_stability / away_lineup_stability
    (近5场首发阵容变化程度: 0=完全相同, 1=完全不同的5人)
```

#### 集成

- 新增 `position_match_features.py` 和 `bench_features.py`
- 在 `feature_builder.py` 中调用新模块
- 更新 `FEATURE_NAMES`（当前 110 → 预计 ~124）
- **训练验证**: 跑一轮训练确认新特征提升了指标
- 新特征组预期贡献: 5-8% importance

#### 预估工作量: 1.5-2 小时

---

### Task 3: API 集成测试修复 (P2 #9)

**当前状态**: 39 tests, 37 pass, 2 个已知问题。

#### 3a. 修复 `test_admin_train_model` 挂起

问题: 测试触发真实训练（Docker 中 ~20 秒），超时挂起。

修复方案: Mock `PredictionService._get_predictor` 或使用 `unittest.mock.patch` 拦截训练调用:

```python
@pytest.mark.anyio
async def test_admin_train_model():
    with patch("app.services.prediction_service.PredictionService._get_predictor"):
        response = await client.post("/api/v1/admin/train-model")
        assert response.status_code == 202  # 训练已触发
```

#### 3b. 修复 `test_list_upcoming_games` 日期敏感

问题: 硬编码日期在休赛期无数据。

修复方案: 改为动态查询：

```python
# 先获取任意一场比赛日期
games = await client.get("/api/v1/games?limit=1")
# 用返回的日期范围查询 upcoming
```

或直接断言 "返回空列表也是合法的":

```python
response = await client.get("/api/v1/games/upcoming")
assert response.status_code == 200
# 休赛期返回空列表是正常行为
assert isinstance(response.json(), list)
```

#### 3c. 确认 `test_predict_unavailable_when_no_model` 行为

问题: 测试期望 "模型不可用"，但 ML 模块在 Docker 中实际可用。

修复: 更新测试预期 — 如果模型已加载，测试应验证预测成功返回。

#### 预估工作量: 0.5-1 小时

---

### Task 4: 预测推送通知 (P2 #13)

**目标**: 新赛季每天比赛前自动生成预测。

#### 4a. 后端自动预测脚本

创建 `backend/app/services/auto_predictor.py`:

```python
async def generate_daily_predictions():
    """获取今天所有比赛，逐一生成预测"""
    today_games = await get_today_games()
    for game in today_games:
        prediction = await predictor.predict(game.home_team_id, game.away_team_id)
        await save_prediction(prediction)
    return len(today_games)
```

#### 4b. APScheduler 定时任务

在 `scheduler.py` 中新增一个 daily job:

```python
# 每天早上 6:00 (北京时间，NBA 比赛通常在美东时间晚上 → 北京时间次日上午)
scheduler.add_job(
    generate_daily_predictions,
    trigger=CronTrigger(hour=6, minute=17),
    id="daily_prediction"
)
```

#### 4c. 前端通知入口

在 Navigation 中增加一个 🔔 图标，点击下拉显示今日预测摘要：

```
🔔 今日 3 场比赛已预测
   └─ 湖人 vs 凯尔特人 — 主胜 62%
   └─ 勇士 vs 篮网 — 客胜 55%
   └─ 雄鹿 vs 热火 — 主胜 71%
```

#### 4d. 管理页控制开关

在 AdminPage 增加:
- 自动预测开关 (on/off)
- 预测时间设置 (默认 06:00)
- 手动触发 "立即为今日比赛生成预测"

#### 预估工作量: 1-1.5 小时

---

## 执行优先级矩阵

| 优先级 | 任务 | 理由 |
|--------|------|------|
| ⭐ 1st | SHAP 可解释性 | 用户最感兴趣，可视化效果好，可立刻在前端看到效果 |
| 2nd | 球员特征扩展 | 最有突破潜力的特征方向，可能显著改善 Spread/Total MAE |
| 3rd | API 测试修复 | 快速修复，低风险 |
| 4th | 预测推送通知 | 依赖实时赛程数据，休赛期难以完整测试 |

---

## 预期产出

| 任务 | 产出 |
|------|------|
| SHAP | `ml/src/shap_analysis.py`, `predictor.py` 中 `explain_prediction()`, 前端瀑布图/总结图, `FEATURE_DISPLAY_NAMES` 映射表 |
| 球员特征 | `position_match_features.py`, `bench_features.py`, 110→~124 维, 新模型训练验证 |
| 测试修复 | 39→41 tests, 40+ pass, 无 hang |
| 推送通知 | `auto_predictor.py`, scheduler daily job, 前端 🔔 入口, 管理页开关 |

---

## 启动命令（复制粘贴）

```bash
# 启动环境
docker compose up -d

# 确认运行
docker compose ps

# 如需重训练（有新特征时）
docker compose --profile trainer run --rm trainer \
    --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
    --lookback 15

# 运行测试
docker compose exec backend python -m pytest backend/tests/ -v

# 运行 SHAP 分析
docker compose --profile trainer run --rm --entrypoint python trainer \
    -m ml.src.shap_analysis
```

---

## 快速参考: 关键文件路径

```
ml/src/
├── features/
│   ├── feature_builder.py       ← 特征编排主入口
│   ├── team_features.py         ← 球队特征 (28个)
│   ├── player_form_features.py  ← 球员状态特征 (7个)
│   ├── matchup_features.py      ← 对位特征 (10个)
│   ├── h2h_features.py          ← 历史对战特征
│   └── context_features.py      ← 比赛情境特征
├── predictor.py                 ← 推理服务 (需增加 explain 方法)
├── trainer.py                   ← 训练流程
└── shap_analysis.py             ← NEW: SHAP 分析脚本

backend/
├── app/
│   ├── api/predictions.py       ← 预测 API (需扩展 explanation 字段)
│   ├── services/
│   │   ├── prediction_service.py ← 预测服务
│   │   └── auto_predictor.py    ← NEW: 自动预测
│   └── scheduler/               ← 定时任务
└── tests/test_api.py            ← 集成测试

frontend/src/
├── pages/HomePage.tsx           ← 预测主页
├── components/
│   ├── PredictionCard.tsx        ← 预测卡片 (需增加 SHAP 解释区域)
│   └── charts/                  ← 图表组件
└── api/predictions.ts           ← API 客户端
```
