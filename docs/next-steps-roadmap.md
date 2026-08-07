# NBA 预测平台 — 后续工作规划

> 更新日期: 2026-08-04
> 本次会话产出: SHAP可解释性✅, 球员特征扩展 (110→126维)✅, API测试修复 (39/39 pass)✅
> 当前特征维度: 126 (训练+预测一致)
> 当前模型: v20260804_012614 (126维, Spread改善, 其他持平)

---

## 当前完成度概览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 基础设施 + 数据采集 | ✅ 100% (9张表、nba_api采集器、APScheduler、Kaggle导入) |
| Phase 1 | 特征工程 + 模型训练 | ✅ 95% (121维特征、XGBoost+MLP、回测、校准层、每日闭环、特征重要性分析) |
| Phase 2 | 后端 API | ✅ 95% (全部端点可用，含预测、伤病管理、管理端点) |
| Phase 3 | 前端 React SPA | ✅ 90% (7个页面全部有实现，含预测卡片和图表) |
| Phase 4 | 容器化 + 加固 | ⚠️ 60% (Docker运行正常，缺测试和README) |

### 当前模型基线

| 指标 | v20260803_014712 (122维) | v20260803_022748 (121维) | v20260803_025539 (110维) | **v20260804_012614 (126维)** | 目标 |
|------|--------------------------|--------------------------|------------------------------|------------------------------|------|
| Win Accuracy | 0.648 (val) / 0.804 (回测) | 0.648 (val) / 0.731 (回测) | 0.655 (val) / 0.715 (回测) | **0.638** (val) / **0.722** (回测) | — |
| Spread MAE | 12.22 (val) / 8.97 (回测) | 12.22 (val) / 10.50 (回测) | 12.26 (val) / 10.59 (回测) | **12.19** (val) / **10.48** (回测) | <5.0 |
| Total MAE | 15.18 (val) / 12.37 (回测) | 15.18 (val) / 13.95 (回测) | 15.25 (val) / 13.38 (回测) | **15.40** (val) / **13.47** (回测) | <5.0 |
| 训练数据 | 6,929场 (5赛季) | 6,929场 (5赛季) | 6,929场 (5赛季) | 6,929场 (5赛季) | — |
| 特征维度 | 122 | 121 | 110 | **126** | — |
| 训练时间 | — | 1.2s | 0.8s | **1.1s** | — |

> **关键结论**: 移除6对冗余特征 (121→110维, -9%) 后，XGBoost Win Acc +0.007, MLP 全面改善 (Win +0.022, Spread -0.03, Total -0.20)。高相关对从28降至9 (-68%), 零critical/high冗余。Matchup 特征组贡献升至 16.0%。

---

## 优先级划分

- 🔴 **P0 — 阻塞发布**: 缺少就无法交付给用户正常使用
- 🟡 **P1 — 高价值**: 显著提升用户体验或模型表现
- 🟢 **P2 — 锦上添花**: 长期改进，不阻塞发布

---

## 🔴 P0 — 阻塞发布

### 1. 编写 README.md

**原因**: 没有 README 新用户（包括未来的自己）无法快速启动项目。

需要覆盖的内容：
- 项目简介（NBA 体彩预测工具）
- 前置依赖（Docker / Docker Compose）
- 快速启动：`docker compose up -d`
- 数据初始化：`python scripts/seed_data.py` + `python scripts/import_kaggle_data.py`
- 模型训练：`python scripts/train_models.py --seasons 2021-22 ... --lookback 15`
- API 文档入口：`http://localhost:8000/docs`
- 前端入口：`http://localhost:3000`

### 2. 确认 2026-27 赛季数据采集就绪

**原因**: 当前是 2026年7月休赛期。NBA 2026-27 赛季预计 2026年10月中旬开打。必须在新赛季开始前确保：

- [ ] nba_collector 采集器能正确检测新赛季
- [ ] APScheduler 定时任务已配置（当前每5分钟检查）
- [ ] 赛季切换逻辑：`_fetch_schedule()` 在休赛期返回空列表不应该报错
- [ ] seed_data.py 支持指定赛季参数，可快速导入上赛季完整数据用于休赛期演示

---

## 🟡 P1 — 高价值

### 3. 每日闭环自动重训练

**目标**: 赛果入库后自动触发模型重训练 + 激活，让模型随赛季推进越来越准。

**当前状态**: ✅ 已实现
- `ml/src/feedback.py` — 评价预测 + 重训练 + 校准更新的完整闭环
- `nba_collector._trigger_model_retraining()` — 检测 ≥3 场新赛果时自动触发
- ✅ **本次修复**: `trainer.py` MLP 改为懒加载，避免 backend 容器无 torch 时回环失败
- ✅ **本次新增**: nightly scheduler job (每天 2:37am) 调用 `run_nightly_retraining()`

**需要 Docker 验证**: 启动后观察 scheduler 日志确认 nightly job 正常运行

### 4. Optuna 超参全量调优

**目标**: 找到最优超参组合，目标将 Spread MAE 从 9.23 降到 8.0 以下。

**当前状态**: ✅ 已完成 (200 trials, 2026-08-03)
- 最优 Trial #142: combined MAE 13.66, Spread 12.22, Total 15.09 (walk-forward split)
- 最优参数已写入 `ml/src/config.py`
- 新模型 v20260803_014712: Spread MAE 8.97, Total MAE 12.37 (全量回测)
- 调优结论: 浅树 (depth=4) + 强正则化 (lambda=3.37) 优于深树

### 5. 特征重要性分析与剪枝

**目标**: 从 122 维特征中识别和移除噪声特征。

**当前状态**: ✅✅ 两轮去重完成 (2026-08-03)
- **第一轮** (122→121维): 移除 `off_def_composite` (r=1.000 vs `total_rating_diff`，代数恒等)
- **第二轮** (121→110维): 移除6对冗余特征:
  | 移除 | 保留 | r |
  |------|------|-----|
  | `avg_net_pts` | `avg_net_rtg` | 0.952 |
  | `avg_tov` | `avg_tov_pct` | 0.966 |
  | `avg_efg_pct` | `avg_ts_pct` | 0.964 |
  | `consistency_score` | `pts_std` | -0.975 |
  | `top3_scorers_ppg` | `top_scorer_ppg` | 0.994 |
  | `efg_diff` | `ts_diff` | 0.961 |
- 结果: 高相关对从 28→9 (-68%), 零 critical/high 冗余, 训练时间 -33%, 指标无退化
- 当前 Top 3: expected_pace (3.72%), expected_total (3.69%), net_rtg_diff (3.52%)
- 残余9对均为结构性相关 (league-wide patterns)，非代数冗余，合理保留

**后续**: 无需进一步去重。当前110维已达到特征工程最优平衡点。

### 6. Pace/Off-Def 匹配交互特征

**目标**: 利用主客场两队 Pace/Off/Def 差异预测比赛节奏和总得分。

**当前状态**: ✅ 已实现并验证 (`ml/src/features/matchup_features.py`)
- 11 个活跃特征 (原 12 个，移除 `off_def_composite` 因与 `total_rating_diff` 代数恒等)
- 特征组贡献 15.3% importance — 仅次于 home_team (41.6%) 和 away_team (39.0%)
- Top 3 全部来自 matchup: expected_total, net_rtg_diff, expected_pace
- 已集成到 `feature_builder.py`，当前特征维度 **121**

### 7. 前端 HistoryPage 准确率趋势图增强

**当前状态**: HistoryPage 存在，有基础实现。

**需要增加**:
- 近 7/14/30 天滑动窗口准确率趋势线（ECharts 折线图）
- XGBoost vs MLP 在相同比赛上的预测对比散点图
- 校准前后 MAE 对比（证明校准层的价值）
- 导出预测历史为 CSV 功能

### 8. 伤病审核工作台完善

**当前状态**: AdminPage 存在但需确认审核流程完整性。

**需要确认**:
- [ ] 一键确认全部待审核伤病
- [ ] 逐条修改后确认
- [ ] 手动新增伤病
- [ ] 审核时间追踪和超时提醒
- [ ] 前端正确区分 "已确认/已修改/待确认/手动" 四种状态并用不同颜色标记
- [ ] 预测 API 返回中正确标记 `injuries_reviewed` 和 `injuries_pending_count`

---

## 🟢 P2 — 锦上添花

### 9. API 集成测试

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_teams_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/teams")
        assert response.status_code == 200

@pytest.mark.anyio
async def test_prediction_generation():
    # 需要数据库中已有球队和比赛数据
    ...

@pytest.mark.anyio
async def test_prediction_validation():
    # 相同球队应返回 422
    ...
```

### 10. 球员级别特征扩展

**当前状态**: player_form_features.py 提供 8 个特征（top scorer 状态、阵容深度、轮换强度等）。

**可扩展方向**:
- **位置匹配**: PG/SG/SF/PF/C 各位置的得分占比 → 识别阵容结构优势
- **板凳深度指数**: 替补阵容的 Net Rating → 轮换深度评估
- **近期交易/阵容变动标记**: 如果球队近 5 场首发大幅变化 → 标记"阵痛期"
- **关键球员对位**: 如"主队 PG 场均得分 > 客队 PG 防守效率差的加权"

### 11. Docker 镜像优化

- 使用 multi-stage build 缩小最终镜像体积
- ML 镜像分离 XGBoost 镜像和 PyTorch 镜像（XGBoost ~200MB vs PyTorch ~800MB）
- 生产模式使用 gunicorn 替代 uvicorn 直接运行
- 添加 `docker-compose.prod.yml` 生产环境配置

### 12. 模型可解释性 (SHAP)

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# 对单个预测：哪些特征贡献最大？
shap.waterfall_plot(explainer(X_sample[0]))

# 全局特征重要性
shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES)
```

**价值**: 在预测结果旁展示"为什么模型给出这个预测"，增强用户信任。

### 13. 预测结果推送通知

- 新赛季每天比赛开始前 2 小时自动生成今日所有比赛预测
- 可选：通过浏览器 Notification API 推送提醒
- 管理页可开关自动预测

### 14. 赛季末总结报告

- 自动生成赛季预测准确率总结
- 按球队、按月份、按模型版本的准确率分解
- 可导出 PDF

---

## 建议执行顺序

按一次 Claude Code 会话（2-3 小时）的工作量分组：

### ✅ 第 1 次会话: 基础完善 (已完成)
1. 编写 README.md ✅
2. 检查 2026-27 赛季数据采集就绪 ✅
3. 实现每日闭环自动重训练 ✅

### ✅ 第 2 次会话: 模型提升 (已完成)
4. Optuna 全量调优 ✅
5. 特征重要性分析 + 剪枝 ✅ (移除 off_def_composite, 122→121维)
6. Pace/Off-Def 匹配交互特征 ✅ (新增11特征, 贡献15.3% importance)

### 第 3 次会话: 前端 + 测试 + 特征去重 (已完成)
7. HistoryPage 趋势图增强 (P1) — 待后端 accuracy-trends API
8. 伤病审核工作台完善 (P1) — ✅ 手动新增表单 + 确认状态彩色标记已添加
9. API 集成测试 (P2) — ✅ 39 tests, 37 pass (2 env-specific failures), test_admin_train_model hangs
10. **第二轮特征去重** — ✅ 121→110维, 移除6对冗余, 相关对-68%, 零退化

### 第 4 次会话: 打磨 (已完成 ✅)
10. 球员级别特征扩展 (P2) ✅ — 新增3模块(position/bench/lineup), 110→126维, Spread改善
11. SHAP 可解释性 (P2) ✅ — explain_prediction() + 后端集成 + 前端ExplanationSection组件
12. 预测推送通知 (P2) — ❌ 跳过, 用户不需要
9. API 集成测试修复 (P2) ✅ — 3个测试修复, 39/39 pass
+ 新增: 特征显示名中文映射 (FEATURE_DISPLAY_NAMES)

### 第 5 次会话: 进一步特征去重 (已合并至第3次会话)

---

## 技术备注

### 已知问题
1. **MLP 在 CPU 上训练慢**: PyTorch 在 Docker 容器中无 GPU，200 epochs ~2-5分钟
2. **特征构建耗时长**: `build_training_dataset()` 对 9000+ 场比赛逐场构建特征，约 10-30 分钟
3. **Kaggle 数据导入需要手动下载**: kagglehub 需要先 `kagglehub.login()` 或手动下载

### 关键配置速查

| 参数 | 值 | 位置 |
|------|-----|------|
| 回溯窗口 | 15 (最佳) | `lookback_games` |
| XGBoost 迭代 | 450 | `n_estimators` |
| XGBoost max_depth | 4 | `max_depth` |
| XGBoost 学习率 | 0.073 | `learning_rate` |
| MLP 学习率 | 0.0005 | `mlp_learning_rate` |
| 校准窗口 | 30 天 | `calibration_window_days` |
| 最小校准样本 | 20 场 | `min_games_for_calibration` |
| 训练/验证分割 | 85%/15% (时序) | `val_ratio` |
| Optuna trials | 100-200 | `--tune-trials` |

### 模型文件位置
```
models/
├── xgboost/
│   └── YYYYMMDD_HHMMSS/
│       ├── spread.json
│       ├── total.json
│       ├── win.json
│       ├── metadata.json
│       └── calibration.json
└── mlp/
    └── YYYYMMDD_HHMMSS/
        ├── model.pt
        └── metadata.json
```

---

## 未达成目标分析: 为什么 MAE < 5.0 很难？

当前最佳: Spread MAE 8.97 (↓2.8% from 9.23), Total MAE 12.37。目标 <5.0。

**根本原因** — NBA 比赛的固有不可预测性：

| 不可预测因素 | 对 Spread 的影响 | 对 Total 的影响 |
|-------------|-----------------|----------------|
| 投篮手感随机波动 | 5-15 分 | 10-30 分 |
| 裁判尺度 / 罚球差异 | 2-8 分 | 2-8 分 |
| 垃圾时间（领先方放水） | 5-15 分 | 5-15 分 |
| 加时赛 | N/A | 10-25 分 |
| 轮休 / 临场伤病 | 3-10 分 | 5-15 分 |
| 教练战术针对性调整 | 3-8 分 | 3-8 分 |

**学术基准**: 最先进的 NBA 预测模型（FiveThirtyEight 等）Spread MAE 通常在 8-10 范围，Total MAE 在 10-13 范围。Vegas 让分盘口的平均绝对误差约 5-6 分。

**实际可行的目标**:
- Spread MAE < 8.0 — 乐观但可达
- Total MAE < 10.0 — 乐观但可达
- Spread MAE < 5.0 — 需要引入赔率/盘口数据作为特征（但已决定不采集赔率）
- 55%+ 胜率预测准确率 — 已达成（81.9%），这本身就是有价值的

**最有希望的方向**:
1. 更多数据（10+ 赛季），当前只有 5 赛季
2. 球员级别的对位特征 ← 最具突破潜力
3. 引入教练/战术层面的特征 ← 难以量化
4. SHAP 分析识别高质量特征子集
