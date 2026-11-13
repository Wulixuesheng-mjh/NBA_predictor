# NBA Predictor

NBA 赛事预测与评估平台，提供胜负概率、主队净胜分和总分预测，保留数据质量、模型版本与预测输入记录。

## 当前发布状态

- XGBoost 为主模型，权重版本 `20261022_050952`。
- MLP 已启用固定输入修复，行为版本为 `20261022_050952+legacy_mlp_1022_constant_ast_v1`。
- 当前发布仍为 **166 维**；代码中的 213 维特征与研究影子未一并晋升。
- 历史重建与真实赛前成绩分开记录。统计／伤病缺口继续显示质量提醒，训练不会自动晋升模型。

详情见[生产部署与回滚记录](docs/fixed-mlp-production-deployment-2026-11-12.md)和[版本变更](CHANGELOG.md)。

## 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 查看现有生产实例 | `./scripts/production_stack.ps1 ps` |
| 启动现有生产实例 | `./scripts/production_stack.ps1 up` |
| 新环境准备、开发运行 | [运行指南](docs/operations.md) |
| 恢复已发布模型文件 | [固定发布包](releases/mlp_input_repair_20261112_release_v1/README.md) |
| 理解训练、测评和历史结论 | [文档导航](docs/README.md) |
| 查找脚本 | [脚本索引](scripts/README.md) |
| 提交代码 | [工程规范](CONTRIBUTING.md) |

生产入口依赖本机已验收镜像与冻结模型卷，不能把克隆仓库等同于完成生产恢复。请按运行指南选择环境，避免省略旧观察器隔离配置。

默认访问：前端 `http://localhost:3000`，API 文档 `http://localhost:8000/api/docs`，后端健康 `http://localhost:8000/health`。

## 仓库结构

| 目录 | 职责 |
| --- | --- |
| `backend/` | FastAPI、ORM、迁移、数据采集、预测服务及单元测试 |
| `frontend/` | React／TypeScript 页面与 Nginx 代理配置 |
| `ml/` | 特征构建、训练、推理、行为兼容与误差记录 |
| `scripts/` | 稳定的运行、训练、研究和验收入口 |
| `deploy/` | 镜像覆盖 Dockerfile 与环境恢复边界 |
| `releases/` | 小型固定模型包、成员哈希与恢复说明 |
| `reports/` | 明确发布的评估／验收证据，新增报告默认忽略 |
| `docs/` | 当前运行指南、文档索引及保留原路径的历史报告 |
| `tools/windows/` | 历史本机磁盘维护工具，不参与应用启动 |

根目录保留项目说明、环境模板、Git 配置和 Compose 编排。实际环境文件、个人笔记、缓存、镜像备份和临时资料不提交。

## 验证与维护

后端测试位于 `backend/tests/`，前端生产构建为 `npm run build`。发布工具还需检查版本、哈希、重放与回滚；不要用数据库备份或本机调试脚本代替测试。

整理或发布前可运行标准库检查器：

```text
python scripts/check_repository_layout.py
```

它检查被跟踪文件的目录边界、入口链接及固定发布包完整性，不访问数据库、不启动模型，也不执行磁盘维护脚本。
