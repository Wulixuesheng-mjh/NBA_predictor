# 脚本入口分类

现有脚本被测试、镜像或历史协议按 `scripts.*` 导入，保持路径稳定。通过本索引区分入口，不为目录美观批量搬动已发布脚本。

| 用途 | 主要入口 |
| --- | --- |
| 生产服务启停 | `production_stack.ps1` |
| 固定包恢复 | `restore_release_artifacts.py` |
| 固定 MLP 发布／回退 | `fixed_mlp_release.py` |
| 原研究模型通用准入 | `promote_model_release.py` |
| 原总分独立观察器 | `legacy_total_observer.py` |
| 胜负／旧双模型／修复安全观察 | `win_shadow_worker.py`、`paired_models_worker.py`、`repaired_mlp_safety_worker.py` |
| 数据采集与补采 | `seed_data.py`、`refresh_current_results.py`、`backfill_box_scores.py`、`import_verified_injuries.py` |
| 训练与滚动研究 | `train_models.py`、`run_rolling_evaluation.py`、`run_model_experiments.py` |
| 真实输入重放 | `replay_prediction.py` |
| 仓库结构检查 | `check_repository_layout.py` |

其余 `audit_*`、`inspect_*`、`probe_*`、`verify_*`、`preflight_*`、`accept_*` 为取证或验收工具；部分需要特定本地资料、模型版本或专用测试库。不要将它们批量运行当作生产启动流程。

SQL 文件属于数据初始化／维护工具；手动执行前核对目标环境。单元测试位于 `backend/tests/`，与依赖特定历史数据的验收脚本分开。
