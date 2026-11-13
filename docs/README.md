# 文档导航

按用途选择文档。带日期的报告记录当时的结论，不能覆盖较新的生产状态；历史报告保留原路径和内容，便于核对已发布证据。

## 当前运行与恢复

- [运行入口](operations.md)：开发环境、现有生产实例、模型恢复及目录边界。
- [已部署版本与真实验收](fixed-mlp-production-deployment-2026-11-12.md)：当前 166 维修复版 MLP、旧观察器隔离及回滚命令。
- [发布与回滚验收](fixed-mlp-release-acceptance-2026-11-12.md)：包校验、版本字段、原子切换和兼容性证据。
- [固定模型包](../releases/mlp_input_repair_20261112_release_v1/README.md)：归档内容、哈希与恢复方法。
- [镜像来源](../deploy/README.md)：本机覆盖镜像的依赖边界。

## 研究结论与数据质量

- [本赛季上线前测评](current-season-prelaunch-review-2026-11-12.md)：156 场重建、修复收益、影子限制。
- [MLP 根因与固定修复](model-mlp-constant-repair-2026-11-10.md)。
- [总分根因取证](model-total-root-cause-findings-2026-11-09.md)。
- [伤病绑定与留档](injury-chain-and-mlp-safety-2026-11-12.md)。
- [V2 历史重评](model-v2-reevaluation-2026-11-10.md)：具名对照与准入边界。
- [训练指南](ML_TRAINING_GUIDE.md)：训练工作流参考；参数以当前脚本 `--help` 为准。

其余 `checkpoint`、`progress`、`study`、`session` 和带日期的工作日志是历史材料；原始数据和完整缓存不随仓库分发。

## 仓库规范

- [贡献与提交规范](../CONTRIBUTING.md)。
- [脚本入口分类](../scripts/README.md)。
- [已发布证据范围](../reports/README.md)。
- [仓库归档范围](repository-publication-2026-11-13.md)。
- [Windows 本机工具](../tools/windows/README.md)。
