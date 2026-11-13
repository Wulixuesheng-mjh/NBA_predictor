# 1022 MLP 固定输入修复发布包

这是 2026-11-12 已部署版本的不可变备份，MLP 权重不变；修复由 `ml/src/legacy_mlp_compatibility.py` 在原缩放器前执行。行为版本为 `20261022_050952+legacy_mlp_1022_constant_ast_v1`。

`artifacts.zip` 约 1.33 MiB，包含原 XGB／MLP 权重、缩放器、ONNX 外部数据及修复包清单和证据。`bundle.json` 保存压缩包及每个成员的 SHA256。无数据库、训练缓存、Docker 镜像、私密配置或自动激活文件。日常训练输出继续忽略；本次按跨站备份要求，显式版本化这一小型固定包，无需 Git LFS。

从仓库根目录恢复到指定模型目录（仅恢复文件，不覆盖不同文件、不激活）：

```text
python scripts/restore_release_artifacts.py --bundle releases/mlp_input_repair_20261112_release_v1/bundle.json --models-dir MODEL_ROOT
```

正式激活仍需兼容运行代码、数据库 010 和专用发布校验。新电脑没有本机 Docker 镜像和冻结卷，不能直接运行按本机 digest 固定的生产覆盖配置；请先按源码构建／恢复运行环境，并参考部署报告安排旧观察器隔离。

只恢复这一包不会安装未晋升的研究影子或历史数据集。详见 `docs/fixed-mlp-production-deployment-2026-11-12.md`。
