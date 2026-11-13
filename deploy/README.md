# 生产镜像来源

此目录保存已用过的修复镜像分层 Dockerfile，供核对覆盖范围与恢复环境。`Dockerfile.mlp-api` 只增加观察状态转接和采集调度隔离；推理修复来自兼容层。

这些覆盖镜像依赖本机已保存的父镜像，生产 Compose 按实际验收 digest 固定。GitHub／Gitee 保存源码与小型模型包，不保存约 1 GB 的 Docker 镜像；全新机器需要先构建或恢复基础镜像及冻结模型卷，不能假定这些本机 digest 已在公共镜像仓库发布。

模型恢复说明见 `releases/mlp_input_repair_20261112_release_v1/README.md`；生产编排与回滚见 `docs/fixed-mlp-production-deployment-2026-11-12.md`。不要在新机器上为了启动而改写旧观察协议或跳过文件哈希检查。
