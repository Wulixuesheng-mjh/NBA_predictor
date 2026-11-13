# 运行入口与环境边界

## 已验收的现有生产实例

从仓库根目录运行：

```powershell
./scripts/production_stack.ps1 ps
./scripts/production_stack.ps1 up
```

该入口组合六份 Compose 配置，保留冻结旧观察器、主 API 调度隔离和动态 DNS 前端。它不重新训练、不重新构建，也不自动激活模型。

现有发布配置依赖本机已验收镜像及外部冻结模型卷。全新电脑只有源码和模型包时，不能直接把本机镜像 digest 当作公共镜像地址。迁移与回滚按[部署记录](fixed-mlp-production-deployment-2026-11-12.md)执行。

## 开发环境

```text
cp .env.example .env
docker compose -f docker-compose.yml up -d --build
```

显式使用基础配置避免误加载个人 `docker-compose.override.yml`。填写本地数据库与管理员配置；真实 `.env` 不提交。开发实例和现有生产实例使用相同默认容器／卷名称，不能直接并行启动来当隔离测试环境。

模型与比赛数据需要单独准备。缺少模型时启动应用不等于可以生成预测；训练不会自动成为已验收发布版本。训练参考[训练指南](ML_TRAINING_GUIDE.md)和当前 `scripts/train_models.py --help`。

## 固定模型包恢复

```text
python scripts/restore_release_artifacts.py --bundle releases/mlp_input_repair_20261112_release_v1/bundle.json --models-dir MODEL_ROOT
```

只恢复哈希通过的文件，不覆盖不同文件、不激活。发布仍需兼容代码、数据库迁移和独立验收，不能把恢复文件当成晋升。

## 本地内容边界

- `notes/local/`：个人笔记和本机路径信息，保留本地并忽略。
- `tmp/`：临时下载、调试和构建材料，忽略。
- `reports/`：默认忽略新增内容，仅提交已明确发布的证据清单。
- `ml/saved_models/`：运行模型与缓存；受控固定包放在 `releases/`。
- `tools/windows/legacy/`：历史磁盘维护脚本，不是应用启动步骤。
