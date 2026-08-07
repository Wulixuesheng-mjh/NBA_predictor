---
name: 2026-08-04-work-summary
description: Docker 镜像优化 + 磁盘清理 + 镜像拉取问题诊断，含明天行动方案
metadata:
  type: project
---

# 2026-08-04 工作总结

## 完成事项

### 1. Trainer 镜像构建成功 ✅
- **修改**: `ml/Dockerfile` — 将 PyTorch 下载源从 `download.pytorch.org` 切换到清华 PyPI 镜像
- **结果**: 526 MB PyTorch 以 7.4 MB/s 下载成功 (之前 24 kB/s，反复超时)
- **新镜像**: `nba-ml-trainer:latest` 2.95 GB (旧 3.18 GB，缩减 230 MB / 7.2%)
- Docker daemon 在最后 unpacking 阶段崩溃，但镜像已写入

### 2. C 盘空间清理 ✅
- **初始状态**: C 盘仅 406 MB 剩余
- **清理步骤**:
  | 项目 | 空间 |
  |------|:----:|
  | pip 缓存 | 1.1 GB |
  | npm 缓存 | 336 MB |
  | 用户临时文件 | ~1 GB |
  | Docker 构建缓存 (`builder prune -af`) | 33 GB |
  | 旧 trainer 镜像 (3.18 GB) | 3.18 GB |
  | Docker WSL 虚拟磁盘 (VHDX) | 53 GB |
- **最终状态**: C 盘 57 GB 剩余

### 3. Docker 镜像加速配置 ✅
- 配置 `daemon.json` 添加国内镜像源:
  - `https://docker.1ms.run`
  - `https://docker.xuanyuan.me`
  - `https://docker.1panel.live`
  - `https://docker.m.daocloud.io`

---

## 未完成事项

### Backend + Trainer 镜像重建失败 ⚠️
- **根因**: 删除 VHDX 时一并清除了所有缓存的 base 镜像 (`python:3.12-slim`, `node:20-alpine`, `nginx:alpine`)。Docker Hub 从国内直连被墙，镜像加速器虽然配置了但速度极不稳定

---

## 镜像拉取失败根因分析

```
                   ┌─────────────────────────────┐
                   │   Docker Hub (registry-1)   │
                   │   国内直连 → 被墙/超时       │
                   └──────────────┬──────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │   国内镜像加速器             │
                   │   docker.1ms.run etc.       │
                   │   - 不稳定 (限速/429/超时)   │
                   │   - 各地运营商差异大         │
                   └─────────────────────────────┘
```

**核心矛盾**: 国内网络环境下 Docker Hub 不可直连，第三方镜像加速器不稳定。高质量方案需要自建或使用云厂商服务。

**为什么昨天能构建**:
- 之前的 VHDX 保留了 `python:3.12-slim` 等基础镜像的缓存层
- 基础镜像层不需要重新 pull，pip 包通过清华源下载很快
- 删除 VHDX 后所有缓存丢失，需要重新从零拉取

---

## 明天行动方案

### 方案 A: 直接通过镜像站拉取基础镜像 (推荐先试)
```bash
# 直接用完整路径从镜像站拉取，绕过 registry-mirrors 机制
docker pull docker.1ms.run/library/python:3.12-slim
docker tag docker.1ms.run/library/python:3.12-slim python:3.12-slim

docker pull docker.1ms.run/library/node:20-alpine
docker tag docker.1ms.run/library/node:20-alpine node:20-alpine

docker pull docker.1ms.run/library/nginx:alpine
docker tag docker.1ms.run/library/nginx:alpine nginx:alpine

docker pull docker.1ms.run/library/postgres:16-alpine
docker tag docker.1ms.run/library/postgres:16-alpine postgres:16-alpine
```
**优点**: 精确控制，绕过 daemon 层面的 mirror 回退机制
**缺点**: 需要手动 tag

### 方案 B: 自建本地 Registry 缓存
```bash
# 在能访问 Docker Hub 的机器上:
docker pull python:3.12-slim
docker tag python:3.12-slim localhost:5000/python:3.12-slim
docker push localhost:5000/python:3.12-slim
```

### 方案 C: 使用代理
- 在 Docker Desktop Settings → Resources → Proxies 中配置 HTTP/HTTPS 代理
- 手动设置 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量

### 方案 D: 离线导入 (备选)
- 从能正常拉取的机器上 `docker save` 导出镜像为 tar
- 在本机 `docker load` 导入

### 基础镜像准备就绪后
```bash
# 启动 postgres
docker compose up -d postgres

# 构建 backend (pip 包通过清华源 → 快)
docker compose build backend

# 构建 trainer
docker build -f ml/Dockerfile -t nba-ml-trainer .

# 启动所有服务
docker compose up -d
```

**Why**: Docker Hub 网络不可达 → 镜像代理不稳定 → 需要多种备选方案。基础镜像拉取是整个构建链的瓶颈，一旦解决，后续 pip 安装通过清华源很快。
**How to apply**: 明天首先尝试方案 A，如果 docker.1ms.run 不可用则依次尝试 xuanyuan.me、1panel.live、daocloud.io。确认基础镜像就绪后再执行 `docker compose build`。
