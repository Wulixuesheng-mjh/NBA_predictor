# Windows 本机维护工具

`legacy/` 保存原仓库的历史 Docker／WSL 磁盘维护脚本，只调整目录，没有执行或重新验收。它们不属于 NBA 应用部署入口。

| 文件 | 原用途 |
| --- | --- |
| `legacy/compact_docker.ps1` | 停止 Docker／WSL，整理并压缩本机虚拟磁盘 |
| `legacy/move_docker_to_e.ps1` | 停止 Docker／WSL，将本机 Docker 数据迁移到 E 盘 |

脚本包含旧版环境假设和本机磁盘操作，不应由 CI、模型训练或生产启动脚本调用。目录整理不代表这些操作已对当前环境验证。

原 `_diskpart_compact.txt` 含用户绝对路径，已移入忽略的 `notes/local/`，保留本地原件。仓库只提供不含真实用户路径的命令示例。不要直接将示例当成当前磁盘操作指令。
