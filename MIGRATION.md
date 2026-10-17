# 历史来源

从主仓提交 `eff0c0f180092b3ee0fd0c625c7734156b539109` 按职责路径使用 Git fast-export/fast-import 导入历史，再在当前提交移动到 `airdrop_calibration/`。相关提交作者、时间和消息保留；由于过滤目录，历史提交 ID 会变化。可用 `git log --follow -- airdrop_calibration/` 中的具体文件追踪迁移前路径。

命令行参数只由 `airdrop_calibration/cli.py` 解析；工具不导入主任务包。
