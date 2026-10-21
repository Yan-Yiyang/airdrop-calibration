# 仓库边界与历史

本工具按职责从任务主仓分离，库代码不导入主仓`airdrop`。迁移前后文件历史可用`git log --follow -- <文件>`追踪；不要依赖过滤前仓库的提交ID或旧目录路径。

命令行参数由`airdrop_calibration/cli.py`统一解析，数据格式与坐标/时间语义见[契约](docs/contracts.md)。
