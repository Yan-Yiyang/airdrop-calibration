# airdrop-calibration AI 执行指南

先核对任务与工作区改动。Python ≥3.14，使用 uv；安装依赖或工具前取得用户同意。注释和 docstring 使用中文，文本 LF，提交使用 conventional 前缀与中文摘要。

- `airdrop_calibration/cli.py` 为唯一命令行解析入口；各工作流暴露 `build_config`、`main`。
- 工具不得导入主仓 `airdrop`、任务控制器、YOLO 或 RapidOCR。
- 保留文件契约与默认数学/布局；调整算法须有独立数值期望或合成回归。
- 不修改 PX4/QGC 源码；脚本只操作独立运行目录。生成资产只在任务要求重生成时更新。
- 验证：`python -m pytest`、`ruff check .`、`ruff format --check .`、`pyright`。无 WSL/ext4 时明确标记未验证脚本。
- 数据契约见 `docs/contracts.md`；迁移历史见 `MIGRATION.md`。
