# airdrop-calibration

独立的内参、画面/遥测时间差、手眼外参离线标定。只需 Python ≥3.14、NumPy 与 OpenCV，不导入主仓任务包。SciPy 仅供合成测试；依赖用 uv 管理。

```bash
uv sync
uv run python -m airdrop_calibration --help
uv run python -m airdrop_calibration calibrate --flight <飞行目录> --out camera_calib.json --strict
```

已有满足依赖的Python环境时，可在独立克隆目录直接运行`python -m airdrop_calibration --help`，无需重复安装。

唯一命令行解析入口为 `airdrop_calibration/cli.py`，安装后也可用 `airdrop-calibration`。Python 工作流入口为 `airdrop_calibration.calibrate`，DTO、算法、读取器和报告分别位于 `models`、`algorithms`、`flight_log`、`report`；库入口保留原函数名与尺量参数覆盖语义。

标准飞行目录由主仓录制器/标定采集生成，离线工具读取 `frames_index.jsonl`、`frames/` 和 `telemetry.jsonl`，输出读兼容的 `camera_calib.json`。字段、单位、时间轴与路径校验见 [契约](docs/contracts.md)，数学与 OpenCV 说明见 [专题](docs/calibration_opencv.md)。默认使用 received 时间轴，缺失测量不补零；模型推导文件不当作实测标定。

主仓专用采集入口为 `python -m airdrop.run mission --mode real --no-mission-control --no-recognize --record --calibration-capture`；在地面手持飞机绕多轴拍固定标靶即可采集内外参与视频延迟，`real` 表示真机输入而非起飞。命令开关选择被动采集、关闭识别并启用记录，profile 逐更新记录遥测且不要求位置数据。PX4 链路预算仍可能降低实际频率，且本进程统计不覆盖 MAVSDK 服务端队列丢弃；采集后检查该服务的 `User callback queue overflown` 日志。`--max-frames` 限制离线分析帧数，不限制记录器保存的采集时段。

```bash
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

测试使用合成几何和手写文件，不能代替真实素材精度验收；运行时不需要 SciPy、YOLO 或 RapidOCR。迁移历史见 [MIGRATION](MIGRATION.md)。
