"""工具包唯一命令行入口；参数默认值取自相应工作流。"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Option:
    """一个命令行选项 → 传给入口模块 ``main(**kwargs)`` 的关键字。

    ``source`` 是入口模块里的常量名：``--help`` 里显示的默认值就是它的当前值，
    所以默认值只在模块里写一遍（这里不重复）。
    """

    flags: tuple[str, ...]
    dest: str
    help: str
    #: str / int / float / path / choice / flag / flag-off / bool / pair / vec2 / vec3 / const
    kind: str = "str"
    choices: tuple[str, ...] = ()
    #: 默认值取自入口模块的这个常量（``const`` 用它作为"开关打开时要传的值"）
    source: str = ""


@dataclass(frozen=True, slots=True)
class Subcommand:
    """一个子命令：入口模块 + 选项表。"""

    name: str
    module: str
    help: str
    options: tuple[Option, ...] = ()
    #: 入口模块里干活的函数（签名形如 ``main(**kwargs) -> int``）
    handler: str = "main"


SUBCOMMANDS = {
    "calibrate": Subcommand(
        name="calibrate",
        module="airdrop_calibration.calibrate",
        help="三步相机标定：内参 → 画面/遥测时间差 → 手眼外参",
        options=(
            Option(
                ("--flight",), "flight_dir", "标定采集的飞行目录", kind="path", source="FLIGHT_DIR"
            ),
            Option(
                ("--out",),
                "output_path",
                "标定结果 JSON 的输出路径",
                kind="path",
                source="OUTPUT_PATH",
            ),
            Option(
                ("--pattern",),
                "pattern_size",
                "棋盘格内角点数，格式 列,行（内角点数，不是方格数）",
                kind="pair",
                source="PATTERN_SIZE",
            ),
            Option(
                ("--square",), "square_size", "方格边长（米）", kind="float", source="SQUARE_SIZE_M"
            ),
            Option(
                ("--lag-search",),
                "lag_search_s",
                "时间差搜索半径（秒）",
                kind="float",
                source="LAG_SEARCH_S",
            ),
            Option(
                ("--lag-step",),
                "lag_step_s",
                "时间差搜索步长（秒）",
                kind="float",
                source="LAG_STEP_S",
            ),
            Option(
                ("--max-frames",),
                "max_frames",
                "调试时只分析索引前缀；默认完整采集",
                kind="int",
                source="MAX_FRAMES",
            ),
            Option(
                ("--strict",),
                "strict",
                "严格模式：外参/时间差没标出来就返回 2（默认只警告）",
                kind="flag",
                source="STRICT",
            ),
            Option(
                ("--estimate-translation",),
                "estimate_translation",
                "使用位置源时标对表平移外参",
                kind="bool",
                source="ESTIMATE_TRANSLATION",
            ),
        ),
    )
}


def _default_of(module: Any, option: Option) -> Any:
    """默认值直接取入口模块的常量（这里不写第二遍）。"""
    if not option.source:
        return None
    if not hasattr(module, option.source):
        raise AttributeError(
            f"{module.__name__} 里没有常量 {option.source}（{option.flags[0]} 的默认值出处）"
        )
    return getattr(module, option.source)


def _add_option(parser: argparse.ArgumentParser, module: Any, option: Option) -> None:
    default = _default_of(module, option)
    if option.kind == "const":
        help_text = f"{option.help}（开关；打开时传 {module.__name__}.{option.source}）"
        parser.add_argument(
            *option.flags, dest=option.dest, action="store_true", default=False, help=help_text
        )
        return
    help_text = f"{option.help}（默认：{module.__name__}.{option.source} = {default!r}）"
    common: dict[str, Any] = {"dest": option.dest, "default": default, "help": help_text}
    if option.kind == "flag":
        parser.add_argument(*option.flags, action="store_true", **common)
    elif option.kind == "flag-off":
        parser.add_argument(*option.flags, action="store_false", **common)
    elif option.kind == "bool":
        parser.add_argument(*option.flags, action=argparse.BooleanOptionalAction, **common)
    elif option.kind == "choice":
        parser.add_argument(*option.flags, choices=option.choices, **common)
    elif option.kind in ("int", "float"):
        parser.add_argument(*option.flags, type=float if option.kind == "float" else int, **common)
    elif option.kind == "path":
        parser.add_argument(*option.flags, type=Path, **common)
    elif option.kind == "pair":
        parser.add_argument(*option.flags, type=_pair, metavar="A,B", **common)
    elif option.kind == "vec2":
        parser.add_argument(*option.flags, type=_vec2, metavar="AZ,EL", **common)
    elif option.kind == "vec3":
        parser.add_argument(*option.flags, type=_vec3, metavar="N,E,D", **common)
    elif option.kind == "str":
        parser.add_argument(*option.flags, **common)
    else:  # pragma: no cover - 注册表写错时立刻抛出异常，不静默
        raise ValueError(f"未知的选项类型：{option.kind!r}（{option.flags[0]}）")


def _pair(text: str) -> tuple[int, int]:
    """``"9,6"`` → ``(9, 6)``（列,行）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"要两个整数（形如 9,6）：{text!r}")
    return int(parts[0]), int(parts[1])


def _vec2(text: str) -> tuple[float, float]:
    """``"218,55"`` → ``(218.0, 55.0)``（例如太阳的 方位角,仰角）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"要两个数（形如 218,55）：{text!r}")
    return float(parts[0]), float(parts[1])


def _vec3(text: str) -> tuple[float, float, float]:
    """``"300,0,-5"`` → ``(300.0, 0.0, -5.0)``（北,东,地）。"""
    parts = text.replace("，", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"要三个数（形如 300,0,0）：{text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def build_parser() -> argparse.ArgumentParser:
    """按 :data:`SUBCOMMANDS` 建出完整解析器（会 import 入口模块读默认值）。

    入口模块都是"纯库模块"（重依赖在函数体内），所以建解析器这一步不加载
    cv2 / mavsdk / torch——``--help`` 因此是廉价的。
    """
    parser = argparse.ArgumentParser(
        prog="python -m airdrop_calibration",
        description="airdrop-calibration 工具入口",
        epilog="子命令的 --help 给出该入口的全部选项与默认值来源。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<子命令>")
    for spec in SUBCOMMANDS.values():
        module = importlib.import_module(spec.module)
        sub = subparsers.add_parser(
            spec.name,
            help=spec.help,
            description=f"{spec.help}（入口模块：{spec.module}）",
        )
        sub.set_defaults(command=spec.name)
        for option in spec.options:
            _add_option(sub, module, option)
    return parser


def _kwargs_of(spec: Subcommand, module: Any, args: argparse.Namespace) -> dict[str, Any]:
    """把解析结果翻成入口模块 ``main(**kwargs)`` 的关键字。"""
    kwargs: dict[str, Any] = {}
    for option in spec.options:
        value = getattr(args, option.dest)
        if option.kind == "const":
            if value:  # 开关打开 → 传入口模块里那个常量（例如 PREFLIGHT_OFF）
                kwargs[option.dest] = _default_of(module, option)
            continue
        kwargs[option.dest] = value
    return kwargs


def resolve_kwargs(command: str, argv: Sequence[str]) -> dict[str, Any]:
    """解析 ``argv`` 并返回"会传给入口模块 ``main(**kwargs)``"的关键字。

    给测试与排障用（``tests/test_cli.py`` 就是靠它验证"选项真的变成了配置覆盖"）：
    它不调用 handler，因此不会碰硬件，也不会跑任务。``argv`` 只写该子命令自己的选项
    （子命令名由 ``command`` 给出）。
    """
    spec = SUBCOMMANDS.get(command)
    if spec is None:
        raise KeyError(f"未知子命令：{command!r}（可选 {sorted(SUBCOMMANDS)}）")
    parser = build_parser()
    # 借用完整解析器：只传入该子命令的选项，避免各子命令的解析器被复制一份
    args = parser.parse_args([command, *argv])
    module = importlib.import_module(spec.module)
    return _kwargs_of(spec, module, args)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；返回进程退出码（参数不合法时由 argparse 以 2 退出）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    spec = SUBCOMMANDS[args.command]
    module = importlib.import_module(spec.module)
    handler = getattr(module, spec.handler)
    result = handler(**_kwargs_of(spec, module, args))
    return 0 if result is None else int(result)
