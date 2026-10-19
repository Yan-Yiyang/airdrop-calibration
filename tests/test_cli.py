"""独立 CLI 参数、默认值与轻量帮助的契约。"""

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from airdrop_calibration import cli as run

pytestmark = pytest.mark.integration


def _config_kwargs(module_name: str, kwargs: dict) -> dict:
    """只挑 ``build_config`` 认的关键字（``dry_run``/``record`` 这类是运行期开关）。"""
    module = importlib.import_module(module_name)
    parameters = inspect.signature(module.build_config).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def test_calibrate_options_override_the_config() -> None:
    from airdrop_calibration import calibrate

    kwargs = run.resolve_kwargs(
        "calibrate",
        ["--flight", "flights/y", "--pattern", "7,5", "--square", "0.03", "--strict"],
    )
    settings = calibrate.build_config(**kwargs)

    assert settings.flight_dir == Path("flights/y")
    assert settings.pattern_size == (7, 5)
    assert settings.square_size == pytest.approx(0.03)
    assert settings.strict is True


@pytest.mark.parametrize("command", list(run.SUBCOMMANDS))
def test_help_is_light(command):
    code = """
import sys
from airdrop_calibration import cli
try:
    cli.main([COMMAND, '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(n in sys.modules for n in ('cv2', 'torch', 'mavsdk', 'rapidocr', 'ultralytics'))
""".replace("COMMAND", repr(command))
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_defaults_match_constants():
    for command, spec in run.SUBCOMMANDS.items():
        module = importlib.import_module(spec.module)
        kwargs = run.resolve_kwargs(command, [])
        for option in spec.options:
            if option.source and option.kind != "const":
                assert kwargs[option.dest] == getattr(module, option.source)


@pytest.mark.parametrize("arguments", [["calibrate", "--pattern", "9"]])
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as error:
        run.main(arguments)
    assert error.value.code == 2


def test_registry_modules_and_exports():
    for command, spec in run.SUBCOMMANDS.items():
        module = importlib.import_module(spec.module)
        assert callable(getattr(module, spec.handler))
        assert callable(module.build_config)
        for option in spec.options:
            if option.source:
                assert hasattr(module, option.source)


def test_public_exports_resolve():
    package = importlib.import_module("airdrop_calibration")
    assert len(package.__all__) == len(set(package.__all__))
    assert all(hasattr(package, name) for name in package.__all__)
