from __future__ import annotations

from typer.testing import CliRunner

from pricelens import __version__
from pricelens.cli import app


def test_version_string_is_set() -> None:
    assert __version__ == "0.1.0"


def test_cli_version_command_runs() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
