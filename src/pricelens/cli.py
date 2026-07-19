"""Typer entrypoints — the Makefile calls these."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(name="pricelens", no_args_is_help=True)
data_app = typer.Typer(name="data", no_args_is_help=True, help="Acquire and clean raw data.")
features_app = typer.Typer(name="features", no_args_is_help=True, help="Build feature matrices.")
model_app = typer.Typer(name="model", no_args_is_help=True, help="Train and evaluate models.")
validate_app = typer.Typer(name="validate", no_args_is_help=True, help="Splits and leakage audit.")

app.add_typer(data_app)
app.add_typer(features_app)
app.add_typer(model_app)
app.add_typer(validate_app)


@app.command()
def version() -> None:
    """Print the installed pricelens version."""
    from pricelens import __version__

    typer.echo(__version__)


@data_app.command("download")
def data_download(
    city: str = typer.Option("nyc", help="Config name under configs/city/ (without .yaml)."),
) -> None:
    """Fetch the pinned Inside Airbnb snapshot (Phase 1)."""
    from pricelens.config import load_city_config
    from pricelens.data.download import download_snapshot

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_city_config(Path("configs/city") / f"{city}.yaml")
    snapshot_dir = download_snapshot(cfg, Path("data/raw"))
    typer.echo(f"snapshot ready at {snapshot_dir}")


@data_app.command("clean")
def data_clean() -> None:
    """Apply the cleaning & filter policy (Phase 2)."""
    raise NotImplementedError("Phase 2: src/pricelens/data/clean.py")


@features_app.command("build")
def features_build() -> None:
    """Build the cold/warm feature matrices (Phase 4)."""
    raise NotImplementedError("Phase 4: src/pricelens/features/pipeline.py")


@model_app.command("train")
def model_train() -> None:
    """Train the GBM models, blend, and calibrate intervals (Phase 5)."""
    raise NotImplementedError("Phase 5: src/pricelens/models/")


@model_app.command("eval")
def model_eval() -> None:
    """Run evaluation, ablation, and explanation reports (Phase 6)."""
    raise NotImplementedError("Phase 6: src/pricelens/explain/")


@validate_app.command("audit")
def validate_audit() -> None:
    """Run the leakage audit (Phase 3+)."""
    raise NotImplementedError("Phase 3: src/pricelens/validation/audit.py")


if __name__ == "__main__":
    app()
