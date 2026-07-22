"""Project paths shared by the Ubuntu training and evaluation entry points.

Defaults are derived from this file instead of the caller's working directory.
Every path can be overridden with an environment variable when the data is moved.
"""

from __future__ import annotations

import os
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_DIR = _env_path("SAM_DATA_DIR", CODE_DIR / "earth_dataset_v3")
DATA_H5 = _env_path("SAM_DATA_H5", DATA_DIR / "faults.h5")
SAM2_CHECKPOINT = _env_path(
    "SAM2_CHECKPOINT", CODE_DIR / "sam2_hiera_small.pt"
)
OUTPUT_DIR = _env_path("SAM_OUTPUT_DIR", PROJECT_ROOT / "outputs")
GEBCO_FILE = _env_path(
    "GEBCO_FILE",
    PROJECT_ROOT / "GEBCO_California" / "gebco_2026_n50.0_s19.0_w-129.0_e-110.0.nc",
)
QFAULTS_FILE = _env_path(
    "QFAULTS_FILE", PROJECT_ROOT / "Qfaults_GIS" / "SHP" / "ca_offshore.shp"
)


def resolve_input(value: str | os.PathLike[str], *, base: Path = CODE_DIR) -> Path:
    """Resolve an input path from cwd first, then from the code directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (base / path).resolve()


def resolve_output(value: str | os.PathLike[str]) -> Path:
    """Resolve an output path relative to the caller's working directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Check the local checkout or override its path with the documented option/environment variable."
        )
    return path
