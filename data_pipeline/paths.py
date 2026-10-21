"""Resolve public stage paths. Inputs: Repository-relative or absolute paths. Outputs: Absolute paths."""

from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPW = "epw/SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw"


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root, independently of the working directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (RELEASE_ROOT / path).resolve()
