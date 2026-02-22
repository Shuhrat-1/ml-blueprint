from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class RuntimeInfo:
    mlb_version: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, str]:
        return {
            "mlb_version": self.mlb_version,
            "python_version": self.python_version,
            "platform": self.platform,
        }


def get_mlb_version() -> str:
    """
    Resolve installed package version. Works for editable installs too.
    """
    try:
        return metadata.version("mlb")
    except metadata.PackageNotFoundError:
        # fallback for local runs if package name differs or not installed
        return "0.0.0+local"


def get_runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        mlb_version=get_mlb_version(),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )