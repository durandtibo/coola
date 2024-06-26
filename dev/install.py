r"""Contain utility functions to install packages."""  # noqa: INP001

from __future__ import annotations

__all__ = ["install"]

import logging
import subprocess

from packaging.version import Version

logger = logging.getLogger(__name__)


def run_bash_command(cmd: str) -> None:
    r"""Execute a bash command.

    Args:
        cmd: The command to run.
    """
    logger.info(f"execute the following command: {cmd}")
    subprocess.run(cmd.split(), check=True)  # noqa: S603


def _install_pandas(version: str) -> None:
    r"""Install the pandas package and associated packages.

    Args:
        version: The target version to install.
    """
    deps = "" if Version(version) >= Version("2.2.2") else " numpy==1.26.4"
    run_bash_command(f"pip install -U pandas=={version}{deps}")


def _install_torch(version: str) -> None:
    r"""Install the torch package and associated packages.

    Args:
        version: The target version to install.
    """
    deps = "" if Version(version) >= Version("2.3.0") else " numpy==1.26.4"
    run_bash_command(f"pip install -U torch=={version}{deps}")


def _install_xarray(version: str) -> None:
    r"""Install the xarray package and associated packages.

    Args:
        version: The target version to install.
    """
    deps = "" if Version(version) >= Version("2023.9") else " numpy==1.26.4"
    run_bash_command(f"pip install -U xarray=={version}{deps}")


_REGISTRY = {
    "pandas": _install_pandas,
    "torch": _install_torch,
    "xarray": _install_xarray,
}


def install(package: str, version: str) -> None:
    r"""Install a package and associated packages.

    Args:
        package: The package name e.g. ``'numpy'``.
        version: The target version to install.
    """
    _REGISTRY[package](version)
