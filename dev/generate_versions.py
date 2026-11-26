# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.version import (
    fetch_latest_versions,
    fetch_package_versions as fetch_feu_package_versions,
    filter_every_n_versions,
    get_package_bounds,
    read_pyproject_bounds,
)

logger: logging.Logger = logging.getLogger(__name__)


def fetch_package_versions(base_dir: Path) -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Args:
        base_dir: Path to the base directory.

    Returns:
        A dictionary with the versions for each package.
    """
    pyproject_path = base_dir.joinpath("pyproject.toml")
    sampled = {"polars": 5, "xarray": 3}

    bounds = read_pyproject_bounds(pyproject_path)
    versions = fetch_feu_package_versions(
        pyproject_path, major_only=["packaging", "pyarrow"], include_lower_bound=True
    )
    for name, n in sampled.items():
        minors = fetch_latest_versions(
            name,
            group="minor",
            lower=get_package_bounds(bounds, name).lower,
            include_lower_bound=True,
        )
        # Every n-th minor version, always keeping the latest one.
        versions[name] = sorted({*filter_every_n_versions(minors, n), minors[-1]}, key=minors.index)
    return dict(sorted(versions.items()))


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    base_dir = Path(__file__).parent.parent
    versions = fetch_package_versions(base_dir)
    logger.info("versions=%s", versions)
    path = base_dir.joinpath("dev/config").joinpath("package_versions.json")
    logger.info("Saving package versions to %s", path)
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
