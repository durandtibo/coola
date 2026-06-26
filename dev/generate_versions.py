# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.utils.mapping import sort_by_keys
from feu.version import (
    fetch_latest_major_versions_map,
    fetch_latest_minor_versions_map,
    fetch_sampled_latest_minor_versions,
    get_package_bounds,
    partition_package_bounds,
    read_pyproject_dependencies,
    read_pyproject_optional_dependencies,
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

    deps = read_pyproject_dependencies(pyproject_path) + read_pyproject_optional_dependencies(
        pyproject_path
    )
    major_deps, minor_deps = partition_package_bounds(deps, ["packaging", "pyarrow"])

    return sort_by_keys(
        fetch_latest_major_versions_map(major_deps)
        | fetch_latest_minor_versions_map(minor_deps)
        | {
            name: fetch_sampled_latest_minor_versions(
                name, lower=get_package_bounds(deps, name).lower, n=n
            )
            for name, n in [("polars", 5), ("xarray", 3)]
        }
    )


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    base_dir = Path(__file__).parent.parent
    versions = fetch_package_versions(base_dir)
    logger.info(f"{versions=}")
    path = base_dir.joinpath("dev/config").joinpath("package_versions.json")
    logger.info(f"Saving package versions to {path}")
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
