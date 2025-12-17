# noqa: INP001
r"""Script to create or update the package versions."""

from __future__ import annotations

import logging
from pathlib import Path

from feu.utils.io import save_json
from feu.version import (
    filter_every_n_versions,
    filter_last_n_versions,
    get_latest_major_versions,
    get_latest_minor_versions,
    sort_versions,
    unique_versions,
)

logger: logging.Logger = logging.getLogger(__name__)


def get_package_versions() -> dict[str, list[str]]:
    r"""Get the versions for each package.

    Returns:
        A dictionary with the versions for each package.
    """
    polars_verions = get_latest_minor_versions("polars", lower="1.0")
    xarray_verions = get_latest_minor_versions("xarray", lower="2023.1")
    return {
        "jax": list(get_latest_minor_versions("jax", lower="0.5")),
        "numpy": list(get_latest_minor_versions("numpy", lower="1.24")),
        "packaging": list(get_latest_major_versions("packaging", lower="22.0")),
        "pandas": list(get_latest_minor_versions("pandas", lower="2.0")),
        "polars": sort_versions(
            unique_versions(
                filter_every_n_versions(polars_verions, n=5)
                + filter_last_n_versions(polars_verions, n=1)
            )
        ),
        "pyarrow": list(get_latest_major_versions("pyarrow", lower="11.0")),
        "torch": list(get_latest_minor_versions("torch", lower="2.0")),
        "xarray": sort_versions(
            unique_versions(
                filter_every_n_versions(xarray_verions, n=3)
                + filter_last_n_versions(xarray_verions, n=1)
            )
        ),
    }


def main() -> None:
    r"""Generate the package versions and save them in a JSON file."""
    versions = get_package_versions()
    logger.info(f"{versions=}")
    path = Path(__file__).parent.parent.joinpath("dev/config").joinpath("package_versions.json")
    logger.info(f"Saving package versions to {path}")
    save_json(versions, path, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
