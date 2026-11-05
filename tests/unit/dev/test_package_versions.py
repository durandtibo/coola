from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8"))
VERSIONS = json.loads(ROOT.joinpath("dev/config/package_versions.json").read_text(encoding="utf-8"))

OPTIONAL_PACKAGES = sorted(
    {
        Requirement(requirement).name
        for requirements in PYPROJECT["project"]["optional-dependencies"].values()
        for requirement in requirements
    }
)

##########################################
#     Tests for package_versions.json    #
##########################################


@pytest.mark.parametrize("name", OPTIONAL_PACKAGES)
def test_package_versions_has_every_optional_dependency(name: str) -> None:
    assert VERSIONS.get(name)
