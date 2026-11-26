from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "dev" / "generate_versions.py"
SPEC = importlib.util.spec_from_file_location("generate_versions", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

MINORS = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]


@pytest.fixture
def feu_mocks() -> dict[str, Mock]:
    r"""Patch the ``feu`` functions used by ``fetch_package_versions``."""
    mocks = {
        "read_pyproject_bounds": Mock(return_value={"bounds": 1}),
        "fetch_feu_package_versions": Mock(return_value={"torch": ["2.0"], "numpy": ["1.0"]}),
        "fetch_latest_versions": Mock(return_value=MINORS),
        "get_package_bounds": Mock(return_value=Mock(lower="1.0")),
    }
    with patch.multiple(module, **mocks):
        yield mocks


############################################
#     Tests for fetch_package_versions     #
############################################


@pytest.mark.usefixtures("feu_mocks")
def test_fetch_package_versions_sorted_by_name(tmp_path: Path) -> None:
    assert list(module.fetch_package_versions(tmp_path)) == ["numpy", "polars", "torch", "xarray"]


@pytest.mark.usefixtures("feu_mocks")
def test_fetch_package_versions_samples_polars_every_five_minors(tmp_path: Path) -> None:
    assert module.fetch_package_versions(tmp_path)["polars"] == ["1.0", "1.5", "1.6"]


@pytest.mark.usefixtures("feu_mocks")
def test_fetch_package_versions_samples_xarray_every_three_minors(tmp_path: Path) -> None:
    assert module.fetch_package_versions(tmp_path)["xarray"] == ["1.0", "1.3", "1.6"]


@pytest.mark.usefixtures("feu_mocks")
def test_fetch_package_versions_keeps_other_versions(tmp_path: Path) -> None:
    assert module.fetch_package_versions(tmp_path)["torch"] == ["2.0"]


def test_fetch_package_versions_keeps_latest_when_it_is_sampled(
    tmp_path: Path, feu_mocks: dict[str, Mock]
) -> None:
    feu_mocks["fetch_latest_versions"].return_value = MINORS[:6]
    assert module.fetch_package_versions(tmp_path)["polars"] == ["1.0", "1.5"]


def test_fetch_package_versions_reads_pyproject_of_base_dir(
    tmp_path: Path, feu_mocks: dict[str, Mock]
) -> None:
    module.fetch_package_versions(tmp_path)
    feu_mocks["read_pyproject_bounds"].assert_called_once_with(tmp_path / "pyproject.toml")


def test_fetch_package_versions_fetches_major_only_packages(
    tmp_path: Path, feu_mocks: dict[str, Mock]
) -> None:
    module.fetch_package_versions(tmp_path)
    feu_mocks["fetch_feu_package_versions"].assert_called_once_with(
        tmp_path / "pyproject.toml", major_only=["packaging", "pyarrow"], include_lower_bound=True
    )


def test_fetch_package_versions_fetches_minors_from_lower_bound(
    tmp_path: Path, feu_mocks: dict[str, Mock]
) -> None:
    module.fetch_package_versions(tmp_path)
    feu_mocks["fetch_latest_versions"].assert_any_call(
        "polars", group="minor", lower="1.0", include_lower_bound=True
    )


##########################
#     Tests for main     #
##########################


def test_main_saves_versions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "dev" / "config" / "package_versions.json"
    path.parent.mkdir(parents=True)
    fake = tmp_path / "dev" / "generate_versions.py"
    monkeypatch.setattr(module, "__file__", str(fake))
    with patch.object(module, "fetch_package_versions", return_value={"numpy": ["1.0"]}):
        module.main()
    assert json.loads(path.read_text()) == {"numpy": ["1.0"]}


def test_main_overwrites_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "dev" / "config" / "package_versions.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"old": ["0.1"]}')
    monkeypatch.setattr(module, "__file__", str(tmp_path / "dev" / "generate_versions.py"))
    with patch.object(module, "fetch_package_versions", return_value={"numpy": ["1.0"]}):
        module.main()
    assert json.loads(path.read_text()) == {"numpy": ["1.0"]}
