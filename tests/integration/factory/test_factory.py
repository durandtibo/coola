from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

import pytest

from coola.factory import OBJECT_TARGET, factory, is_object_config, resolve_object
from coola.io import resolve_loader
from tests.integration.helpers import run_threads

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

MODULE_NAME = "coola_factory_integration_target"


@pytest.fixture
def target_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    r"""Write a module whose top-level code appends a line to a log file.

    The module is not imported by the fixture. It is removed from
    ``sys.modules`` after the test.
    """
    log = tmp_path.joinpath("imports.log")
    tmp_path.joinpath(f"{MODULE_NAME}.py").write_text(
        f"with open({str(log)!r}, 'a') as file:\n"
        "    file.write('imported\\n')\n\n\n"
        "class Target:\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(MODULE_NAME, None)
    yield log
    sys.modules.pop(MODULE_NAME, None)


def count_imports(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


########################################
#     Tests for the import of target     #
########################################


def test_resolve_object_runs_the_top_level_code_of_the_target_once(target_module: Path) -> None:
    resolve_object({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, cls=object)
    resolve_object({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, cls=object)
    assert count_imports(target_module) == 1


def test_resolve_object_with_cls_runs_the_top_level_code_of_the_target_once(
    target_module: Path,
) -> None:
    config = {OBJECT_TARGET: f"{MODULE_NAME}.Target"}
    target = type(resolve_object(config, cls=object))
    resolve_object(config, cls=target)
    assert count_imports(target_module) == 1


@pytest.mark.usefixtures("target_module")
def test_is_object_config_does_not_import_the_target_by_default() -> None:
    assert not is_object_config({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, object)
    assert MODULE_NAME not in sys.modules


def test_is_object_config_does_not_run_the_code_of_the_target_by_default(
    target_module: Path,
) -> None:
    is_object_config({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, object)
    assert count_imports(target_module) == 0


@pytest.mark.usefixtures("target_module")
def test_is_object_config_imports_the_target_with_import_target() -> None:
    assert is_object_config({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, object, import_target=True)
    assert MODULE_NAME in sys.modules


@pytest.mark.usefixtures("target_module")
def test_is_object_config_imports_the_target_with_allowed_prefixes() -> None:
    assert is_object_config(
        {OBJECT_TARGET: f"{MODULE_NAME}.Target"}, object, allowed_prefixes=[MODULE_NAME]
    )


###################################
#     Tests for the allow-list     #
###################################


def test_resolve_loader_refuses_a_nested_configuration_with_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader", "extra": {OBJECT_TARGET: "os.getcwd"}}
    with pytest.raises(ValueError, match="An argument is a configuration"):
        resolve_loader(config, allowed_prefixes=["coola.io"])


@pytest.mark.usefixtures("target_module")
def test_resolve_loader_refuses_a_nested_configuration_before_importing_it() -> None:
    config = {
        OBJECT_TARGET: "coola.io.JsonLoader",
        "extra": {OBJECT_TARGET: f"{MODULE_NAME}.Target"},
    }
    with pytest.raises(ValueError, match="An argument is a configuration"):
        resolve_loader(config, allowed_prefixes=["coola.io"])
    assert MODULE_NAME not in sys.modules


def test_factory_does_not_call_the_nested_configuration_of_resolve_loader() -> None:
    with pytest.raises(ValueError, match="An argument is a configuration"):
        factory(
            "coola.io.resolve_loader",
            loader={OBJECT_TARGET: "os.getcwd"},
            _allowed_prefixes_=["coola.io"],
        )


def test_resolve_loader_refuses_a_target_outside_the_allowed_prefixes_without_importing_it(
    target_module: Path,
) -> None:
    with pytest.raises(ImportError, match="is not allowed"):
        resolve_loader({OBJECT_TARGET: f"{MODULE_NAME}.Target"}, allowed_prefixes=["coola.io"])
    assert count_imports(target_module) == 0


##############################
#     Tests for threads     #
##############################


def test_resolve_object_from_several_threads() -> None:
    """Test that ``resolve_object`` returns the right object when it is
    called from several threads at the same time."""
    results: list[tuple[int, dict[str, int]]] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        result = resolve_object({OBJECT_TARGET: "collections.Counter", "a": i}, dict)
        with lock:
            results.append((i, dict(result)))

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(16)])

    assert sorted(results) == [(i, {"a": i}) for i in range(16)]
