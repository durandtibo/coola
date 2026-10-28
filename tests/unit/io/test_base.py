from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    EqualityTesterRegistry,
    EqualNanEqualityTester,
    get_default_registry,
    interface as equality_interface,
)
from coola.factory import OBJECT_TARGET
from coola.io import (
    BaseFileSaver,
    BaseLoader,
    BaseSaver,
    JsonLoader,
    JsonSaver,
    TextLoader,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)

######################################
#     Tests for is_loader_config     #
######################################

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_is_loader_config_true() -> None:
    assert is_loader_config({OBJECT_TARGET: "coola.io.JsonLoader"})


def test_is_loader_config_false() -> None:
    assert not is_loader_config({OBJECT_TARGET: "coola.io.JsonSaver"})


#####################################
#     Tests for is_saver_config     #
#####################################


def test_is_saver_config_true() -> None:
    assert is_saver_config({OBJECT_TARGET: "coola.io.JsonSaver"})


def test_is_saver_config_false() -> None:
    assert not is_saver_config({OBJECT_TARGET: "coola.io.JsonLoader"})


####################################
#     Tests for resolve_loader     #
####################################


def test_resolve_loader_object() -> None:
    loader = JsonLoader()
    assert resolve_loader(loader) is loader


def test_resolve_loader_dict() -> None:
    assert isinstance(resolve_loader({OBJECT_TARGET: "coola.io.JsonLoader"}), JsonLoader)


def test_resolve_loader_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseLoader instance"):
        resolve_loader({OBJECT_TARGET: "coola.io.JsonSaver"})


###################################
#     Tests for resolve_saver     #
###################################


def test_resolve_saver_object() -> None:
    saver = JsonSaver()
    assert resolve_saver(saver) is saver


def test_resolve_saver_dict() -> None:
    assert isinstance(resolve_saver({OBJECT_TARGET: "coola.io.JsonSaver"}), JsonSaver)


def test_resolve_saver_incorrect_type() -> None:
    with pytest.raises(TypeError, match="Received object is not a BaseSaver instance"):
        resolve_saver({OBJECT_TARGET: "coola.io.JsonLoader"})


def test_equality_tester_registry_has_equality_tester_loader() -> None:
    assert get_default_registry().has_equality_tester(BaseLoader)


def test_equality_tester_registry_has_equality_tester_saver() -> None:
    assert get_default_registry().has_equality_tester(BaseSaver)


##################################################
#     Tests for the equality tester registry     #
##################################################


@pytest.fixture
def reset_default_registry() -> Iterator[EqualityTesterRegistry]:
    # The testers are part of the default registry: they do not depend on a registration
    # made when ``coola.io`` was imported, so they survive a reset of the singleton.
    equality_interface._default_registry.reset()
    try:
        yield get_default_registry()
    finally:
        equality_interface._default_registry.reset()


def test_equality_tester_registry_has_the_loader_tester_after_the_singleton_is_reset(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    assert isinstance(
        reset_default_registry.find_equality_tester(BaseLoader), EqualNanEqualityTester
    )


def test_equality_tester_registry_has_the_saver_tester_after_the_singleton_is_reset(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    assert isinstance(
        reset_default_registry.find_equality_tester(BaseSaver), EqualNanEqualityTester
    )


def test_equality_tester_registry_uses_the_loader_tester_for_the_subclasses(
    reset_default_registry: EqualityTesterRegistry,
) -> None:
    registry = reset_default_registry
    assert registry.find_equality_tester(JsonLoader) is registry.find_equality_tester(BaseLoader)


def test_loaders_and_savers_are_compared_with_equal_equal_loaders() -> None:
    assert objects_are_equal(JsonLoader(), JsonLoader())


def test_loaders_and_savers_are_compared_with_equal_different_loaders() -> None:
    assert not objects_are_equal(JsonLoader(), TextLoader())


def test_loaders_and_savers_are_compared_with_equal_equal_savers_in_a_dict() -> None:
    assert objects_are_equal({"saver": JsonSaver(indent=2)}, {"saver": JsonSaver(indent=2)})


def test_loaders_and_savers_are_compared_with_equal_different_savers() -> None:
    assert not objects_are_equal(JsonSaver(indent=2), JsonSaver(indent=4))


####################################################
#     Tests for allowed_prefixes of the config     #
####################################################


def test_is_loader_config_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    assert is_loader_config(config, allowed_prefixes=["coola.io"])


def test_is_loader_config_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    with pytest.raises(ImportError, match="not allowed"):
        is_loader_config(config, allowed_prefixes=["os"])


def test_is_saver_config_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert is_saver_config(config, allowed_prefixes=["coola.io"])


def test_is_saver_config_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    with pytest.raises(ImportError, match="not allowed"):
        is_saver_config(config, allowed_prefixes=["os"])


def test_is_loader_config_module_not_imported() -> None:
    assert not is_loader_config({OBJECT_TARGET: "coola_missing_module.Loader"})


def test_is_loader_config_import_target() -> None:
    with pytest.raises(ImportError, match="coola_missing_module"):
        is_loader_config({OBJECT_TARGET: "coola_missing_module.Loader"}, import_target=True)


def test_is_saver_config_module_not_imported() -> None:
    assert not is_saver_config({OBJECT_TARGET: "coola_missing_module.Saver"})


def test_is_saver_config_import_target() -> None:
    with pytest.raises(ImportError, match="coola_missing_module"):
        is_saver_config({OBJECT_TARGET: "coola_missing_module.Saver"}, import_target=True)


def test_resolve_loader_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    assert isinstance(resolve_loader(config, allowed_prefixes=["coola.io"]), JsonLoader)


def test_resolve_loader_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonLoader"}
    with pytest.raises(ImportError, match="not allowed"):
        resolve_loader(config, allowed_prefixes=["os"])


def test_resolve_saver_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    assert isinstance(resolve_saver(config, allowed_prefixes=["coola.io"]), JsonSaver)


def test_resolve_saver_not_allowed_prefixes() -> None:
    config = {OBJECT_TARGET: "coola.io.JsonSaver"}
    with pytest.raises(ImportError, match="not allowed"):
        resolve_saver(config, allowed_prefixes=["os"])


####################################################
#     Tests for BaseLoader, BaseSaver (shared)     #
####################################################


class PathTypeLoader(BaseLoader[type]):
    r"""A loader that returns the type of the path given to
    ``_load``."""

    def _load(self, path: Path) -> type:
        return type(path)


class ConfigurableLoader(BaseLoader[Any]):
    def __init__(self, encoding: str = "utf-8", scale: float = 1.0) -> None:
        self._encoding = encoding
        self._scale = scale

    def _display_fields(self) -> dict[str, Any]:
        return {"encoding": self._encoding, "scale": self._scale}

    def _load(self, path: Path) -> Any:  # noqa: ARG002
        return None


class ConfigurableSaver(BaseFileSaver[Any]):
    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def _display_fields(self) -> dict[str, Any]:
        return {"indent": self._indent}

    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


def test_base_loader_load_passes_a_path_for_a_path(tmp_path: Path) -> None:
    assert PathTypeLoader().load(tmp_path) is type(tmp_path)


def test_base_loader_load_passes_a_path_for_a_str(tmp_path: Path) -> None:
    # Regression: a ``load`` that calls ``Path`` methods failed for a ``str``.
    assert issubclass(PathTypeLoader().load(str(tmp_path)), Path)


def test_base_loader_load_passes_a_path_for_a_file_uri(tmp_path: Path) -> None:
    assert issubclass(PathTypeLoader().load(tmp_path.as_uri()), Path)


def test_base_loader_load_expands_a_file_uri(tmp_path: Path) -> None:
    class Loader(BaseLoader[Path]):
        def _load(self, path: Path) -> Path:
            return path

    assert Loader().load(tmp_path.joinpath("data.txt").as_uri()) == tmp_path.joinpath("data.txt")


def test_base_loader_equal_same_configuration() -> None:
    assert ConfigurableLoader(scale=2.0).equal(ConfigurableLoader(scale=2.0))


def test_base_loader_equal_different_field() -> None:
    assert not ConfigurableLoader(scale=2.0).equal(ConfigurableLoader(scale=3.0))


def test_base_loader_equal_different_type() -> None:
    assert not ConfigurableLoader().equal(PathTypeLoader())


def test_base_loader_equal_subclass_is_not_equal_to_base() -> None:
    class Child(ConfigurableLoader): ...

    assert not ConfigurableLoader().equal(Child())


def test_base_loader_equal_not_a_loader() -> None:
    assert not ConfigurableLoader().equal("abc")


def test_base_loader_equal_no_fields() -> None:
    assert PathTypeLoader().equal(PathTypeLoader())


def test_base_loader_equal_nan_false() -> None:
    assert not ConfigurableLoader(scale=float("nan")).equal(ConfigurableLoader(scale=float("nan")))


def test_base_loader_equal_nan_true() -> None:
    assert ConfigurableLoader(scale=float("nan")).equal(
        ConfigurableLoader(scale=float("nan")), equal_nan=True
    )


def test_base_loader_repr() -> None:
    assert repr(ConfigurableLoader(scale=2.0)) == "ConfigurableLoader(encoding='utf-8', scale=2.0)"


def test_base_loader_repr_no_fields() -> None:
    assert repr(PathTypeLoader()) == "PathTypeLoader()"


def test_base_loader_str() -> None:
    assert str(ConfigurableLoader(scale=2.0)) == "ConfigurableLoader(encoding=utf-8, scale=2.0)"


def test_base_saver_equal_same_configuration() -> None:
    assert ConfigurableSaver(indent=4).equal(ConfigurableSaver(indent=4))


def test_base_saver_equal_different_field() -> None:
    assert not ConfigurableSaver(indent=4).equal(ConfigurableSaver(indent=2))


class OtherSaver(BaseFileSaver[Any]):
    def _save_file(self, to_save: Any, path: Path) -> None:
        path.write_text(str(to_save))


def test_base_saver_equal_different_type() -> None:
    assert not ConfigurableSaver().equal(OtherSaver())


def test_base_saver_equal_subclass_is_not_equal_to_base() -> None:
    class Child(ConfigurableSaver): ...

    assert not ConfigurableSaver().equal(Child())


def test_base_saver_repr() -> None:
    assert repr(ConfigurableSaver(indent=4)) == "ConfigurableSaver(indent=4)"


def test_base_saver_repr_no_fields() -> None:
    assert repr(OtherSaver()) == "OtherSaver()"


def test_base_saver_str() -> None:
    assert str(ConfigurableSaver(indent=4)) == "ConfigurableSaver(indent=4)"
