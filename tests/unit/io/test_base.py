from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality.tester import get_default_registry
from coola.factory import OBJECT_TARGET
from coola.io import (
    BaseLoader,
    BaseSaver,
    JsonLoader,
    JsonSaver,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)

if TYPE_CHECKING:
    import pytest

######################################
#     Tests for is_loader_config     #
######################################


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


def test_resolve_loader_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(resolve_loader({OBJECT_TARGET: "coola.io.JsonSaver"}), JsonSaver)
        assert caplog.messages


###################################
#     Tests for resolve_saver     #
###################################


def test_resolve_saver_object() -> None:
    saver = JsonSaver()
    assert resolve_saver(saver) is saver


def test_resolve_saver_dict() -> None:
    assert isinstance(resolve_saver({OBJECT_TARGET: "coola.io.JsonSaver"}), JsonSaver)


def test_resolve_saver_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(resolve_saver({OBJECT_TARGET: "coola.io.JsonLoader"}), JsonLoader)
        assert caplog.messages


def test_equality_tester_registry_has_equality_tester() -> None:
    assert get_default_registry().has_equality_tester(BaseLoader)
    assert get_default_registry().has_equality_tester(BaseSaver)
