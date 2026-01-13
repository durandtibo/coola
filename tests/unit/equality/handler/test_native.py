from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    ObjectEqualHandler,
    SameAttributeHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if TYPE_CHECKING:
    from collections.abc import Sized


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


##################################
#     Tests for FalseHandler     #
##################################


def test_false_handler_equal_true() -> None:
    assert FalseHandler() == FalseHandler()


def test_false_handler_equal_false_different_type() -> None:
    assert FalseHandler() != TrueHandler()


def test_false_handler_equal_false_different_type_child() -> None:
    class Child(FalseHandler): ...

    assert FalseHandler() != Child()


def test_false_handler_repr() -> None:
    assert repr(FalseHandler()) == "FalseHandler()"


def test_false_handler_str() -> None:
    assert str(FalseHandler()) == "FalseHandler()"


@pytest.mark.parametrize(
    ("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_false_handler_handle(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert not FalseHandler().handle(actual, expected, config)


def test_false_handler_set_next_handler() -> None:
    FalseHandler().set_next_handler(TrueHandler())


##################################
#     Tests for TrueHandler     #
##################################


def test_true_handler_equal_true() -> None:
    assert TrueHandler() == TrueHandler()


def test_true_handler_equal_false_different_type() -> None:
    assert TrueHandler() != FalseHandler()


def test_true_handler_equal_false_different_type_child() -> None:
    class Child(TrueHandler): ...

    assert TrueHandler() != Child()


def test_true_handler_repr() -> None:
    assert repr(TrueHandler()) == "TrueHandler()"


def test_true_handler_str() -> None:
    assert str(TrueHandler()) == "TrueHandler()"


@pytest.mark.parametrize(
    ("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_true_handler_handle(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert TrueHandler().handle(actual, expected, config)


def test_true_handler_set_next_handler() -> None:
    TrueHandler().set_next_handler(FalseHandler())


########################################
#     Tests for ObjectEqualHandler     #
########################################


def test_object_equal_handler_equal_true() -> None:
    assert ObjectEqualHandler() == ObjectEqualHandler()


def test_object_equal_handler_equal_false_different_type() -> None:
    assert ObjectEqualHandler() != FalseHandler()


def test_object_equal_handler_equal_false_different_type_child() -> None:
    class Child(ObjectEqualHandler): ...

    assert ObjectEqualHandler() != Child()


def test_object_equal_handler_repr() -> None:
    assert str(ObjectEqualHandler()) == "ObjectEqualHandler()"


def test_object_equal_handler_str() -> None:
    assert str(ObjectEqualHandler()) == "ObjectEqualHandler()"


@pytest.mark.parametrize(("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_object_equal_handler_handle_true(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert ObjectEqualHandler().handle(actual, expected, config)


@pytest.mark.parametrize(("actual", "expected"), [(0, 1), (4, 4.2), ("abc", "ABC")])
def test_object_equal_handler_handle_false(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert not ObjectEqualHandler().handle(actual, expected, config)


def test_object_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = ObjectEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=[1, 2, 3], expected=[1, 2, 3, 4], config=config)
        assert caplog.messages[0].startswith("objects are different:")


def test_object_equal_handler_set_next_handler() -> None:
    ObjectEqualHandler().set_next_handler(FalseHandler())


##########################################
#     Tests for SameAttributeHandler     #
##########################################


def test_same_attribute_handler_equal_true() -> None:
    assert SameAttributeHandler(name="name") == SameAttributeHandler(name="name")


def test_same_attribute_handler_equal_false_different_type() -> None:
    assert SameAttributeHandler(name="data") != FalseHandler()


def test_same_attribute_handler_equal_false_different_type_child() -> None:
    class Child(SameAttributeHandler): ...

    assert SameAttributeHandler(name="data") != Child(name="data")


def test_same_attribute_handler_equal_false_different_name() -> None:
    assert SameAttributeHandler(name="data1") != SameAttributeHandler(name="data2")


def test_same_attribute_handler_repr() -> None:
    assert repr(SameAttributeHandler(name="data")).startswith("SameAttributeHandler(")


def test_same_attribute_handler_str() -> None:
    assert str(SameAttributeHandler(name="data")) == "SameAttributeHandler(name=data)"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Mock(data=1), Mock(data=1)),
        (Mock(data="abc"), Mock(data="abc")),
        (Mock(data=[1, 2, 3]), Mock(data=[1, 2, 3])),
    ],
)
def test_same_attribute_handler_handle_true(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert SameAttributeHandler(name="data", next_handler=TrueHandler()).handle(
        actual, expected, config
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Mock(data=1), Mock(data=2)),
        (Mock(data="abc"), Mock(data="abcd")),
        (Mock(data=[1, 2, 3]), Mock(data=[1, 2, 4])),
    ],
)
def test_same_attribute_handler_handle_false(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert not SameAttributeHandler(name="data").handle(actual, expected, config)


@numpy_available
def test_same_attribute_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameAttributeHandler(name="data")
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=Mock(data=1),
            expected=Mock(data=2),
            config=config,
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@numpy_available
def test_same_attribute_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameAttributeHandler(name="data")
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(
            actual=Mock(spec=Any, data=1), expected=Mock(spec=Any, data=1), config=config
        )


def test_same_attribute_handler_set_next_handler() -> None:
    handler = SameAttributeHandler(name="data")
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_attribute_handler_set_next_handler_incorrect() -> None:
    handler = SameAttributeHandler(name="data")
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)


@numpy_available
def test_same_attribute_handler_handle_true_numpy(config: EqualityConfig) -> None:
    assert SameAttributeHandler(name="dtype", next_handler=TrueHandler()).handle(
        np.ones(shape=(2, 3)), np.ones(shape=(2, 3)), config
    )


@numpy_available
def test_same_attribute_handler_handle_false_numpy(config: EqualityConfig) -> None:
    assert not SameAttributeHandler(name="dtype", next_handler=TrueHandler()).handle(
        np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=int), config
    )


#######################################
#     Tests for SameLengthHandler     #
#######################################


def test_same_length_handler_equal_true() -> None:
    assert SameLengthHandler() == SameLengthHandler()


def test_same_length_handler_equal_false_different_type() -> None:
    assert SameLengthHandler() != FalseHandler()


def test_same_length_handler_equal_false_different_type_child() -> None:
    class Child(SameLengthHandler): ...

    assert SameLengthHandler() != Child()


def test_same_length_handler_repr() -> None:
    assert repr(SameLengthHandler()).startswith("SameLengthHandler(")


def test_same_length_handler_str() -> None:
    assert str(SameLengthHandler()) == "SameLengthHandler()"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([], []),
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        ({1, 2, 3}, {4, 5, 6}),
        ("abc", "abc"),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_same_length_handler_handle_true(
    actual: Sized, expected: Sized, config: EqualityConfig
) -> None:
    assert SameLengthHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([1, 2], [1, 2, 3]),
        ((4, 5), (4, 5, None)),
        ({"a", "b", "c"}, {"a"}),
    ],
)
def test_same_length_handler_handle_false(
    actual: Sized, expected: Sized, config: EqualityConfig
) -> None:
    assert not SameLengthHandler().handle(actual, expected, config)


def test_same_length_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameLengthHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=[1, 2, 3], expected=[1, 2, 3, 4], config=config)
        assert caplog.messages[0].startswith("objects have different lengths:")


def test_same_length_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameLengthHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_same_length_handler_set_next_handler() -> None:
    handler = SameLengthHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_length_handler_set_next_handler_incorrect() -> None:
    handler = SameLengthHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)


#######################################
#     Tests for SameObjectHandler     #
#######################################


def test_same_object_handler_equal_true() -> None:
    assert SameObjectHandler() == SameObjectHandler()


def test_same_object_handler_equal_false_different_type() -> None:
    assert SameObjectHandler() != FalseHandler()


def test_same_object_handler_equal_false_different_type_child() -> None:
    class Child(SameObjectHandler): ...

    assert SameObjectHandler() != Child()


def test_same_object_handler_repr() -> None:
    assert repr(SameObjectHandler()).startswith("SameObjectHandler(")


def test_same_object_handler_str() -> None:
    assert str(SameObjectHandler()) == "SameObjectHandler()"


@pytest.mark.parametrize(("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_same_object_handler_handle_true(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert SameObjectHandler().handle(actual, expected, config)


@pytest.mark.parametrize(("actual", "expected"), [(0, 1), (4, 4.0), ("abc", "ABC")])
def test_same_object_handler_handle_false(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert not SameObjectHandler(next_handler=FalseHandler()).handle(actual, expected, config)


def test_same_object_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual="abc", expected="ABC", config=config)


def test_same_object_handler_set_next_handler() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_object_handler_set_next_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)


#####################################
#     Tests for SameTypeHandler     #
#####################################


def test_same_type_handler_equal_true() -> None:
    assert SameTypeHandler() == SameTypeHandler()


def test_same_type_handler_equal_false_different_type() -> None:
    assert SameTypeHandler() != FalseHandler()


def test_same_type_handler_equal_false_different_type_child() -> None:
    class Child(SameTypeHandler): ...

    assert SameTypeHandler() != Child()


def test_same_type_handler_repr() -> None:
    assert repr(SameTypeHandler()).startswith("SameTypeHandler(")


def test_same_type_handler_str() -> None:
    assert str(SameTypeHandler()) == "SameTypeHandler()"


@pytest.mark.parametrize(("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_same_type_handler_handle_true(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert SameTypeHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(("actual", "expected"), [(0, "abc"), (4, 4.0), (None, 0)])
def test_same_type_handler_handle_false(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert not SameTypeHandler().handle(actual, expected, config)


def test_same_type_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameTypeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=0, expected="abc", config=config)
        assert caplog.messages[0].startswith("objects have different types:")


def test_same_type_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameTypeHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual="abc", expected="ABC", config=config)


def test_same_type_handler_set_next_handler() -> None:
    handler = SameTypeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_type_handler_set_next_handler_incorrect() -> None:
    handler = SameTypeHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for `handler`."):
        handler.set_next_handler(None)
