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


def test_false_handler_repr() -> None:
    assert repr(FalseHandler()) == "FalseHandler()"


def test_false_handler_str() -> None:
    assert str(FalseHandler()) == "FalseHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(FalseHandler(), FalseHandler(), id="without next handler"),
        pytest.param(
            FalseHandler(FalseHandler()),
            FalseHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_false_handler_equal_true(handler1: FalseHandler, handler2: FalseHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            FalseHandler(TrueHandler()),
            FalseHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            FalseHandler(),
            FalseHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            FalseHandler(FalseHandler()),
            FalseHandler(),
            id="other next handler is none",
        ),
        pytest.param(FalseHandler(), TrueHandler(), id="different type"),
    ],
)
def test_false_handler_equal_false(handler1: FalseHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_false_handler_equal_false_different_type_child() -> None:
    class Child(FalseHandler): ...

    assert not FalseHandler().equal(Child())


@pytest.mark.parametrize(
    ("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_false_handler_handle(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert not FalseHandler().handle(actual, expected, config)


def test_false_handler_equal_handle_set_next_handler() -> None:
    handler = FalseHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_false_handler_equal_handle_set_next_handler_none() -> None:
    handler = FalseHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_false_handler_equal_handle_set_next_handler_incorrect() -> None:
    handler = FalseHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#################################
#     Tests for TrueHandler     #
#################################


def test_true_handler_repr() -> None:
    assert repr(TrueHandler()) == "TrueHandler()"


def test_true_handler_str() -> None:
    assert str(TrueHandler()) == "TrueHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(TrueHandler(), TrueHandler(), id="without next handler"),
        pytest.param(
            TrueHandler(FalseHandler()),
            TrueHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_true_handler_equal_true(handler1: TrueHandler, handler2: TrueHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            TrueHandler(TrueHandler()),
            TrueHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            TrueHandler(),
            TrueHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            TrueHandler(FalseHandler()),
            TrueHandler(),
            id="other next handler is none",
        ),
        pytest.param(TrueHandler(), FalseHandler(), id="different type"),
    ],
)
def test_true_handler_equal_false(handler1: TrueHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_true_handler_equal_false_different_type_child() -> None:
    class Child(TrueHandler): ...

    assert not TrueHandler().equal(Child())


@pytest.mark.parametrize(
    ("actual", "expected"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_true_handler_handle(actual: Any, expected: Any, config: EqualityConfig) -> None:
    assert TrueHandler().handle(actual, expected, config)


def test_true_handler_set_next_handler() -> None:
    TrueHandler().set_next_handler(FalseHandler())


def test_true_handler_equal_handle_set_next_handler() -> None:
    handler = TrueHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_true_handler_equal_handle_set_next_handler_none() -> None:
    handler = TrueHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_true_handler_equal_handle_set_next_handler_incorrect() -> None:
    handler = TrueHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


########################################
#     Tests for ObjectEqualHandler     #
########################################


def test_object_equal_handler_repr() -> None:
    assert str(ObjectEqualHandler()) == "ObjectEqualHandler()"


def test_object_equal_handler_str() -> None:
    assert str(ObjectEqualHandler()) == "ObjectEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(ObjectEqualHandler(), ObjectEqualHandler(), id="without next handler"),
        pytest.param(
            ObjectEqualHandler(FalseHandler()),
            ObjectEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_object_equal_handler_equal_true(
    handler1: ObjectEqualHandler, handler2: ObjectEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            ObjectEqualHandler(TrueHandler()),
            ObjectEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            ObjectEqualHandler(),
            ObjectEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            ObjectEqualHandler(FalseHandler()),
            ObjectEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(ObjectEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_object_equal_handler_equal_false(handler1: ObjectEqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_object_equal_handler_equal_false_different_type_child() -> None:
    class Child(ObjectEqualHandler): ...

    assert not ObjectEqualHandler().equal(Child())


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


def test_object_equal_handler_equal_handle_set_next_handler() -> None:
    handler = ObjectEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_object_equal_handler_equal_handle_set_next_handler_none() -> None:
    handler = ObjectEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_object_equal_handler_equal_handle_set_next_handler_incorrect() -> None:
    handler = ObjectEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


##########################################
#     Tests for SameAttributeHandler     #
##########################################


def test_same_attribute_handler_repr() -> None:
    assert repr(SameAttributeHandler(name="data")).startswith("SameAttributeHandler(")


def test_same_attribute_handler_str() -> None:
    assert str(SameAttributeHandler(name="data")) == "SameAttributeHandler(name=data)"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameAttributeHandler(name="data"),
            SameAttributeHandler(name="data"),
            id="without next handler",
        ),
        pytest.param(
            SameAttributeHandler(name="data", next_handler=FalseHandler()),
            SameAttributeHandler(name="data", next_handler=FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_same_dtype_handler_equal_true(
    handler1: SameAttributeHandler, handler2: SameAttributeHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameAttributeHandler(name="data"),
            SameAttributeHandler(name="meow"),
            id="different name",
        ),
        pytest.param(
            SameAttributeHandler(name="data", next_handler=TrueHandler()),
            SameAttributeHandler(name="data", next_handler=FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SameAttributeHandler(name="data"),
            SameAttributeHandler(name="data", next_handler=FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SameAttributeHandler(name="data", next_handler=FalseHandler()),
            SameAttributeHandler(name="data"),
            id="other next handler is none",
        ),
        pytest.param(SameAttributeHandler(name="data"), FalseHandler(), id="different type"),
    ],
)
def test_same_dtype_handler_equal_false(handler1: SameAttributeHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_same_attribute_handler_equal_false_different_type_child() -> None:
    class Child(SameAttributeHandler): ...

    assert not SameAttributeHandler(name="data").equal(Child(name="data"))


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


def test_same_attribute_handler_set_next_handler() -> None:
    handler = SameAttributeHandler(name="data")
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_same_attribute_handler_set_next_handler_none() -> None:
    handler = SameAttributeHandler(name="data")
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_attribute_handler_set_next_handler_incorrect() -> None:
    handler = SameAttributeHandler(name="data")
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#######################################
#     Tests for SameLengthHandler     #
#######################################


def test_same_length_handler_repr() -> None:
    assert repr(SameLengthHandler()) == "SameLengthHandler()"


def test_same_length_handler_str() -> None:
    assert str(SameLengthHandler()) == "SameLengthHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(SameLengthHandler(), SameLengthHandler(), id="without next handler"),
        pytest.param(
            SameLengthHandler(FalseHandler()),
            SameLengthHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_same_length_handler_equal_true(
    handler1: SameLengthHandler, handler2: SameLengthHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameLengthHandler(TrueHandler()),
            SameLengthHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SameLengthHandler(),
            SameLengthHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SameLengthHandler(FalseHandler()),
            SameLengthHandler(),
            id="other next handler is none",
        ),
        pytest.param(SameLengthHandler(), FalseHandler(), id="different type"),
    ],
)
def test_same_length_handler_equal_false(handler1: SameLengthHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_same_length_handler_equal_false_different_type_child() -> None:
    class Child(SameLengthHandler): ...

    assert not SameLengthHandler().equal(Child())


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


def test_same_length_handler_set_next_handler_none() -> None:
    handler = SameLengthHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_length_handler_set_next_handler_incorrect() -> None:
    handler = SameLengthHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#######################################
#     Tests for SameObjectHandler     #
#######################################


def test_same_object_handler_repr() -> None:
    assert repr(SameObjectHandler()) == "SameObjectHandler()"


def test_same_object_handler_str() -> None:
    assert str(SameObjectHandler()) == "SameObjectHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(SameObjectHandler(), SameObjectHandler(), id="without next handler"),
        pytest.param(
            SameObjectHandler(FalseHandler()),
            SameObjectHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_same_object_handler_equal_true(
    handler1: SameObjectHandler, handler2: SameObjectHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameObjectHandler(TrueHandler()),
            SameObjectHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SameObjectHandler(),
            SameObjectHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SameObjectHandler(FalseHandler()),
            SameObjectHandler(),
            id="other next handler is none",
        ),
        pytest.param(SameObjectHandler(), FalseHandler(), id="different type"),
    ],
)
def test_same_object_handler_equal_false(handler1: SameObjectHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_same_object_handler_equal_false_different_type_child() -> None:
    class Child(SameObjectHandler): ...

    assert not SameObjectHandler().equal(Child())


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


def test_same_object_handler_set_next_handler_none() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_object_handler_set_next_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#####################################
#     Tests for SameTypeHandler     #
#####################################


def test_same_type_handler_repr() -> None:
    assert repr(SameTypeHandler()) == "SameTypeHandler()"


def test_same_type_handler_str() -> None:
    assert str(SameTypeHandler()) == "SameTypeHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(SameTypeHandler(), SameTypeHandler(), id="without next handler"),
        pytest.param(
            SameTypeHandler(FalseHandler()),
            SameTypeHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_same_type_handler_equal_true(handler1: SameTypeHandler, handler2: SameTypeHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            SameTypeHandler(TrueHandler()),
            SameTypeHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            SameTypeHandler(),
            SameTypeHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            SameTypeHandler(FalseHandler()),
            SameTypeHandler(),
            id="other next handler is none",
        ),
        pytest.param(SameTypeHandler(), FalseHandler(), id="different type"),
    ],
)
def test_same_type_handler_equal_false(handler1: SameTypeHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_same_type_handler_equal_false_different_type_child() -> None:
    class Child(SameTypeHandler): ...

    assert not SameTypeHandler().equal(Child())


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


def test_same_type_handler_set_next_handler_none() -> None:
    handler = SameTypeHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_same_type_handler_set_next_handler_incorrect() -> None:
    handler = SameTypeHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)
