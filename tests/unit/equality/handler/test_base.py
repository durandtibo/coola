from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    ObjectEqualHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
)


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_init_next_handler() -> None:
    assert SameObjectHandler(ObjectEqualHandler()).next_handler.equal(ObjectEqualHandler())


def test_init_next_handler_default() -> None:
    assert SameObjectHandler().next_handler is None


def test_init_next_handler_incorrect() -> None:
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        SameObjectHandler(42)


def test_repr_without_next_handler() -> None:
    assert repr(SameObjectHandler()) == "SameObjectHandler()"


def test_repr_with_next_handler() -> None:
    assert (
        repr(SameObjectHandler(ObjectEqualHandler()))
        == "SameObjectHandler(next_handler=ObjectEqualHandler())"
    )


def test_repr_with_multiple_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    assert (
        repr(handler) == "SameObjectHandler(next_handler=SameTypeHandler("
        "next_handler=SameLengthHandler(next_handler=ObjectEqualHandler())))"
    )


def test_str_without_next_handler() -> None:
    assert str(SameObjectHandler()) == "SameObjectHandler()"


def test_str_with_next_handler() -> None:
    assert str(SameObjectHandler(ObjectEqualHandler())) == "SameObjectHandler()"


def test_str_with_multiple_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    assert str(handler) == "SameObjectHandler()"


def test_chain_1_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain(ObjectEqualHandler())
    assert handler.next_handler.equal(ObjectEqualHandler())
    assert handler.next_handler.next_handler is None
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_multiple_handlers(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(ObjectEqualHandler())
    assert handler.next_handler.equal(SameTypeHandler(SameLengthHandler(ObjectEqualHandler())))
    assert handler.next_handler.next_handler.equal(SameLengthHandler(ObjectEqualHandler()))
    assert handler.next_handler.next_handler.next_handler.equal(ObjectEqualHandler())
    assert handler.next_handler.next_handler.next_handler.next_handler is None
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_with_cycle() -> None:
    handler = SameObjectHandler()
    with pytest.raises(
        RuntimeError, match=r"Cycle detected! the current handler cannot be its next handler"
    ):
        handler.chain(handler)


def test_chain_handler_none() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"The next handler in the chain cannot be None"):
        handler.chain(None)


def test_chain_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.chain(42)


def test_chain_all_1_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain_all(ObjectEqualHandler())
    assert handler.next_handler.equal(ObjectEqualHandler())
    assert handler.next_handler.next_handler is None
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_all_multiple_handlers(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    assert handler.next_handler.equal(SameTypeHandler(SameLengthHandler(ObjectEqualHandler())))
    assert handler.next_handler.next_handler.equal(SameLengthHandler(ObjectEqualHandler()))
    assert handler.next_handler.next_handler.next_handler.equal(ObjectEqualHandler())
    assert handler.next_handler.next_handler.next_handler.next_handler is None
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_all_with_cycle() -> None:
    handler = SameObjectHandler()
    with pytest.raises(
        RuntimeError, match=r"Cycle detected! the current handler cannot be its next handler"
    ):
        handler.chain_all(handler, SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())


def test_get_chain_length_1() -> None:
    handler = SameObjectHandler()
    assert handler.get_chain_length() == 1


def test_get_chain_length_2() -> None:
    handler = SameObjectHandler()
    handler.chain(ObjectEqualHandler())
    assert handler.get_chain_length() == 2


def test_get_chain_length_4() -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    assert handler.get_chain_length() == 4


def test_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=1, expected=2, config=config)


def test_set_next_handler_handler() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(ObjectEqualHandler())
    assert handler.next_handler.equal(ObjectEqualHandler())


def test_set_next_handler_with_cycle() -> None:
    handler = SameObjectHandler()
    with pytest.raises(
        RuntimeError, match=r"Cycle detected! the current handler cannot be its next handler"
    ):
        handler.set_next_handler(handler)


def test_set_next_handler_handler_none() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_set_next_handler_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


def test_visualize_chain_without_next_handler() -> None:
    assert SameObjectHandler().visualize_chain() == "(0): SameObjectHandler()"


def test_visualize_chain_with_next_handler() -> None:
    assert (
        SameObjectHandler(ObjectEqualHandler()).visualize_chain() == "(0): SameObjectHandler()\n"
        "(1): ObjectEqualHandler()"
    )


def test_visualize_chain_with_multiple_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    assert (
        handler.visualize_chain() == "(0): SameObjectHandler()\n"
        "(1): SameTypeHandler()\n"
        "(2): SameLengthHandler()\n"
        "(3): ObjectEqualHandler()"
    )
