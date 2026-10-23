from __future__ import annotations

import contextlib

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    HandlerCycleError,
    ObjectEqualHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
    create_chain,
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


def test_chain_1_handler_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain(ObjectEqualHandler())
    assert handler.next_handler.equal(ObjectEqualHandler())


def test_chain_1_handler_end_of_chain() -> None:
    handler = SameObjectHandler()
    handler.chain(ObjectEqualHandler())
    assert handler.next_handler.next_handler is None


def test_chain_1_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain(ObjectEqualHandler())
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def make_chain() -> SameObjectHandler:
    handler = SameObjectHandler()
    handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(ObjectEqualHandler())
    return handler


def test_chain_multiple_handlers_first_next_handler() -> None:
    handler = make_chain()
    assert handler.next_handler.equal(SameTypeHandler(SameLengthHandler(ObjectEqualHandler())))


def test_chain_multiple_handlers_second_next_handler() -> None:
    handler = make_chain()
    assert handler.next_handler.next_handler.equal(SameLengthHandler(ObjectEqualHandler()))


def test_chain_multiple_handlers_third_next_handler() -> None:
    handler = make_chain()
    assert handler.next_handler.next_handler.next_handler.equal(ObjectEqualHandler())


def test_chain_multiple_handlers_end_of_chain() -> None:
    handler = make_chain()
    assert handler.next_handler.next_handler.next_handler.next_handler is None


def test_chain_multiple_handlers(config: EqualityConfig) -> None:
    handler = make_chain()
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_with_cycle() -> None:
    handler = SameObjectHandler()
    with pytest.raises(
        HandlerCycleError, match=r"Cycle detected! the current handler cannot be its next handler"
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


def test_chain_all_1_handler_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain_all(ObjectEqualHandler())
    assert handler.next_handler.equal(ObjectEqualHandler())


def test_chain_all_1_handler_end_of_chain() -> None:
    handler = SameObjectHandler()
    handler.chain_all(ObjectEqualHandler())
    assert handler.next_handler.next_handler is None


def test_chain_all_1_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain_all(ObjectEqualHandler())
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def make_chain_all() -> SameObjectHandler:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    return handler


def test_chain_all_multiple_handlers_first_next_handler() -> None:
    handler = make_chain_all()
    assert handler.next_handler.equal(SameTypeHandler(SameLengthHandler(ObjectEqualHandler())))


def test_chain_all_multiple_handlers_second_next_handler() -> None:
    handler = make_chain_all()
    assert handler.next_handler.next_handler.equal(SameLengthHandler(ObjectEqualHandler()))


def test_chain_all_multiple_handlers_third_next_handler() -> None:
    handler = make_chain_all()
    assert handler.next_handler.next_handler.next_handler.equal(ObjectEqualHandler())


def test_chain_all_multiple_handlers_end_of_chain() -> None:
    handler = make_chain_all()
    assert handler.next_handler.next_handler.next_handler.next_handler is None


def test_chain_all_multiple_handlers(config: EqualityConfig) -> None:
    handler = make_chain_all()
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_all_no_handlers_returns_the_handler() -> None:
    handler = SameObjectHandler()
    assert handler.chain_all() is handler


def test_chain_all_no_handlers_has_no_next_handler() -> None:
    handler = SameObjectHandler()
    handler.chain_all()
    assert handler.next_handler is None


def test_chain_all_no_handlers_chain_length() -> None:
    handler = SameObjectHandler()
    handler.chain_all()
    assert handler.get_chain_length() == 1


def test_chain_all_with_cycle() -> None:
    handler = SameObjectHandler()
    with pytest.raises(
        HandlerCycleError, match=r"Cycle detected! the current handler cannot be its next handler"
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
        HandlerCycleError, match=r"Cycle detected! the current handler cannot be its next handler"
    ):
        handler.set_next_handler(handler)


def test_set_next_handler_with_indirect_cycle() -> None:
    first = SameObjectHandler()
    second = SameTypeHandler()
    first.set_next_handler(second)
    with pytest.raises(HandlerCycleError, match=r"Cycle detected"):
        second.set_next_handler(first)


def test_set_next_handler_with_indirect_cycle_does_not_set_the_next_handler() -> None:
    first = SameObjectHandler()
    second = SameTypeHandler()
    first.set_next_handler(second)
    with contextlib.suppress(HandlerCycleError):
        second.set_next_handler(first)
    assert second.next_handler is None


def test_set_next_handler_with_indirect_cycle_keeps_the_chain_length() -> None:
    first = SameObjectHandler()
    second = SameTypeHandler()
    first.set_next_handler(second)
    with contextlib.suppress(HandlerCycleError):
        second.set_next_handler(first)
    assert first.get_chain_length() == 2


def test_create_chain_with_repeated_handler() -> None:
    handler = SameObjectHandler()
    with pytest.raises(HandlerCycleError, match=r"Cycle detected"):
        create_chain(handler, SameTypeHandler(), handler)


def test_set_next_handler_handler_none() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_set_next_handler_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


def test_validate_chain_1_handler() -> None:
    handler = SameObjectHandler()
    handler.validate_chain()


def test_validate_chain_multiple_handlers() -> None:
    handler = SameObjectHandler()
    handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
    handler.validate_chain()


def test_validate_chain_multiple_handlers_with_cycle() -> None:
    handler = SameObjectHandler()
    last = handler.chain_all(SameTypeHandler(), SameLengthHandler())
    last._next_handler = handler  # bypass the check to create a cycle
    with pytest.raises(HandlerCycleError, match=r"Cycle detected in the chain"):
        handler.validate_chain()


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


def test_handler_cycle_error_is_value_error() -> None:
    assert issubclass(HandlerCycleError, ValueError)


def test_handler_cycle_error_is_runtime_error() -> None:
    assert issubclass(HandlerCycleError, RuntimeError)


def test_set_next_handler_self_raises_handler_cycle_error() -> None:
    handler = TrueHandler()
    with pytest.raises(HandlerCycleError, match="Cycle detected"):
        handler.set_next_handler(handler)


def test_validate_chain_cycle_raises_handler_cycle_error() -> None:
    first, second = TrueHandler(), TrueHandler()
    first.set_next_handler(second)
    second._next_handler = first  # bypass the check to create a cycle
    with pytest.raises(HandlerCycleError, match="Cycle detected in the chain"):
        first.validate_chain()


class FalsyHandler(TrueHandler):
    r"""A handler that is falsy, like a handler with an empty
    ``__len__``."""

    def __bool__(self) -> bool:
        return False

    def __len__(self) -> int:
        return 0


def test_falsy_handler_can_be_the_next_handler() -> None:
    handler = TrueHandler(next_handler=FalsyHandler())
    assert isinstance(handler.next_handler, FalsyHandler)


def test_falsy_handler_get_chain_length() -> None:
    handler = TrueHandler(next_handler=FalsyHandler(next_handler=FalseHandler()))
    assert handler.get_chain_length() == 3


def test_falsy_handler_validate_chain() -> None:
    TrueHandler(next_handler=FalsyHandler(next_handler=FalseHandler())).validate_chain()


def test_falsy_handler_validate_chain_detects_a_cycle() -> None:
    first = TrueHandler()
    second = FalsyHandler()
    first.set_next_handler(second)
    second._next_handler = first
    with pytest.raises(HandlerCycleError, match="Cycle detected"):
        first.validate_chain()


def test_falsy_handler_visualize_chain() -> None:
    handler = TrueHandler(next_handler=FalsyHandler(next_handler=FalseHandler()))
    assert (
        handler.visualize_chain() == "(0): TrueHandler()\n(1): FalsyHandler()\n(2): FalseHandler()"
    )


def test_falsy_handler_repr_shows_the_next_handler() -> None:
    assert repr(TrueHandler(next_handler=FalsyHandler())) == (
        "TrueHandler(next_handler=FalsyHandler())"
    )


def test_falsy_handler_is_called_by_the_previous_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler(next_handler=FalsyHandler())
    assert handler.handle(1, 2, config)
