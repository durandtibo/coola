from __future__ import annotations

import pytest

from coola.equality.handler import BaseEqualityHandler, FalseHandler, TrueHandler
from coola.equality.handler.mixin import HandlerEqualityMixin


class MyHandler(HandlerEqualityMixin, BaseEqualityHandler):
    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: object,  # noqa: ARG002
    ) -> bool:
        return True


class NamedHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Handler with extra state compared via ``_equality_attrs``."""

    def __init__(self, name: str, next_handler: BaseEqualityHandler | None = None) -> None:
        super().__init__(next_handler=next_handler)
        self.name = name

    def _equality_attrs(self) -> tuple[str, ...]:
        return ("name",)

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: object,  # noqa: ARG002
    ) -> bool:
        return True


class MultiAttrHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Handler with several extra attributes compared via
    ``_equality_attrs``."""

    def __init__(
        self, name: str, mode: str, next_handler: BaseEqualityHandler | None = None
    ) -> None:
        super().__init__(next_handler=next_handler)
        self.name = name
        self.mode = mode

    def _equality_attrs(self) -> tuple[str, ...]:
        return ("name", "mode")

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: object,  # noqa: ARG002
    ) -> bool:
        return True


##########################################
#     Tests for HandlerEqualityMixin     #
##########################################


def test_handler_equality_mixin_equal_true() -> None:
    assert MyHandler().equal(MyHandler())


def test_handler_equality_mixin_equal_true_with_next_handler() -> None:
    assert MyHandler(next_handler=FalseHandler()).equal(MyHandler(next_handler=FalseHandler()))


def test_handler_equality_mixin_equal_false_different_type() -> None:
    assert not MyHandler().equal(FalseHandler())


def test_handler_equality_mixin_equal_false_different_next_handler() -> None:
    assert not MyHandler(next_handler=TrueHandler()).equal(MyHandler(next_handler=FalseHandler()))


def test_handler_equality_mixin_equality_attrs_default() -> None:
    assert MyHandler()._equality_attrs() == ()


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(NamedHandler(name="data"), NamedHandler(name="data"), id="same name"),
        pytest.param(
            NamedHandler(name="data", next_handler=FalseHandler()),
            NamedHandler(name="data", next_handler=FalseHandler()),
            id="same name and next handler",
        ),
    ],
)
def test_handler_equality_mixin_equal_true_extra_attrs(
    handler1: NamedHandler, handler2: NamedHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(NamedHandler(name="data"), NamedHandler(name="meow"), id="different name"),
        pytest.param(
            NamedHandler(name="data", next_handler=TrueHandler()),
            NamedHandler(name="data", next_handler=FalseHandler()),
            id="same name but different next handler",
        ),
    ],
)
def test_handler_equality_mixin_equal_false_extra_attrs(
    handler1: NamedHandler, handler2: NamedHandler
) -> None:
    assert not handler1.equal(handler2)


def test_handler_equality_mixin_equality_attrs_multiple() -> None:
    assert MultiAttrHandler(name="data", mode="strict")._equality_attrs() == ("name", "mode")


def test_handler_equality_mixin_equal_true_multiple_extra_attrs() -> None:
    assert MultiAttrHandler(name="data", mode="strict").equal(
        MultiAttrHandler(name="data", mode="strict")
    )


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            MultiAttrHandler(name="data", mode="strict"),
            MultiAttrHandler(name="meow", mode="strict"),
            id="different name",
        ),
        pytest.param(
            MultiAttrHandler(name="data", mode="strict"),
            MultiAttrHandler(name="data", mode="loose"),
            id="different mode",
        ),
    ],
)
def test_handler_equality_mixin_equal_false_multiple_extra_attrs(
    handler1: MultiAttrHandler, handler2: MultiAttrHandler
) -> None:
    assert not handler1.equal(handler2)
