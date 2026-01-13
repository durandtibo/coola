from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.handler import (
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


def test_chain_1(config: EqualityConfig2) -> None:
    handler = SameObjectHandler()
    handler.chain(TrueHandler())
    assert handler.next_handler == TrueHandler()
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)


def test_chain_multiple(config: EqualityConfig2) -> None:
    handler = SameObjectHandler()
    handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
    assert handler.next_handler == SameTypeHandler()
    assert handler.handle(actual=[1, 2, 3], expected=[1, 2, 3], config=config)
