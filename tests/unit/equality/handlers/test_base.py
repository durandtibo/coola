from __future__ import annotations

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import (
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)
from coola.equality.testers import EqualityTester


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


def test_chain_1(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain(TrueHandler())
    assert handler.next_handler == TrueHandler()
    assert handler.handle(object1=[1, 2, 3], object2=[1, 2, 3], config=config)


def test_chain_multiple(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
    assert handler.next_handler == SameTypeHandler()
    assert handler.handle(object1=[1, 2, 3], object2=[1, 2, 3], config=config)
