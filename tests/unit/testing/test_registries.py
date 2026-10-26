from __future__ import annotations

import pytest

from coola.equality.tester import DefaultEqualityTester, get_default_registry


class Point:
    pass


@pytest.mark.usefixtures("isolated_default_registries")
def test_isolated_default_registries_registers_a_type() -> None:
    get_default_registry().register(Point, DefaultEqualityTester())
    assert get_default_registry().has_equality_tester(Point)
