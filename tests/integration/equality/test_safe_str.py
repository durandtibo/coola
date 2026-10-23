from __future__ import annotations

from dataclasses import dataclass

import pytest

from coola.equality import assert_objects_equal, compare
from tests.unit.equality.utils import BadStr


@dataclass
class Holder:
    value: object


def test_compare_dataclass_field_with_failing_str() -> None:
    assert not compare(Holder(BadStr(1)), Holder(BadStr(2)))


def test_assert_objects_equal_dataclass_field_with_failing_str_raises_assertion_error() -> None:
    with pytest.raises(AssertionError, match=r"objects are not equal at actual\.value"):
        assert_objects_equal(Holder(BadStr(1)), Holder(BadStr(2)))
