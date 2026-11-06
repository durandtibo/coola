from __future__ import annotations

import pytest

from coola.equality.tester import EqualEqualityTester

#########################################
#     Tests for BaseEqualityTester      #
#########################################


def test_base_equality_tester_hash_raises() -> None:
    with pytest.raises(TypeError, match=r"unhashable type: 'EqualEqualityTester'"):
        hash(EqualEqualityTester())
