from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.hashing import interface as hashing_interface

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test.

    Shared by ``test_content.py`` and ``test_uuid5.py``, whose
    generators both delegate to ``coola.hashing.hash_object`` and thus
    to this cached, otherwise test-order-dependent, singleton.
    """
    hashing_interface._default_registry._instance = None
    yield
    hashing_interface._default_registry._instance = None
