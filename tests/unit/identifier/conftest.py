from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.hashing import get_default_registry

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test.

    Shared by ``test_content.py`` and ``test_uuid5.py``, whose
    generators both delegate to ``coola.hashing.hash_object`` and thus
    to this cached, otherwise test-order-dependent, singleton.
    """
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
