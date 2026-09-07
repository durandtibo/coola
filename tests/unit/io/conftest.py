from __future__ import annotations

import pytest

from coola.equality.tester import EqualNanEqualityTester, get_default_registry
from coola.io import BaseLoader, BaseSaver


@pytest.fixture(autouse=True)
def _ensure_io_equality_testers_registered() -> None:
    # Some test modules reset the default equality tester registry singleton
    # (e.g. to test its construction from scratch), which drops the
    # registrations performed once at import time in ``coola.io.base``.
    # Re-register them here so io tests do not depend on module import order.
    get_default_registry().register_many(
        {BaseLoader: EqualNanEqualityTester(), BaseSaver: EqualNanEqualityTester()}, exist_ok=True
    )
