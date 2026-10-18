from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

# Hashing a large nested example can be slow on a busy CI machine, and a
# deadline only makes the tests flaky.
hypothesis.settings.register_profile("coola", deadline=None)
hypothesis.settings.load_profile("coola")
