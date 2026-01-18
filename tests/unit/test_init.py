from __future__ import annotations

import coola


def test_version_exists() -> None:
    assert hasattr(coola, "__version__")
    assert isinstance(coola.__version__, str)
