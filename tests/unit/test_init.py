from __future__ import annotations

import coola


def test_version_exists() -> None:
    assert hasattr(coola, "__version__")


def test_version_is_a_str() -> None:
    assert isinstance(coola.__version__, str)
