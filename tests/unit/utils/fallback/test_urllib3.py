from __future__ import annotations

import pytest

from coola.utils.fallback.urllib3 import Retry


def test_retry() -> None:
    with pytest.raises(RuntimeError, match=r"'urllib3' package is required but not installed."):
        Retry()


def test_retry_with_args() -> None:
    with pytest.raises(RuntimeError, match=r"'urllib3' package is required but not installed."):
        Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
