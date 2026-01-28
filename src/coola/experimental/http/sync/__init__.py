r"""Contain functionalities to send sync HTTP requests with httpx
library."""

from __future__ import annotations

__all__ = ["get_with_automatic_retry", "post_with_automatic_retry"]

from coola.experimental.http.sync.get import get_with_automatic_retry
from coola.experimental.http.sync.post import post_with_automatic_retry
