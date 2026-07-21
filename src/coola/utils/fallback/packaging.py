r"""Contain fallback implementations used when ``packaging`` dependency
is not available."""

from __future__ import annotations

__all__ = ["Version"]

from coola.utils.fallback.factory import make_fake_class
from coola.utils.imports import raise_packaging_missing_error

Version = make_fake_class(raise_packaging_missing_error)
