r"""Contain fallback implementations used when ``packaging`` dependency
is not available."""

from __future__ import annotations

__all__ = ["InvalidVersion", "Version"]

from coola.utils.fallback.factory import make_fake_class

Version: type = make_fake_class("packaging")
InvalidVersion: type = ValueError
