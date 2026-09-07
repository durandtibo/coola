r"""Contain data loaders and savers."""

from __future__ import annotations

__all__ = [
    "BaseFileSaver",
    "BaseLoader",
    "BaseSaver",
    "add_uuid_suffix",
    "is_loader_config",
    "is_saver_config",
    "resolve_loader",
    "resolve_saver",
]

from coola.io.base import (
    BaseFileSaver,
    BaseLoader,
    BaseSaver,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)
from coola.io.utils import add_uuid_suffix
