r"""Contain helpers shared across the hashing unit tests."""

from __future__ import annotations

__all__ = ["Unhashable"]


class Unhashable:
    r"""A type with no registered hasher."""
