r"""Provide deterministic identifiers for nested data."""

from __future__ import annotations

__all__ = ["generate_content_id", "generate_stable_uuid", "generate_ulid"]

from coola.identifier.content import generate_content_id
from coola.identifier.ulid import generate_ulid
from coola.identifier.uuid import generate_stable_uuid
