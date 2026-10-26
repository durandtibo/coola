r"""Helpers shared by the equality testers: the path parts, the keys of
the sets and the mappings, and the kernels of the numbers and the
arrays."""

from __future__ import annotations

__all__ = [
    "AttributeName",
    "MaxDepthExceededError",
    "comparison_key",
    "get_object_attributes",
    "is_exact_scalar_sequence",
]

from coola.equality._helpers.object import get_object_attributes
from coola.equality._helpers.utils import (
    AttributeName,
    MaxDepthExceededError,
    comparison_key,
    is_exact_scalar_sequence,
)
