r"""Define a transformer for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassTransformer"]

import copy
import dataclasses
from typing import Any

from coola.display import InlineDisplayMixin
from coola.recursive.base import BaseTransformer, Expansion


class DataclassTransformer(InlineDisplayMixin, BaseTransformer[Any]):
    r"""Transformer for dataclass instances that recursively transforms
    the fields.

    The fields are the children. The instance is rebuilt with a shallow
    copy in which every field is set with ``object.__setattr__``, so a
    frozen or slotted dataclass is supported, the fields with
    ``init=False`` are transformed, an ``InitVar`` without a default
    value is not a problem, the input is not modified, and no user code
    (``__init__``, ``__post_init__``) runs during the traversal.

    Notes:
        - ``__post_init__`` is not run again, so a field derived from
          another field in ``__post_init__`` is not recomputed: it can
          become inconsistent with the field it was derived from after
          the function is applied. Register a transformer for the type
          to rebuild it differently.
        - It is registered as a fallback for ``dataclasses.is_dataclass``
          in the default registry. A transformer registered for the type of the
          dataclass has priority; register ``DefaultTransformer`` to
          keep a dataclass as a leaf.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.recursive import (
        ...     DataclassTransformer,
        ...     DefaultTransformer,
        ...     SequenceTransformer,
        ...     TransformerRegistry,
        ... )
        >>> @dataclass
        ... class Config:
        ...     lr: float
        ...     layers: list
        ...
        >>> transformer = DataclassTransformer()
        >>> registry = TransformerRegistry(
        ...     {object: DefaultTransformer(), list: SequenceTransformer(), Config: transformer}
        ... )
        >>> registry.transform(Config(0.1, [1, 2]), str)
        Config(lr='0.1', layers=['1', '2'])

        ```
    """

    def expand(self, data: Any) -> Expansion | None:
        r"""Expand an object into its children.

        Args:
            data: The object.

        Returns:
            ``None`` if the object is a leaf. Otherwise the children and
                the function that rebuilds a copy of the instance from
                the transformed children.
        """
        names = [field.name for field in dataclasses.fields(data)]

        def rebuild(items: list[Any]) -> Any:
            r"""Rebuild the dataclass instance.

            Args:
                items: The transformed values of the fields, in the
                    order of the fields.

            Returns:
                A copy of the instance with the new values.
            """
            result = copy.copy(data)
            for name, item in zip(names, items, strict=True):
                object.__setattr__(result, name, item)
            return result

        return [getattr(data, name) for name in names], rebuild
