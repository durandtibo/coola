r"""Stable hashing utilities for pydantic models, with secret-field
handling."""

from __future__ import annotations

__all__ = ["PydanticModelHasher", "hash_pydantic_model", "unwrap_secrets"]

import threading
from typing import TYPE_CHECKING, Any, Literal, assert_never

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
    from pydantic import SecretBytes, SecretStr

    _SECRET_TYPES = (SecretStr, SecretBytes)
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import pydantic

    _SECRET_TYPES = ()


class PydanticModelHasher(InlineDisplayMixin, BaseHasher[pydantic.BaseModel]):
    r"""Hasher for pydantic ``BaseModel`` objects.

    This hasher computes a stable content hash of a pydantic model from
    its qualified class name and its fields, with every field value
    hashed through the registry - see ``hash_pydantic_model`` for
    details. Nested models found directly in fields or in plain
    containers (``dict``, ``list``, ``tuple``, ``set``, ``frozenset``)
    are hashed with the same ``on_secret`` policy.

    Args:
        on_secret: How ``SecretStr``/``SecretBytes`` fields are
            handled when hashing - one of ``"reveal"``, ``"exclude"``,
            or ``"error"``. See ``unwrap_secrets`` for the semantics of
            each option. Defaults to ``"error"`` so secret fields are
            never silently included/excluded without an explicit
            decision.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.hashing import PydanticModelHasher, get_default_registry
        >>> class Point(BaseModel):
        ...     x: int
        ...     y: int
        ...
        >>> registry = get_default_registry()
        >>> hasher = PydanticModelHasher()
        >>> hasher
        PydanticModelHasher(on_secret='error')
        >>> len(hasher.hash(Point(x=1, y=2), registry=registry))
        64

        ```
    """

    def __init__(
        self,
        on_secret: Literal["reveal", "exclude", "error"] = "error",  # noqa: S107
    ) -> None:
        self._on_secret: Literal["reveal", "exclude", "error"] = on_secret

    def _display_fields(self) -> dict[str, Any]:
        return {"on_secret": self._on_secret}

    def hash(
        self,
        data: pydantic.BaseModel,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        r"""Compute a deterministic hash of a pydantic model.

        Args:
            data: The pydantic model instance to hash.
            registry: The hasher registry used to hash the field
                values.
            length: The desired length of the returned hex string. See
                ``hash_string`` for constraints. Defaults to 64.
            ignore_unhashable: If ``True``, field values for which no
                hasher is registered are replaced by a deterministic
                placeholder hash instead of raising an error.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``on_secret="error"`` (the default) and
                ``data`` contains a ``SecretStr``/``SecretBytes``
                field, or if ``length`` is not an even number between
                2 and 128.
            KeyError: If a field value has a type for which no hasher
                is registered in ``registry`` and ``ignore_unhashable``
                is ``False``.
        """
        return hash_pydantic_model(
            data,
            length=length,
            on_secret=self._on_secret,
            registry=registry,
            ignore_unhashable=ignore_unhashable,
        )


def unwrap_secrets(
    value: Any,
    *,
    on_secret: Literal["reveal", "exclude", "error"],
    on_model: Callable[[pydantic.BaseModel], Any] | None = None,
) -> Any:
    """Recursively resolve ``SecretStr``/``SecretBytes`` values before
    hashing.

    Only plain ``dict``, ``list``, ``tuple``, ``set`` and ``frozenset``
    containers are traversed, and they are rebuilt with the same type so
    the type information is preserved. Any other value is returned
    unchanged.

    Args:
        value: A (possibly nested) structure of containers and
            primitives, e.g. the field values of a model.
        on_secret: How to handle secret fields:

            - ``"reveal"``: hash the real underlying secret value (via
              ``get_secret_value()``). Use only when the hash itself
              will be treated as sensitive (e.g. never logged, stored
              encrypted-at-rest, or the secret has enough entropy that
              the hash isn't practically reversible).
            - ``"exclude"``: drop secret values entirely from the
              hashed payload - removed as dict values, and removed as
              sequence/set elements (never replaced with a placeholder
              such as ``None``, so the "shape" of the payload doesn't
              leak where a secret used to be). The hash then reflects
              everything *except* secrets - two models differing only
              in a secret value hash the same.
            - ``"error"``: raise ``ValueError`` if any secret field is
              encountered, forcing the caller to make an explicit
              choice rather than hashing (or silently ignoring) secret
              data by accident. The message never contains the secret.
        on_model: If given, called with every nested pydantic model
            found in the traversed containers, and its return value
            replaces the model. ``hash_pydantic_model`` uses it to hash
            nested models with the same ``on_secret`` policy.

    Returns:
        The value with secrets resolved (or removed) according to
        ``on_secret``. Never returns a bare secret wrapper; a
        top-level ``SecretStr``/``SecretBytes`` passed in directly is
        only valid when ``on_secret="reveal"``.

    Raises:
        ValueError: If ``on_secret="error"`` and a secret field is
            found, or if ``on_secret="exclude"`` and ``value`` is
            itself a bare secret with no enclosing container to drop
            it from.

    Example:
        ```pycon
        >>> from pydantic import SecretStr
        >>> from coola.hashing.pydantic import unwrap_secrets
        >>> unwrap_secrets({"user": "alice", "pwd": SecretStr("hunter2")}, on_secret="reveal")
        {'user': 'alice', 'pwd': 'hunter2'}
        >>> unwrap_secrets({"user": "alice", "pwd": SecretStr("hunter2")}, on_secret="exclude")
        {'user': 'alice'}

        ```
    """
    if isinstance(value, _SECRET_TYPES):
        if on_secret == "reveal":  # noqa: S105
            return value.get_secret_value()
        if on_secret == "exclude":  # noqa: S105
            msg = (
                "Cannot exclude a top-level secret value with no enclosing "
                "dict/list/tuple/set to drop it from."
            )
            raise ValueError(msg)
        if on_secret == "error":  # noqa: S105
            msg = (
                "Encountered a SecretStr/SecretBytes field while hashing; "
                "pass on_secret='reveal' or on_secret='exclude' to hash_pydantic_model()."
            )
            raise ValueError(msg)
        assert_never(on_secret)

    def keep(item: Any) -> bool:
        return not (isinstance(item, _SECRET_TYPES) and on_secret == "exclude")  # noqa: S105

    if on_model is not None and isinstance(value, pydantic.BaseModel):
        return on_model(value)

    value_type = type(value)
    if value_type is dict:
        return {
            k: unwrap_secrets(v, on_secret=on_secret, on_model=on_model)
            for k, v in value.items()
            if keep(v)
        }
    if value_type in {list, tuple, set, frozenset}:
        return value_type(
            unwrap_secrets(v, on_secret=on_secret, on_model=on_model) for v in value if keep(v)
        )
    return value


# Ids of the models being hashed by the current thread, to detect a model that
# contains itself. The state is thread-local, like the cycle guard of
# ``HasherRegistry.hash``, so a model hashed by two threads at the same time
# is not reported as a cycle. The nested models are reached through the
# ``on_model`` callback of ``unwrap_secrets``, which has no argument to carry
# the set of active models.
_state = threading.local()


def hash_pydantic_model(
    model: pydantic.BaseModel,
    *,
    length: int = 64,
    on_secret: Literal["reveal", "exclude", "error"] = "error",  # noqa: S107
    registry: HasherRegistry | None = None,
    ignore_unhashable: bool = False,
) -> str:
    """Compute a stable content hash of a pydantic model.

    The hash combines the qualified class name of the model, so two
    different model classes with the same field values hash
    differently, with the hash of a mapping from field name to field
    value (extra fields included). Every value is hashed through the
    registry, so it follows the same rules as any other data (type
    sensitivity, order-insensitive sets, ...), and the hash is stable
    regardless of field declaration order.

    The nested models found in the dicts, lists, tuples and sets of the
    fields are hashed with the same ``on_secret`` policy. A model nested
    in another kind of object (a dataclass, for example) is hashed by the
    registry with the default policy.

    Args:
        model: The pydantic model instance to hash.
        length: The desired length of the returned hex string. Must be
            an even number between 2 and 128 inclusive. Defaults to
            64.
        on_secret: How ``SecretStr``/``SecretBytes`` fields are
            handled - see ``unwrap_secrets`` for details. Defaults to
            ``"error"`` so secret fields are never silently
            included/excluded/masked without an explicit decision.
        registry: The registry used to hash the field values. If
            ``None``, the default registry is used.
        ignore_unhashable: If ``True``, field values for which no
            hasher is registered are replaced by a deterministic
            placeholder hash instead of raising an error.

    Returns:
        The hex digest of the hash, as a string.

    Raises:
        ValueError: If ``on_secret="error"`` (the default) and the
            model contains a ``SecretStr``/``SecretBytes`` field, if
            the model contains itself, or if ``length`` is not an even
            number between 2 and 128.
        KeyError: If a field value has a type for which no hasher is
            registered in ``registry`` and ``ignore_unhashable`` is
            ``False``.

    Example:
        ```pycon
        >>> from pydantic import BaseModel, SecretStr
        >>> from coola.hashing import hash_pydantic_model
        >>> class Creds(BaseModel):
        ...     username: str
        ...     password: SecretStr
        ...
        >>> creds = Creds(username="alice", password="hunter2")
        >>> hash_pydantic_model(creds, on_secret="exclude") == hash_pydantic_model(
        ...     Creds(username="alice", password="different"), on_secret="exclude"
        ... )
        True

        ```
    """
    if registry is None:
        from coola.hashing.interface import get_default_registry  # noqa: PLC0415

        registry = get_default_registry()
    active: set[int] | None = getattr(_state, "active", None)
    if active is None:
        active = _state.active = set()
    if id(model) in active:
        msg = f"Cannot hash a self-referencing {type(model)!r}."
        raise ValueError(msg)
    active.add(id(model))
    try:
        return _hash_pydantic_model(model, length, on_secret, registry, ignore_unhashable)
    finally:
        active.discard(id(model))


def _hash_pydantic_model(
    model: pydantic.BaseModel,
    length: int,
    on_secret: Literal["reveal", "exclude", "error"],
    registry: HasherRegistry,
    ignore_unhashable: bool,
) -> str:
    cls = type(model)
    fields = {name: getattr(model, name) for name in cls.model_fields}
    fields.update(model.model_extra or {})

    def hash_nested(nested: pydantic.BaseModel) -> str:
        return hash_pydantic_model(
            nested,
            length=length,
            on_secret=on_secret,
            registry=registry,
            ignore_unhashable=ignore_unhashable,
        )

    resolved = unwrap_secrets(fields, on_secret=on_secret, on_model=hash_nested)
    digest = registry.hash(resolved, length=length, ignore_unhashable=ignore_unhashable)
    return hash_tagged(f"pydantic:{cls.__module__}.{cls.__qualname__}", digest, length=length)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the pydantic models.

    Returns:
        The hashers of the pydantic models and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the pydantic models.
    """
    if not is_pydantic_available():
        return {}
    # Pydantic models - the fields are hashed through the registry.
    return {pydantic.BaseModel: PydanticModelHasher()}
