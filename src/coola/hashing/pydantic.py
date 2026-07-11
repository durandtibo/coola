r"""Stable hashing utilities for pydantic models, with secret-field
handling."""

from __future__ import annotations

__all__ = ["PydanticModelHasher", "hash_pydantic_model", "unwrap_secrets"]

import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from coola.hashing.base import BaseHasher
from coola.hashing.bytes import hash_bytes
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry

if is_pydantic_available():  # pragma: no cover
    from pydantic import SecretBytes, SecretStr


class PydanticModelHasher(BaseHasher[BaseModel]):
    r"""Hasher for pydantic ``BaseModel`` objects.

    This hasher computes a stable content hash of a pydantic model by
    serializing its fields to a sorted-key JSON payload and hashing
    the resulting bytes - see ``hash_pydantic_model`` for details.

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
        >>> from coola.hashing import PydanticModelHasher, HasherRegistry
        >>> class Point(BaseModel):
        ...     x: int
        ...     y: int
        ...
        >>> registry = HasherRegistry()
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
        self._on_secret = on_secret

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(on_secret={self._on_secret!r})"

    def hash(
        self,
        data: BaseModel,
        registry: HasherRegistry,  # noqa: ARG002
        length: int = 64,
    ) -> str:
        r"""Compute a deterministic hash of a pydantic model.

        Args:
            data: The pydantic model instance to hash.
            registry: The hasher registry. Unused by this hasher since
                the model is serialized and hashed directly with no
                need to dispatch to another hasher for nested data;
                accepted only to satisfy the common ``BaseHasher``
                interface.
            length: The desired length of the returned hex string. See
                ``hash_bytes`` for constraints. Defaults to 64.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``on_secret="error"`` (the default) and
                ``data`` contains a ``SecretStr``/``SecretBytes``
                field, or if ``length`` is not an even number between
                2 and 128.
        """
        return hash_pydantic_model(data, length=length, on_secret=self._on_secret)


def unwrap_secrets(
    value: Any,
    *,
    on_secret: Literal["reveal", "exclude", "error"],
) -> Any:
    """Recursively resolve ``SecretStr``/``SecretBytes`` values before
    hashing.

    Args:
        value: A (possibly nested) structure of dicts/lists/primitives,
            as produced by ``model.model_dump(mode="python")`` -
            secrets are NOT yet masked at this stage since we bypass
            JSON mode's default masking.
        on_secret: How to handle secret fields:

            - ``"reveal"``: hash the real underlying secret value (via
              ``get_secret_value()``). Use only when the hash itself
              will be treated as sensitive (e.g. never logged, stored
              encrypted-at-rest, or the secret has enough entropy that
              the hash isn't practically reversible).
            - ``"exclude"``: drop secret values entirely from the
              hashed payload - removed as dict keys, and removed as
              list elements (never replaced with a placeholder such as
              ``None``, so the "shape" of the payload doesn't leak
              where a secret used to be). The hash then reflects
              everything *except* secrets - two models differing only
              in a secret value hash the same.
            - ``"error"``: raise ``ValueError`` if any secret field is
              encountered, forcing the caller to make an explicit
              choice rather than hashing (or silently ignoring) secret
              data by accident.

    Returns:
        The value with secrets resolved (or removed) according to
        ``on_secret``. Never returns a bare secret wrapper; a
        top-level ``SecretStr``/``SecretBytes`` passed in directly is
        only valid when ``on_secret="reveal"``.

    Raises:
        ValueError: If ``on_secret="error"`` and a secret field is
            found, or if ``on_secret="exclude"`` and ``value`` is
            itself a bare secret with no enclosing dict/list to drop
            it from.
        UnicodeDecodeError: If ``on_secret="reveal"``, the secret is a
            ``SecretBytes``, and its content isn't valid UTF-8.

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
    if isinstance(value, (SecretStr, SecretBytes)):
        if on_secret == "reveal":  # noqa: S105
            secret = value.get_secret_value()
            return secret.decode("utf-8") if isinstance(secret, bytes) else secret
        if on_secret == "exclude":  # noqa: S105
            msg = (
                "Cannot exclude a top-level secret value with no enclosing "
                "dict/list to drop it from."
            )
            raise ValueError(msg)
        msg = (
            "Encountered a SecretStr/SecretBytes field while hashing; "
            "pass on_secret='reveal' or on_secret='exclude' to hash_model()."
        )
        raise ValueError(msg)

    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(v, (SecretStr, SecretBytes)) and on_secret == "exclude":  # noqa: S105
                continue
            result[k] = unwrap_secrets(v, on_secret=on_secret)
        return result

    if isinstance(value, list):
        result_list: list[Any] = []
        for v in value:
            if isinstance(v, (SecretStr, SecretBytes)) and on_secret == "exclude":  # noqa: S105
                continue
            result_list.append(unwrap_secrets(v, on_secret=on_secret))
        return result_list

    return value


def hash_pydantic_model(
    model: BaseModel,
    *,
    length: int = 64,
    on_secret: Literal["reveal", "exclude", "error"] = "error",  # noqa: S107
) -> str:
    """Compute a stable content hash of a pydantic model.

    Serializes the model to a sorted-key JSON payload and hashes the
    resulting bytes with BLAKE2b (via ``hash_bytes``), so the hash is
    stable regardless of field declaration order or dict insertion
    order.

    Note:
        Serialization uses ``model.model_dump(mode="python")`` rather
        than ``mode="json"``, since JSON mode pre-masks
        ``SecretStr``/``SecretBytes`` fields to ``"**********"``
        before we can apply ``on_secret`` ourselves. One consequence:
        other special types (e.g. ``datetime``, ``UUID``, ``Decimal``,
        ``Enum``) aren't normalized through pydantic's JSON encoders -
        they fall back to plain ``str()`` via ``json.dumps(...,
        default=str)`` instead. This is usually fine for hashing
        purposes (it's still deterministic), but the exact string form
        may differ from what ``mode="json"`` would have produced.

    Args:
        model: The pydantic model instance to hash.
        length: The desired length of the returned hex string, passed
            through to ``hash_bytes``. Must be an even number between
            2 and 128 inclusive. Defaults to 64.
        on_secret: How ``SecretStr``/``SecretBytes`` fields are
            handled - see ``unwrap_secrets`` for details. Defaults to
            ``"error"`` so secret fields are never silently
            included/excluded/masked without an explicit decision.

    Returns:
        The hex digest of the hash, as a string.

    Raises:
        ValueError: If ``on_secret="error"`` (the default) and the
            model contains a ``SecretStr``/``SecretBytes`` field, or
            if ``length`` is not an even number between 2 and 128.

    Example:
        ```pycon
        >>> from pydantic import BaseModel, SecretStr
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
    payload = model.model_dump(mode="python")
    resolved = unwrap_secrets(payload, on_secret=on_secret)
    serialized = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(serialized.encode("utf-8"), length=length)
