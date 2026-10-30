r"""Stable hashing utilities for pydantic models, with secret
handling."""

from __future__ import annotations

__all__ = ["PydanticModelHasher", "SecretHasher", "get_secret_hashers"]

from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Literal

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node, tagged
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING:
    from coola.hashing.base import Hasher

if TYPE_CHECKING or is_pydantic_available():
    import pydantic
    from pydantic import SecretBytes, SecretStr
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import SecretBytes, SecretStr, pydantic

# The policies are compared and defaulted by name: a string literal assigned or compared to
# a name that contains "secret" is reported by the rules S105 and S107.
_REVEAL: Final = "reveal"
_EXCLUDE: Final = "exclude"
_ERROR: Final = "error"


class PydanticModelHasher(InlineDisplayMixin, BaseHasher[pydantic.BaseModel]):
    r"""Hasher for pydantic ``BaseModel`` objects.

    The model is described by its qualified class name and its fields
    (extra fields included), with every field value hashed through the
    registry. The fields are unordered, so the hash does not depend on
    the order of the declaration. The secrets (``SecretStr`` and
    ``SecretBytes``) are values like the others: their hasher decides
    what happens (see ``SecretHasher``), wherever the secret is.

    Example:
        ```pycon
        >>> from pydantic import BaseModel
        >>> from coola.hashing import PydanticModelHasher
        >>> class Point(BaseModel):
        ...     x: int
        ...     y: int
        ...
        >>> hasher = PydanticModelHasher()
        >>> hasher
        PydanticModelHasher()
        >>> hasher.encode(Point(x=1, y=2)).ordered
        False

        ```
    """

    def encode(self, data: pydantic.BaseModel) -> Encoding:
        r"""Describe a pydantic model.

        Args:
            data: The pydantic model instance to describe.

        Returns:
            The unordered ``Node`` of the field names and values (extra
            fields included), with the qualified class name in the tag.
        """
        cls = type(data)
        fields = {name: getattr(data, name) for name in cls.model_fields}
        fields.update(data.model_extra or {})
        return Node(
            f"pydantic:{cls.__module__}.{cls.__qualname__}",
            chain.from_iterable(fields.items()),
            False,
            2,
        )


class SecretHasher(InlineDisplayMixin, BaseHasher["SecretStr | SecretBytes"]):
    r"""Hasher for ``SecretStr`` and ``SecretBytes`` values.

    A secret is a value with a hasher, so the policy is the one of the
    registry and applies wherever the secret is (in a model, a ``dict``,
    a ``list``, a dataclass...).

    Args:
        on_secret: What to do with a secret:

            - ``"error"``: raise a ``ValueError``. The message never
              contains the secret. It forces an explicit decision
              instead of hashing, or silently ignoring, secret data by
              accident.
            - ``"reveal"``: hash the real value (``get_secret_value()``).
              Use it only when the hash itself is treated as sensitive.
            - ``"exclude"``: hash a constant, so the value is never
              hashed and two objects that differ only by a secret have
              the same hash. The position of the secret is kept: it is
              part of the schema of the class.

            Defaults to ``"error"``.

    Raises:
        ValueError: If ``on_secret`` is not one of the three values.

    Example:
        ```pycon
        >>> from pydantic import SecretStr
        >>> from coola.hashing import SecretHasher
        >>> hasher = SecretHasher("exclude")
        >>> hasher
        SecretHasher(on_secret='exclude')
        >>> hasher.encode(SecretStr("a")) == hasher.encode(SecretStr("b"))
        True

        ```
    """

    def __init__(
        self,
        on_secret: Literal["reveal", "exclude", "error"] = _ERROR,
    ) -> None:
        if on_secret not in {"reveal", "exclude", "error"}:
            msg = f"on_secret must be 'reveal', 'exclude' or 'error', got {on_secret!r}."
            raise ValueError(msg)
        self._on_secret: Literal["reveal", "exclude", "error"] = on_secret

    def _display_fields(self) -> dict[str, Any]:
        return {"on_secret": self._on_secret}

    def encode(self, data: SecretStr | SecretBytes) -> Encoding:
        r"""Describe a secret.

        Args:
            data: The secret to describe.

        Returns:
            The leaf ``tagged("pydantic.SecretStr", value)`` for
            ``"reveal"`` and ``tagged("pydantic.SecretStr")`` for
            ``"exclude"`` (``pydantic.SecretBytes`` for a
            ``SecretBytes``).

        Raises:
            ValueError: If ``on_secret="error"``. The message never
                contains the secret.
        """
        tag = f"pydantic.{type(data).__name__}"
        if self._on_secret == _REVEAL:
            return tagged(tag, data.get_secret_value())
        if self._on_secret == _EXCLUDE:
            return tagged(tag)
        msg = (
            "Encountered a SecretStr/SecretBytes value while hashing; register "
            "SecretHasher('reveal') or SecretHasher('exclude') for it "
            "(see `get_secret_hashers`)."
        )
        raise ValueError(msg)


def get_secret_hashers(
    on_secret: Literal["reveal", "exclude", "error"] = _ERROR,
) -> dict[type, Hasher[Any]]:
    r"""Get the hashers of ``SecretStr`` and ``SecretBytes`` for a
    policy.

    Args:
        on_secret: The policy of the hashers, see ``SecretHasher``.
            Defaults to ``"error"``.

    Returns:
        The mapping from ``SecretStr`` and ``SecretBytes`` to a
        ``SecretHasher``, to give to ``HasherRegistry.register_many``.

    Example:
        ```pycon
        >>> from pydantic import BaseModel, SecretStr
        >>> from coola.hashing import get_default_registry, get_secret_hashers
        >>> class Creds(BaseModel):
        ...     user: str
        ...     password: SecretStr
        ...
        >>> registry = get_default_registry().copy()
        >>> registry.register_many(get_secret_hashers("exclude"), exist_ok=True)
        >>> registry.hash(Creds(user="a", password="x")) == registry.hash(
        ...     Creds(user="a", password="y")
        ... )
        True

        ```
    """
    hasher = SecretHasher(on_secret)
    return {SecretStr: hasher, SecretBytes: hasher}


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the pydantic models and secrets.

    Returns:
        The hashers of the pydantic models and secrets and their
        associated types. The secrets are not hashed by default
        (``on_secret="error"``).

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the pydantic models.
    """
    if not is_pydantic_available():
        return {}
    return {pydantic.BaseModel: PydanticModelHasher(), **get_secret_hashers("error")}
