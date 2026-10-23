from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pytest

from coola.summary import SummarizerRegistry, get_default_registry, summarize
from coola.summary.pydantic import PydanticModelSummarizer, get_default_summarizers
from coola.testing.fixtures import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, SecretStr
else:
    BaseModel, ConfigDict, SecretStr = object, dict, object


class Point(BaseModel):
    x: float
    y: float


class User(BaseModel):
    name: str
    password: SecretStr
    token: Optional[SecretStr] = None  # noqa: UP045


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: int


class Outer(BaseModel):
    point: Point


@pytest.fixture
def registry() -> SummarizerRegistry:
    return get_default_registry()


@pydantic_available
def test_pydantic_model_summarizer_repr() -> None:
    assert repr(PydanticModelSummarizer()) == "PydanticModelSummarizer(max_items=5, num_spaces=2)"


@pydantic_available
def test_pydantic_model_summarizer_summarize(registry: SummarizerRegistry) -> None:
    assert PydanticModelSummarizer().summarize(Point(x=1.0, y=2.0), registry) == (
        f"{Point} (fields=2)\n  (x): 1.0\n  (y): 2.0"
    )


@pydantic_available
def test_pydantic_model_summarizer_summarize_hides_the_secret_fields(
    registry: SummarizerRegistry,
) -> None:
    user = User(name="a", password="secret", token="tok")  # noqa: S106
    assert PydanticModelSummarizer().summarize(user, registry) == f"{User} (fields=1)\n  (name): a"


@pydantic_available
def test_pydantic_model_summarizer_summarize_extra_values(registry: SummarizerRegistry) -> None:
    assert PydanticModelSummarizer().summarize(Open(x=1, e=2), registry) == (
        f"{Open} (fields=2)\n  (x): 1\n  (e): 2"
    )


@pydantic_available
def test_pydantic_model_summarizer_summarize_depth_limit(registry: SummarizerRegistry) -> None:
    assert PydanticModelSummarizer().summarize(Point(x=1.0, y=2.0), registry, max_depth=0) == (
        "x=1.0 y=2.0"
    )


@pydantic_available
def test_pydantic_model_summarizer_summarize_nested(registry: SummarizerRegistry) -> None:
    assert PydanticModelSummarizer().summarize(
        Outer(point=Point(x=1.0, y=2.0)), registry, max_depth=2
    ) == (f"{Outer} (fields=1)\n  (point): {Point} (fields=2)\n      (x): 1.0\n      (y): 2.0")


@pydantic_available
def test_summarize_pydantic_model_uses_the_default_summarizer() -> None:
    assert summarize(Point(x=1.0, y=2.0)) == f"{Point} (fields=2)\n  (x): 1.0\n  (y): 2.0"


@pydantic_available
def test_get_default_summarizers_keys() -> None:
    assert list(get_default_summarizers()) == [BaseModel]


def test_get_default_summarizers_without_pydantic() -> None:
    with patch("coola.summary.pydantic.is_pydantic_available", lambda: False):
        assert get_default_summarizers() == {}
