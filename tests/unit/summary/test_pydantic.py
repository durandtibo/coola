from __future__ import annotations

from typing import Optional
from unittest.mock import patch

from coola.summary import SummaryConfig, summarize
from coola.summary.pydantic import PydanticModelSummarizer, get_default_summarizers
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

CONFIG = SummaryConfig()

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, SecretStr

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

    class Empty(BaseModel):
        pass


#############################################
#     Tests for PydanticModelSummarizer     #
#############################################


@pydantic_available
def test_pydantic_model_summarizer_summarize() -> None:
    node = PydanticModelSummarizer().summarize(Point(x=1.0, y=2.0), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        f"{Point} (fields=2)",
        [("x", 1.0), ("y", 2.0)],
        2,
    )


@pydantic_available
def test_pydantic_model_summarizer_summarize_empty() -> None:
    node = PydanticModelSummarizer().summarize(Empty(), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (f"{Empty} (fields=0)", [], 0)


@pydantic_available
def test_pydantic_model_summarizer_summarize_hides_the_secret_fields() -> None:
    user = User.model_validate({"name": "a", "password": "secret", "token": "tok"})
    node = PydanticModelSummarizer().summarize(user, CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        f"{User} (fields=1)",
        [("name", "a")],
        1,
    )


@pydantic_available
def test_pydantic_model_summarizer_summarize_extra_values() -> None:
    node = PydanticModelSummarizer().summarize(Open(x=1, e=2), CONFIG, compact=False)
    assert (list(node.children), node.total) == ([("x", 1), ("e", 2)], 2)


@pydantic_available
def test_pydantic_model_summarizer_summarize_nested_model_is_not_expanded() -> None:
    node = PydanticModelSummarizer().summarize(
        Outer(point=Point(x=1.0, y=2.0)), CONFIG, compact=False
    )
    assert list(node.children) == [("point", Point(x=1.0, y=2.0))]


@pydantic_available
def test_summarize_pydantic_model_uses_the_default_registry() -> None:
    assert summarize(Point(x=1.0, y=2.0)) == f"{Point} (fields=2)\n  (x): 1.0\n  (y): 2.0"


@pydantic_available
def test_summarize_nested_pydantic_model() -> None:
    assert summarize(Outer(point=Point(x=1.0, y=2.0)), max_depth=2) == (
        f"{Outer} (fields=1)\n  (point): {Point} (fields=2)\n    (x): 1.0\n    (y): 2.0"
    )


@pydantic_available
def test_summarize_pydantic_model_at_the_display_depth() -> None:
    assert summarize(Outer(point=Point(x=1.0, y=2.0))) == (
        f"{Outer} (fields=1)\n  (point): Point(x=1.0, y=2.0)"
    )


@pydantic_available
def test_get_default_summarizers_keys() -> None:
    assert list(get_default_summarizers()) == [BaseModel]


def test_get_default_summarizers_without_pydantic() -> None:
    with patch("coola.summary.pydantic.is_pydantic_available", lambda: False):
        assert get_default_summarizers() == {}
