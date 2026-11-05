from __future__ import annotations

from coola.iterator import dfs_iterate
from coola.nested import to_flat_dict
from coola.recursive import recursive_apply
from coola.rounding import round_values
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel

    class Model(BaseModel):
        a: int = 1
        b: list[float] = [1.234, 2.345]


@pydantic_available
def test_recursive_apply_enters_a_model_with_the_default_registry() -> None:
    assert recursive_apply(Model(), lambda v: v + 1 if isinstance(v, int) else v) == Model(
        a=2, b=[1.234, 2.345]
    )


@pydantic_available
def test_dfs_iterate_enters_a_model_with_the_default_registry() -> None:
    assert list(dfs_iterate(Model())) == [1, 1.234, 2.345]


@pydantic_available
def test_round_values_recursive_rounds_the_fields_of_a_model() -> None:
    assert round_values(Model(), decimals=1, recursive=True) == Model(a=1, b=[1.2, 2.3])


@pydantic_available
def test_to_flat_dict_flattens_a_model() -> None:
    assert to_flat_dict(Model()) == {"a": 1, "b.0": 1.234, "b.1": 2.345}
