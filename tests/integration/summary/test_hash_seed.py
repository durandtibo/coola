from __future__ import annotations

from tests.integration.helpers import run_python

CODE = (
    "from coola.summary import summarize; "
    "print(repr(summarize({1, 'a', 'b', 'c', 'd', None}, max_items=3)).replace(' ', '_'))"
)


def test_summarize_set_of_not_comparable_items_does_not_depend_on_the_hash_seed() -> None:
    outputs = [run_python(CODE, env={"PYTHONHASHSEED": str(seed)}) for seed in (1, 2, 3)]
    assert outputs[0] == outputs[1] == outputs[2]
