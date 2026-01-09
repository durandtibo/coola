from __future__ import annotations

import random

from coola.random.interface import random_seed

#################################
#     Tests for random_seed     #
#################################


def test_random_seed_nested_contexts() -> None:
    """Test that nested random_seed contexts work correctly."""
    with random_seed(42):
        value1 = random.uniform(0, 1)  # noqa: S311

        with random_seed(123):
            value2 = random.uniform(0, 1)  # noqa: S311

        # After inner context exits, outer seed should still be active
        value3 = random.uniform(0, 1)  # noqa: S311

    # Verify inner context had different seed
    assert value1 != value2

    # Verify we can reproduce the sequence
    with random_seed(42):
        check1 = random.uniform(0, 1)  # noqa: S311
        with random_seed(123):
            check2 = random.uniform(0, 1)  # noqa: S311
        check3 = random.uniform(0, 1)  # noqa: S311

    assert value1 == check1
    assert value2 == check2
    assert value3 == check3
