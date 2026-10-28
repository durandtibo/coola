from __future__ import annotations

from tests.integration.helpers import run_python

# A fresh interpreter: registering a hasher changes the default registry of the process.
CODE = """
from coola.hashing import StringHasher, register_hashers
from coola.identifier import GENERATE

class MyInt(int):
    pass

data = {"a": MyInt(1), "b": [MyInt(2)]}
before = GENERATE(data)
register_hashers({MyInt: StringHasher()})
print(before == GENERATE(data))
"""

############################################
#     Tests for the stable identifiers     #
############################################


def test_generate_stable_content_id_ignores_hashers_registered_in_default_registry() -> None:
    """Test that a hasher registered in the default registry does not
    change the stable content identifier."""
    assert run_python(CODE.replace("GENERATE", "generate_stable_content_id")) == ["True"]


def test_generate_stable_uuid5_ignores_hashers_registered_in_default_registry() -> None:
    """Test that a hasher registered in the default registry does not
    change the stable UUID."""
    assert run_python(CODE.replace("GENERATE", "generate_stable_uuid5")) == ["True"]
