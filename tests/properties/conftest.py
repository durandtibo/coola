from __future__ import annotations

import os

import hypothesis

# Hashing a large nested example can be slow on a busy CI machine, and a
# deadline only makes the tests flaky.
hypothesis.settings.register_profile("coola", deadline=None)
# On the CI, the examples are the same for every run (a failure can be reproduced) and the
# failing example is printed as a blob that can be replayed with ``@reproduce_failure``.
hypothesis.settings.register_profile(
    "ci", parent=hypothesis.settings.get_profile("coola"), derandomize=True, print_blob=True
)
hypothesis.settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "coola"))
