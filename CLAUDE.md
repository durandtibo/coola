# Instructions for coding agents

## Environment

- Use the virtual environment in `.venv`: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`.
  There is no `python` on the `PATH`.
- Lint and format before committing: `.venv/bin/ruff check src tests` and `.venv/bin/ruff format src tests`.
  Fix the issues raised by ruff. Ignoring an issue (`noqa`, a changed rule set) is not fixing it.
- Type check with `.venv/bin/ty check`. Fix the issues raised by ty. Ignoring an issue
  (`type: ignore`, `ty: ignore`, a cast that hides the problem) is not fixing it.
- Run the unit tests of the packages you changed (for example `.venv/bin/pytest tests/unit/hashing -q`).
  Some tests fail on `main`; check with `git stash` before blaming your change.

## Workflow

- One branch and one pull request per item (for example per item of `review/suggestions.md`).
  Branch from an up to date `main`, name the branch `fix/...` or `docs/...`.
- Do not commit or push to `main`.
- End commit messages with the `Co-Authored-By` line and PR descriptions with the "Generated with
  Claude Code" line given by the harness.
- In the PR description, say what was changed, what was deliberately not changed, and any
  behaviour change (for example hashes that change, or a breaking change of the signature).
- When a PR conflicts with `main`, merge `origin/main` into the branch and resolve the conflicts
  (do not rebase and force-push). Most conflicts are two PRs appending tests at the end of the
  same test file: keep both. Check what `main` already contains before choosing a side.
- Update the status table of `review/suggestions.md` when items are fixed, with the PR number.

## Code

- Match the surrounding code: comment density, naming, docstring style (Google style with
  `Args:`, `Returns:`, `Raises:` and a `pycon` example that runs as a doctest).
- Follow the current style of the repository in everything you add (source, tests, docs, commit
  messages, PR descriptions). Look at a few neighbouring files first, and do not introduce a new
  convention or reformat code you do not otherwise change.
- Shared helpers belong in `coola.utils` as public functions (no leading underscore) with their
  own tests, not as private functions imported across modules.
- Validate an argument with `coola.validation`; do not write `raise ValueError` for a type, range
  or membership check, and do not add a validator that is one `isinstance` or one comparison.
- Prefer `LazySingleton` for lazily built process-wide objects, not `functools.cache`.
- Keep new options keyword-only (`*`), as `hash_object` does.
- Optional libraries (numpy, torch, pandas, polars, ...) are imported lazily; do not import them
  at module level in `src`.
- Every function, public or private (leading underscore), needs a docstring and its own unit
  tests. Private functions are not exempt.
- Keep docstrings and documentation up to date and comprehensive: update them whenever the
  behaviour, signature or options of the code change.

## Tests

- Always write tests for the code in `src/`, including every private function. Unit tests must be fast and efficient; move big or
  slow tests to `tests/integration/` if needed.
- Unit tests mirror the source tree (`tests/unit/<package>/test_<module>.py`). Cross-thread
  behaviour goes in `tests/integration/`, using `run_threads` from `tests/integration/helpers.py`,
  a lock around the shared list, and a docstring starting with "Test that ...".
- Do not use local imports inside tests (for example `import polars as pl` in a test body). Import
  at the top of the module; use the `*_available` fixtures from `coola.testing.marks` for
  optional libraries.
- One behaviour per test: a test has one `assert` or one `pytest.raises`. Split a test that checks
  several things (equal, different, negative, ...) into several tests. Thread-safety tests may
  assert the count and the uniqueness together, as the existing ones do.
- Keep the unit tests simple and fast: no sleeps, no large data, no real network or heavy I/O,
  and no logic (loops, branches) in the test body. Anything slow or concurrent belongs in
  `tests/integration/`.
- Add the missing tests to get 100% coverage of the code you add or change
  (`.venv/bin/pytest <tests> --cov=coola --cov-report=term-missing`). Do not use
  `# pragma: no cover` to skip a line that can be tested.
- Use `pytest.mark.parametrize` for the same check on several values.
- Every `pytest.raises` (and `pytest.warns`) must define `match` with a regular expression for the
  message (raw string if it has metacharacters), so a test does not pass on an unrelated error of
  the same type. An exception raised by a patched stand-in needs an explicit message for the same
  reason (`side_effect=KeyboardInterrupt("interrupted")`, not `side_effect=KeyboardInterrupt`).
- Add a regression test for every bug fixed, and check that it fails without the fix (revert the
  `src` change with `git stash push src` and run it).
- Do not add tests for the exports of `__all__` or other trivial structure.
- Section headers in the tests have three lines. The second line is `#`, 5 spaces, the name,
  5 spaces and `#`. The first and third lines repeat `#` to match the length of the second line:

  ```python
  ###########################
  #     Tests for Fixed     #
  ###########################
  ```

- Seed random data (`default_rng(0)`); no unseeded randomness in tests.
- Patching (`monkeypatch`, `unittest.mock.patch`) is fine in unit tests: it keeps them fast and
  simple and isolates the logic under test. But a patched test only proves the logic works against
  the patched stand-in, which may not match the real collaborator (signature, return type,
  behaviour). A passing patched unit test does not mean the feature works. Every patched behaviour
  needs an integration test in `tests/integration/` that exercises the real thing without patching.
  The two kinds of test have different purposes; do not drop one because the other exists.

### Test assertions

- When asserting an expected value, write the final computed number, not the arithmetic
  expression that produces it (for example `0.6666666666666666`, not `1.0 * 0.4 + 0.0 * 0.2`).
  If the derivation is not obvious, keep the expression as a comment above the assertion.
- When a function returns a composite value (a dict, a list, ...), include at least one test that
  asserts the entire output against a fully-specified literal, not only individual keys or fields.
  This catches unexpected extra or missing entries that per-field assertions miss:

  ```python
  result = compute(...)
  assert result == {...}
  ```

### Test coverage

When implementing or modifying a function, write comprehensive unit tests, not just a happy-path
check. Cover, where applicable:

- Typical inputs.
- Edge cases: empty inputs, single-element inputs, duplicate inputs.
- Boundary values (0, negative numbers, min/max of a valid range).
- Error conditions: inputs that should raise, with the right exception type and message
  (including errors raised by collaborators the function calls).
- Interactions between parameters, for example a cross-product of two input lists, or the order
  of output keys or entries.
- Side effects: the function does not mutate its inputs, is deterministic across repeated calls,
  and does not leak state when given reused or stateful collaborators.

Prefer many small, precisely named tests (one behavior per test) over a few broad ones, so a
failure points at the specific behavior that broke.

## Reporting

- Report faithfully: say which tests were run, which were not, and that CI was not awaited.
- Mention test failures that already exist on `main` instead of hiding them.
