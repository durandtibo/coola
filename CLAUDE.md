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
- Shared helpers belong in `coola.utils` as public functions (no leading underscore) with their
  own tests, not as private functions imported across modules.
- Prefer `LazySingleton` for lazily built process-wide objects, not `functools.cache`.
- Keep new options keyword-only (`*`), as `hash_object` does.
- Optional libraries (numpy, torch, pandas, polars, ...) are imported lazily; do not import them
  at module level in `src`.

## Tests

- Unit tests mirror the source tree (`tests/unit/<package>/test_<module>.py`). Cross-thread
  behaviour goes in `tests/integration/`, using `run_threads` from `tests/integration/helpers.py`,
  a lock around the shared list, and a docstring starting with "Test that ...".
- Do not use local imports inside tests (for example `import polars as pl` in a test body). Import
  at the top of the module; use the `*_available` fixtures from `coola.testing.fixtures` for
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

## Reporting

- Report faithfully: say which tests were run, which were not, and that CI was not awaited.
- Mention test failures that already exist on `main` instead of hiding them.
