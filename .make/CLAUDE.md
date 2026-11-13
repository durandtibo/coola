# CLAUDE.md

Guidelines for coding agents working in this repository.

## Project

Shared, reusable Makefiles for formatting and linting common file types (YAML, Markdown, shell,
TOML, Makefiles, GitHub Actions). Consumers vendor this repo as a git subtree (conventionally at
`.make/`) and `include` the `.mk` files they need.

- `*.mk` — one self-contained module per file type or concern (`yaml.mk`, `shell.mk`, `toml.mk`,
  `markdown.mk`, `makefile.mk`, `actions.mk`, `uv.mk`), plus helpers (`prettier.mk`, `help.mk`,
  `self.mk`).
- `Makefile` — the repo's own entry point; includes every module and defines `install-tools`,
  `format` and `lint`.
- `testdata/` — sample files used by CI to exercise each module.
- `tests/` — shell tests (e.g. `tests/test-self-mk.sh` for `update-subtree`).
- `.github/workflows/` — CI (`ci.yaml` orchestrates the reusable `ci-<module>.yaml` workflows,
  which run each module against `testdata/`; `lib-get-test-matrix.yaml` loads the OS/Python
  matrices).

## Environment

- Tools (`prettier`, `yamllint`, `mbake`, `checkmake`, `shellcheck`, `shfmt`, `markdownlint`,
  `actionlint`, ...) are installed on demand by `install-*` targets; run `make install-tools` to
  install them all.
- Run `make help` to list documented targets.
- Before committing, run `make format` and `make lint`.
- Run `bash tests/test-self-mk.sh` when touching `self.mk`.

## Workflow

- One branch and one PR per logical change, branched from an up-to-date `main`.
- Name branches `<type>/<short-description>` in kebab-case (feat, bugfix, docs, refactor, test,
  chore, ci).
- Never commit or push directly to `main`.
- Resolve conflicts by merging `origin/main` into the branch, not by rebasing and force-pushing.
- PR descriptions state what changed, what was deliberately left alone, and any behaviour change
  (renamed targets, changed variable defaults) since consumers depend on them.
- Follow `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`.

## Makefile conventions

- Every `.mk` file is include-guarded:
  `ifndef FOO_MK_INCLUDED` / `FOO_MK_INCLUDED := 1` ... `endif`.
- Document optional variables in a header comment and define them with `?=` so consumers can
  override them before including.
- Include sibling modules via `$(dir $(lastword $(MAKEFILE_LIST)))other.mk` so the file works from
  any vendored path.
- Mark every target `.PHONY`.
- Public targets carry a `## description` comment so they appear in `make help`. Internal
  `install-*` targets do not.
- Each `format-*`/`lint-*` target depends on an `install-*` target that installs the tool only if
  it is missing from `PATH` (Homebrew on Darwin, a sensible fallback elsewhere).
- Keep modules independent: a consumer must be able to include only the files they need.
- Match the existing output style (emoji-prefixed `@echo` start/finish messages).
- Recipes must work on both macOS and Linux; avoid GNU-only flags.
- Never hardcode paths to the consumer's layout; expose them as variables.
- When adding or renaming a module or target, update `README.md` and the root `Makefile`
  (`install-tools`, `format`, `lint`).

## Tests and CI

- Each module has a reusable workflow `.github/workflows/ci-<module>.yaml` (taking the `test-matrix`
  input of `lib-get-test-matrix.yaml`) that runs its targets against files in `testdata/`, called
  from a job in `ci.yaml`. Add a sample file, a workflow and a job for any new module.
- Shell tests live in `tests/`; they use temp dirs and must not touch the real repo or network.
- Pin GitHub Actions by commit SHA (see `bot-pin-action.yaml`) and keep workflow `permissions`
  minimal and explicit.

## Workflow structure (`.github/workflows/`)

Files are prefixed by role:

- `ci.yaml` — entry point for pull requests, pushes to `main`, a nightly schedule and manual runs.
  It holds no steps, only jobs that call the reusable `ci-*.yaml` workflows below.
- `ci-format.yaml` — runs `make format` and checks that nothing changed; not matrix-based, so
  `ci.yaml` calls it without `test-matrix`.
- `ci-<module>.yaml` (`ci-yaml`, `ci-makefile`, `ci-shell`, `ci-toml`, `ci-markdown`, `ci-actions`,
  `ci-uv`) — run the targets of `<module>.mk` against `testdata/` on the OS matrix (`ci-uv` also
  uses the Python matrix).
- `ci-self.yaml` — runs `tests/test-self-mk.sh` for `self.mk` on the OS matrix.
- `ci-tools.yaml` — runs `tests/test-tools-mk.sh` for `tools.mk` on the OS matrix.
- `ci-verify-workflows.yaml` — lints the workflow files. It has its own triggers and is not called
  from `ci.yaml`, so it can be a required check.
- `lib-get-test-matrix.yaml` — reusable workflow that loads the OS/Python matrices once from
  `durandtibo/workflow-config-action`; callers pass them down via the `test-matrix` input.
- `bot-*.yaml` — scheduled automation (`bot-format-files`, `bot-pin-action`,
  `bot-cancel-stale-queued-runs`); the first two open PRs using a GitHub App token.
- `security-*.yaml` — dependency review and OSSF Scorecard.

Conventions:

- Set `permissions: {}` at the top of every workflow and explicit `permissions` on every job,
  including jobs that call reusable workflows.
- Each matrix `ci-<module>` workflow is `workflow_call` only, takes the required `test-matrix`
  string input, and sets `timeout-minutes` and `fail-fast: false`.
- Call `make` targets in steps instead of duplicating commands, so local runs and CI match.
- A new module gets its own `ci-<module>.yaml` plus a job (with `needs: get-matrix`) in `ci.yaml`.
- Start each file with a comment describing what it does and update it when behaviour changes.
- Pin actions by commit SHA with a `# ratchet:` comment (see `bot-pin-action.yaml`).

## Reporting

- State which commands and tests were run and which were not, and that CI was not awaited.
- Report failures that already exist on `main` rather than hiding them.
