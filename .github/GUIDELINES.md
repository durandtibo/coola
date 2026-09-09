# GitHub Actions workflows

This directory holds every workflow for the repository. There are no
subdirectories: GitHub only discovers workflows placed directly under
`workflows`, so reusable workflows live here too, alongside the
top-level ones that trigger on events.

## Naming convention

Every file is prefixed by what it does:

| Prefix       | Purpose                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------- |
| `ci-`        | Quality/test checks, most of them reusable workflows called from `workflows/ci.yaml`.                  |
| `lib-`       | Reusable workflows with no event trigger of their own; they only expose `workflow_call` outputs (e.g. reading a JSON config file) and are `needs:`-ed by other jobs. |
| `bot-`       | Scheduled automation that pushes commits / opens PRs as the `ci-bot` GitHub App.              |
| `nightly-`   | Scheduled checks against the *published* PyPI package (as opposed to `ci-*`, which checks the repo's source). |
| `release-`   | Publishing: PyPI package, GitHub release assets, documentation.                              |
| `security-`  | Supply-chain/security scanning (Scorecard, dependency review).                               |

`workflows/ci.yaml` is the entry point for pull requests/pushes to `main`; it fans out
to the `ci-*.yaml` reusable workflows so each check can also be run/dispatched
on its own. Job ids in `ci.yaml` and in the workflows it calls are
load-bearing for branch protection (see the comment at the top of `ci.yaml`)
— keep them in sync if you rename either side.

## Composite actions vs. reusable workflows

- **`.github/actions/*`** (composite actions): used when the shared unit is a
  handful of *steps* inside a single job (e.g. checkout + `uv` setup,
  generating a bot token + checking out, or opening the resulting PR as
  ci-bot). Composite actions cannot produce a job-level output usable in a
  `strategy.matrix`.
- **`.github/workflows/lib-*.yaml`** (reusable workflows called with `uses:
  ./.github/workflows/lib-....yaml`): used when the output needs to feed a
  matrix, or when the shared logic is naturally a whole job (e.g. reading
  `../dev/config/test_matrix.json` once and handing the JSON down to several
  jobs via `needs:`).

When adding new duplication, prefer extending an existing composite
action/reusable workflow over copy-pasting steps.

## Conventions applied to every workflow

- **Action pinning**: every third-party (and first-party `actions/*`) step is
  pinned to a full commit SHA with a trailing `# ratchet:owner/repo@vX.Y.Z`
  comment, added and refreshed automatically by
  [`ratchet`](https://github.com/sethvargo/ratchet) via `workflows/bot-pin-action.yaml`.
  Never hand-pin a SHA without that comment — the next `bot-pin-action` run
  would otherwise silently downgrade or drift it. Local composite actions
  (`uses: ./.github/actions/...`) are referenced by path, not pinned.
- **Permissions**: top-level `permissions:` is always the least the workflow
  needs — `contents: read` or `{}` — and any job that needs more (e.g.
  `contents: write` to push, `id-token: write` for OIDC) declares it on that
  job only, never widening the workflow default.
- **Timeouts**: every job sets `timeout-minutes`, sized to what the job
  actually does (2 for a config-read job, 5 for most checks, 10 for slower
  jobs like Scorecard or benchmarks) so a hung step can't occupy a runner
  indefinitely.
- **Runners**: `ubuntu-slim` for lightweight jobs that only run a small
  action or a couple of shell commands (no Python/build tooling); otherwise
  `ubuntu-latest`, or the OS matrix under test for `workflows/ci-test.yaml` /
  `nightly-test-package.yaml`.
- **`workflow_dispatch`**: added to every workflow (checks and automation
  alike) so it can be re-run manually without waiting for its normal trigger.

## Shared configuration

Values that would otherwise be duplicated across workflows are centralized in
`../dev/config`:

- `../dev/config/test_matrix.json` — supported Python versions / OS matrix,
  read once per run by `lib-get-test-matrix.yaml`.
- `../dev/config/package_versions.json` — version ranges to test optional
  dependencies against, read by `lib-get-package-versions.yaml` and kept
  current by `bot-generate-package-versions.yaml`.

Optional-dependency *names* are never duplicated in config: they're read
directly from `../pyproject.toml`'s `[project.optional-dependencies]` by the
`get-package-extras` composite action / `lib-get-package-extras.yaml`, so a
new extra only needs to be added in one place.

## Validating changes to this directory

`workflows/ci-verify-workflows.yaml` runs `durandtibo/verify-github-workflow-action` on
every PR/push that touches `.github/workflows/*`, checking pinning and
general workflow correctness. Locally, `actionlint` (run from this
directory) and a YAML syntax check are good pre-flight checks before pushing:

```bash
cd .github/workflows && actionlint
python3 -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('*.yaml')]"
```
