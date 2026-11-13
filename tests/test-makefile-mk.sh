#!/usr/bin/env bash
# Test that `lint-makefile` skips unmatched globs instead of passing them to checkmake.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Fake checkmake: records its arguments and fails if any is not an existing file.
mkdir -p "$work/bin"
cat >"$work/bin/checkmake" <<'SH'
#!/usr/bin/env bash
echo "$*" >"$CHECKMAKE_ARGS_FILE"
for f in "$@"; do
	[ -f "$f" ] || { echo "Error opening '$f'" >&2; exit 1; }
done
SH
chmod +x "$work/bin/checkmake"
export PATH="$work/bin:$PATH"
export CHECKMAKE_ARGS_FILE="$work/args"

project="$work/project"
mkdir -p "$project"
cd "$project"
printf '.PHONY: all\nall:\n\t@true\n' >Makefile

fail() {
	echo "FAIL: $1" >&2
	exit 1
}

# 1. Default pattern with no *.mk file: only Makefile is linted.
rm -f "$CHECKMAKE_ARGS_FILE"
make -f "$repo_root/makefile.mk" lint-makefile >/dev/null || fail "default patterns without *.mk should pass"
[ "$(cat "$CHECKMAKE_ARGS_FILE")" = "Makefile" ] || fail "expected only Makefile, got: $(cat "$CHECKMAKE_ARGS_FILE")"

# 2. Default pattern with a *.mk file: both files are linted.
touch extra.mk
make -f "$repo_root/makefile.mk" lint-makefile >/dev/null || fail "default patterns with *.mk should pass"
[ "$(cat "$CHECKMAKE_ARGS_FILE")" = "Makefile extra.mk" ] || fail "expected Makefile and extra.mk, got: $(cat "$CHECKMAKE_ARGS_FILE")"

# 3. Nothing matches: checkmake is not run and the target succeeds.
rm -f "$CHECKMAKE_ARGS_FILE"
out="$(make -f "$repo_root/makefile.mk" lint-makefile MAKEFILE_LINT_FILES="no-such-dir/*.mk")" || fail "no matching files should not fail"
[ ! -e "$CHECKMAKE_ARGS_FILE" ] || fail "checkmake should not run when nothing matches"
grep -q "No files found matching" <<<"$out" || fail "expected 'No files found' message"

echo "PASS: test-makefile-mk"
