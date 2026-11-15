#!/usr/bin/env bash
# Test that `install-zizmor` installs zizmor only when it is missing (fake installers, no network).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() {
	echo "FAIL: $1" >&2
	exit 1
}

# Minimal PATH: only make, uname, chmod, grep and head are real, so a zizmor/brew/uv installed on the machine is hidden.
mkdir -p "$work/base" "$work/fake"
for cmd in make uname chmod grep head; do
	ln -s "$(command -v "$cmd")" "$work/base/$cmd"
done
run_make() {
	PATH="$work/fake:$work/base" make -f "$repo_root/actions.mk" install-zizmor 2>&1
}

# Fake installer that records its call and puts a fake zizmor on the PATH.
installer="uv"
[ "$(uname -s)" = "Darwin" ] && installer="brew"
cat >"$work/fake/$installer" <<SH
#!/bin/sh
echo "\$*" >>"$work/calls"
printf '#!/bin/sh\n' >"$work/fake/zizmor"
chmod +x "$work/fake/zizmor"
SH
chmod +x "$work/fake/$installer"

# 1. zizmor missing: it is installed with the platform installer.
out="$(run_make)" || fail "install should succeed when zizmor is missing"
grep -q "zizmor not found" <<<"$out" || fail "should report that zizmor is missing"
grep -q "zizmor" "$work/calls" || fail "$installer should be called with zizmor"
[ -x "$work/fake/zizmor" ] || fail "zizmor should be installed"

# 2. zizmor present: nothing is installed.
rm -f "$work/calls"
out="$(run_make)" || fail "install should succeed when zizmor is present"
[ ! -e "$work/calls" ] || fail "installer must not be called when zizmor is present"
! grep -q "not found" <<<"$out" || fail "should be silent when zizmor is present"

# 3. Linux without any installer: fails with a hint.
if [ "$(uname -s)" != "Darwin" ]; then
	rm -f "$work/fake/zizmor" "$work/fake/uv"
	rc=0
	out="$(run_make)" || rc=$?
	[ "$rc" -ne 0 ] || fail "should fail without any installer"
	grep -q "Please install zizmor" <<<"$out" || fail "should explain how to install zizmor"
fi
echo "PASS"
