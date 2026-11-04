#!/usr/bin/env bash
# Test that `update-subtree` syncs the subtree and removes unnecessary files.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git_cfg=(-c user.name=test -c user.email=test@example.com)

# Upstream repo containing files that must be removed and files that must be kept.
upstream="$work/upstream"
mkdir -p "$upstream/.github" "$upstream/testdata" "$upstream/tests"
cp "$repo_root/self.mk" "$upstream/self.mk"
for f in .gitignore CODE_OF_CONDUCT.md CONTRIBUTING.md README.md SECURITY.md LICENSE .github/x.yml testdata/x.txt tests/x.sh; do
	echo "# $f" >"$upstream/$f"
done
git -C "$upstream" init -q -b main
git -C "$upstream" add -A
git -C "$upstream" "${git_cfg[@]}" commit -q -m init

# Consuming project.
project="$work/project"
mkdir -p "$project"
git -C "$project" init -q -b main
git -C "$project" "${git_cfg[@]}" commit -q --allow-empty -m init
cat >"$project/Makefile" <<MK
SHARED_MAKEFILES_REMOTE_URL := $upstream
include $upstream/self.mk
MK

cd "$project"
export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com
make update-subtree

for f in .gitignore CODE_OF_CONDUCT.md CONTRIBUTING.md README.md SECURITY.md .github testdata tests; do
	if [ -e ".make/$f" ]; then
		echo "FAIL: .make/$f should have been removed" >&2
		exit 1
	fi
done
for f in LICENSE self.mk; do
	if [ ! -e ".make/$f" ]; then
		echo "FAIL: .make/$f should have been kept" >&2
		exit 1
	fi
done
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
	echo "FAIL: working tree not clean after sync" >&2
	exit 1
fi
echo "PASS"
