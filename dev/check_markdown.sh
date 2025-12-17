#!/usr/bin/env bash

# check_markdown.sh - Validate code examples in markdown files
#
# Description:
#   Finds and tests all markdown files in the repository using Python's doctest
#   module. This ensures that code examples in documentation are correct and
#   up-to-date. Excludes virtual environments and cache directories.
#
# Usage:
#   dev/check_markdown.sh
#
# Requirements:
#   - Python must be available
#   - Markdown files should contain testable Python code blocks
#
# Exit Codes:
#   0 - All markdown files passed doctest validation
#   1 - One or more markdown files failed doctest validation

set -euo pipefail

# List of folders to exclude
exclude_folders=(.venv .pytest_cache)

# Build the find command with exclusions
find_cmd="find ."
for folder in "${exclude_folders[@]}"; do
	find_cmd+=" -path \"./$folder\" -prune -o"
done
find_cmd+=" -type f -name \"*.md\" -print"

# Execute the find command and capture files
files=$(eval "$find_cmd")
count=$(printf "%s\n" "$files" | awk 'END {print NR}')

echo "Found $count markdown files"

printf "%s\n" "$files" | while IFS= read -r f; do
	echo "Checking: $f"
	python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS -o REPORT_NDIFF "$f"
done

echo "All $count markdown files have been checked"
