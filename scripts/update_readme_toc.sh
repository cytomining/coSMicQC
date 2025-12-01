#!/usr/bin/env bash
set -euo pipefail

# Use ekalinin/github-markdown-toc to refresh the README TOC.
# This script fetches the upstream helper on demand (requires network).

TARGET_FILE="${1:-README.md}"

tmp_script="$(mktemp)"
trap 'rm -f "$tmp_script"' EXIT

curl -fsSL https://raw.githubusercontent.com/ekalinin/github-markdown-toc/master/gh-md-toc > "$tmp_script"
chmod +x "$tmp_script"

"$tmp_script" --no-backup "$TARGET_FILE"
