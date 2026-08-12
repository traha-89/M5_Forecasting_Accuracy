#!/usr/bin/env bash
# SessionStart hook: prints where the M5 project stands and what's next,
# derived from docs/plan/*.md gate checkboxes rather than hardcoded.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
plan_dir="$repo_root/docs/plan"

last_done_phase=""
next_phase=""
next_phase_file=""

for f in "$plan_dir"/P[0-9]-*.md; do
  [ -e "$f" ] || continue
  name="$(basename "$f" .md)"
  total=$(grep -c '^\- \[[ x]\]' "$f" 2>/dev/null || true)
  total=${total:-0}
  if [ "$total" -eq 0 ]; then
    continue
  fi
  checked=$(grep -c '^\- \[x\]' "$f" 2>/dev/null || true)
  checked=${checked:-0}
  if [ "$checked" -eq "$total" ]; then
    last_done_phase="$name"
  elif [ -z "$next_phase" ]; then
    next_phase="$name ($checked/$total gate items checked)"
    next_phase_file="$f"
  fi
done

branch="$(git -C "$repo_root" branch --show-current 2>/dev/null || echo "unknown")"
last_commit="$(git -C "$repo_root" log -1 --format='%h %s' 2>/dev/null || echo "none")"
dirty="$(git -C "$repo_root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

echo "M5 Forecasting Accuracy - project status"
echo "  Branch: $branch (last commit: $last_commit)"
if [ "$dirty" -gt 0 ]; then
  echo "  Uncommitted/untracked changes: $dirty file(s)"
fi
if [ -n "$last_done_phase" ]; then
  echo "  Last completed gate: $last_done_phase"
fi
if [ -n "$next_phase" ]; then
  echo "  Next up: $next_phase"
  echo "  Brief: docs/plan/$(basename "$next_phase_file")"
else
  echo "  All phase gates in docs/plan checked off."
fi
