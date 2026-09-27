#!/usr/bin/env bash
# Usage: scripts/validate.sh [output_dir] [--check-ids]   (default: outputs/)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs"
if [ -n "$1" ] && [ "${1#--}" = "$1" ]; then OUT="$1"; shift; fi
uv run python "$ROOT/data/raw/student_resource/utils/validate_submission.py" \
  --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir "$ROOT/data/raw/student_resource/dataset/test" "$@"
