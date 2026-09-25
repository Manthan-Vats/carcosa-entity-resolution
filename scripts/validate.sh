#!/usr/bin/env bash
# Usage: scripts/validate.sh [output_dir]   (default: outputs/)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/outputs}"
uv run python "$ROOT/data/raw/student_resource/utils/validate_submission.py" \
  --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir "$ROOT/data/raw/student_resource/dataset/test"
