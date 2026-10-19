#!/usr/bin/env bash
set -euo pipefail

# Run from the Git repository root, regardless of caller location.
cd "$(dirname "$0")/../.."

echo "=== Python syntax validation ==="
python3 -m compileall -q research/attention_rc1

echo "=== MoA attention numerical validation ==="
python3 research/attention_rc1/moa_codegen_v1.py
