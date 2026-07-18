#!/usr/bin/env bash
# Pre-run the benchmark ONCE (needs an API key — vision models cost money),
# then the Model & Benchmark page shows the multi-model comparison read-only.
#
# Usage:  ./scripts/run_benchmark.sh [n_per_type] [budget_usd]
#   e.g.  ./scripts/run_benchmark.sh 20 5
set -e
cd "$(dirname "$0")/.."

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "✗ Set ANTHROPIC_API_KEY first — the vision models (Haiku/Sonnet/Opus) need it."
  echo "  export ANTHROPIC_API_KEY=sk-ant-...   then re-run this script."
  exit 1
fi

N="${1:-20}"
BUDGET="${2:-5}"
echo "Running benchmark: n=$N per type, budget \$$BUDGET, models = M1 + Haiku + Sonnet + Opus (vision)"

./.venv/bin/python -m benchmark.run \
  --n "$N" --max-edge 1300 --max-usd "$BUDGET" --workers 8 --yes \
  --methods M1_ocr_rules,M5_vision_haiku,M3_vision_sonnet,M4_vision_opus

./.venv/bin/python report/generate.py
echo "✓ Done. Open http://localhost:8000/report (Model & Benchmark) to see the comparison."
