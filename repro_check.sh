#!/usr/bin/env bash
# repro_check.sh — Run the pipeline 3 times and diff the results.
# Results are saved to eval/repro_diff.txt.

set -e

echo "Running pipeline 3 times for reproducibility check..."

python src/run.py --no-llm
cp output.csv eval/repro_run1.csv

python src/run.py --no-llm
cp output.csv eval/repro_run2.csv

python src/run.py --no-llm
cp output.csv eval/repro_run3.csv

echo "Diffing run1 vs run2..."
diff eval/repro_run1.csv eval/repro_run2.csv > eval/repro_diff.txt 2>&1 || true

echo "Diffing run1 vs run3..."
diff eval/repro_run1.csv eval/repro_run3.csv >> eval/repro_diff.txt 2>&1 || true

echo ""
if [ -s eval/repro_diff.txt ]; then
    echo "DIFFERENCES FOUND:"
    cat eval/repro_diff.txt
else
    echo "All 3 runs produced identical output. Pipeline is reproducible."
    echo "No differences found." > eval/repro_diff.txt
fi
