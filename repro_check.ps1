# PowerShell repro check (Windows equivalent of repro_check.sh)
# Run the pipeline 3 times and diff the results.

$env:PYTHONIOENCODING = "utf-8"
$env:LLM_PROVIDER = "none"

Write-Host "Running pipeline 3 times for reproducibility check..."

python src/run.py --no-llm
Copy-Item output.csv eval/repro_run1.csv

python src/run.py --no-llm
Copy-Item output.csv eval/repro_run2.csv

python src/run.py --no-llm
Copy-Item output.csv eval/repro_run3.csv

Write-Host "Comparing run outputs..."

$diff12 = Compare-Object (Get-Content eval/repro_run1.csv) (Get-Content eval/repro_run2.csv)
$diff13 = Compare-Object (Get-Content eval/repro_run1.csv) (Get-Content eval/repro_run3.csv)

if ($diff12 -or $diff13) {
    "DIFFERENCES FOUND" | Out-File eval/repro_diff.txt
    $diff12 | Out-File -Append eval/repro_diff.txt
    $diff13 | Out-File -Append eval/repro_diff.txt
    Write-Host "DIFFERENCES FOUND — see eval/repro_diff.txt"
} else {
    "No differences found. All 3 runs produced identical output." | Out-File eval/repro_diff.txt
    Write-Host "All 3 runs produced identical output. Pipeline is reproducible."
}
