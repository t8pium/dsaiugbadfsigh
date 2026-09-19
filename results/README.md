# Local generated results

Canonical experiment output is written here and ignored by Git. `_runs/` records success/failure, source hash, input mtime, arguments, and files changed by each run; `_logs/` retains child-process output.

The dashboard treats an output as a **LOCAL REPRODUCTION** only when the latest successful manifest links it to the current processed dataset. Frozen published values always come from `reference_results/reference_metrics.json`.
