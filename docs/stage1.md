# Stage 1 — heuristic baseline validation

Executed on 2026-11-17 using Python 3.9.6, NumPy 2.0.2 and NetworkX 3.2.1.

## Actual results

| Policy | Seed | Completed jobs | Mean JCT (s) | Makespan (s) |
|---|---:|---:|---:|---:|
| dynamic_partition | 1000 | 201 | 53.873900 | 5154.188 |
| dynamic_partition | 1001 | 201 | 68.833841 | 4933.751 |
| dynamic_partition | 1002 | 201 | 47.070871 | 5909.508 |
| spark_fifo | 1000 | 201 | 301.072637 | 5345.562 |
| spark_fifo | 1001 | 201 | 400.590090 | 5453.009 |
| spark_fifo | 1002 | 201 | 110.440761 | 5909.566 |

This table was generated from `results/stage1/run_a/summary.json` during the stage build.
Run B independently reproduced every job-level CSV byte-for-byte. See `results/stage1/repeatability.json`.

## Validation and scope

- 12 completed simulations: two policies × three seeds × two independent batches.
- All 201 jobs completed in every simulation.
- Metrics independently recalculated from saved job CSVs.
- Sum of JCT agrees with the simulator reward accounting in every run.
- Arrival profiles/timestamps identical across policies for each seed.
- Two metric unit tests passed (arrival offsets/unit conversion and invalid completion rejection).
- Upstream tracked source unchanged; runner records SHA-256 for source and profile files.
- No historical manuscript values used as expected outputs.
- No TensorFlow, learned-policy training, AWS infrastructure, or disruption experiments run.

## Limits and next-stage questions

These are development smoke tests, not held-out research evaluation. Exact agreement in
two batches on one interpreter is not a guarantee across Python versions or machines.
A fresh virtual-environment installation was not tested; commands used the available
interpreter with the recorded pinned package versions. The upstream clone/setup commands
are provided for users; this stage used the existing checkout, not a fresh network clone.

Policies share arrival workloads, but task-duration sampling is action-dependent in the
upstream simulator. Paired failures will need explicit treatment of random streams.
The potential `node` versus `next_node` issue identified in the audit has not been changed
or classified as a confirmed bug. It needs a targeted investigation before baseline changes.

Hardware inspection through sysctl was denied by the environment. The provenance files
record the available platform and logical CPU information; learned training cost still
requires a timing pilot. No clean-environment or cross-platform reproduction is claimed.

## Review checkpoint

Stage 1 is complete. Review the runner, configuration, README, raw results and this report.
No files were staged, committed or pushed. Stop here for the user to commit/push manually.
