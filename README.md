# Decima-based resource allocation and dynamic scheduling

Group 28 · Section 7AB · i221295 / i230040

Authors: Ahmad Haaroon and Ibrahim Babar

This incomplete research project evaluates DAG scheduling in the Decima simulator.
The proposed extension is failure-aware scheduling (Decima-FT). AWS deployment is
outside the current scope. Drift-aware autoscaling was an instructor example,
not this project's topic.

## Stage 1: heuristic baseline and evidence collection

Implemented: an isolated-process runner for upstream fair sharing and FIFO,
per-job timestamps, JCT/makespan calculation, source/data checksums, environment
metadata, logs, arrival-workload pairing checks, and independent repeatability
comparison. See `docs/stage1.md` for measured results and limitations.

Not implemented yet: learned-policy environment/training, tuned weighted fair,
failure injection, Decima-FT, utilization/recovery metrics, or the final paper.

### Setup

The tested interpreter is Python 3.9.6. Use a dedicated environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-stage1.txt
```

Obtain the authors' source separately and pin its version:

```sh
git clone https://github.com/hongzimao/decima-sim.git external/decima-sim
git -C external/decima-sim checkout c010dd74ff4b7566bd0ac989c90a32cfbc630d84
```

The local audit used the already available `../Reference Project/decima-sim`
checkout at that commit. The upstream code and profiles are not redistributed
in this project. The upstream snapshot has no license file; no license grant is
assumed here. Cite Mao et al., *Learning Scheduling Algorithms for Data Processing
Clusters*, SIGCOMM 2019, DOI: 10.1145/3341302.3342080.

### Run

From this repository's root (each output directory must be new):

```sh
python -B -m unittest discover -s tests -v
python -B scripts/run_baselines.py --source external/decima-sim --output results/reproduction/run_a
python -B scripts/run_baselines.py --source external/decima-sim --output results/reproduction/run_b
python -B scripts/check_results.py results/reproduction/run_a results/reproduction/run_b
```

Substitute `--source '../Reference Project/decima-sim'` for the existing sibling
checkout. The runner requires its tracked files to be unchanged and verifies the
commit. It runs no TensorFlow code and never overwrites an experiment directory.
A failed worker retains its execution log and causes the batch to exit with an error.

`configs/smoke.json` specifies 50 executors, one initial job, 200 streaming jobs,
25,000 ms mean arrival interval, and seeds 1000–1002. These are development seeds,
not held-out final-evaluation seeds. Workloads use the 154 bundled TPC-H profiles;
arrivals and task durations are simulated, not AWS measurements.

### Output layout

Each batch includes `config.json`, `provenance.json`, `summary.json`, and
`checks.json`. Each policy/seed directory contains `jobs.csv`, `summary.json`
(including effective upstream parameters), and `execution.log`.
`repeatability.json` compares job-level CSVs across independent batches; a false
identity flag is an observed reproducibility issue, not a successful exact reproduction.

JCT is completion minus arrival, converted from milliseconds to seconds.
Makespan is final completion time measured from simulation time zero.
The JCT sum is checked against the simulator's integrated reward. Identical
arrival workloads are checked across policies. This does not guarantee identical
task-duration samples: policies consume the simulator's randomness differently.

### Remaining stages

1. Resolve and document baseline repeatability and any confirmed upstream defects.
2. Establish a compatible learned-policy environment and benchmark training cost.
3. Train/evaluate Decima with a predeclared validation/test protocol and fair baselines.
4. Implement controlled executor failures and evaluate the proposed extension/ablations.
5. Verify literature and generate an 8–11 page `main.tex` manuscript from real outputs.

Stop for manual review/commit/push after each stage. No automated Git publication.
