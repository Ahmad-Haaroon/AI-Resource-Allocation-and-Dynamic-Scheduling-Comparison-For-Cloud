"""Run unmodified Decima heuristics in isolated processes; preserve raw evidence."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = 'c010dd74ff4b7566bd0ac989c90a32cfbc630d84'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(rows):
    if not rows:
        raise ValueError('No completed jobs')
    jcts = []
    for row in rows:
        arrival, completion = float(row['arrival_ms']), float(row['completion_ms'])
        if not math.isfinite(arrival) or not math.isfinite(completion) or completion < arrival:
            raise ValueError('Invalid job timestamps')
        jcts.append((completion - arrival) / 1000)
    return {'completed_jobs': len(rows), 'mean_jct_s': math.fsum(jcts) / len(jcts),
            'makespan_s': max(float(r['completion_ms']) for r in rows) / 1000}


def worker(opts):
    cfg = json.loads(opts.config.read_text())
    sys.path.insert(0, str(opts.source))
    # Upstream parses arguments on import. Supply every workload parameter explicitly.
    sys.argv = ['decima', '--exec_cap', str(cfg['executors']),
                '--num_init_dags', str(cfg['initial_jobs']),
                '--num_stream_dags', str(cfg['streaming_jobs']),
                '--stream_interval', str(cfg['arrival_interval_ms']),
                '--job_folder', str(opts.source / 'spark_env/tpch') + '/']
    import numpy as np
    from spark_env.env import Environment
    from heuristic_agent import DynamicPartitionAgent
    from spark_agent import SparkAgent
    from param import args
    np.random.seed(opts.seed)
    env = Environment()
    env.seed(opts.seed)
    started = time.perf_counter()
    env.reset()
    # Capture job identity before scheduling; timeline contains streaming arrivals.
    jobs = list(env.job_dags) + [event[2] for event in sorted(env.timeline.pq)]
    agent = DynamicPartitionAgent() if opts.policy == 'dynamic_partition' else SparkAgent(cfg['executors'])
    obs, done, steps, reward_sum = env.observe(), False, 0, 0.0
    while not done:
        node, count = agent.get_action(obs)
        obs, reward, done = env.step(node, count)
        reward_sum += reward
        steps += 1
        if steps > 10_000_000:
            raise RuntimeError('Decision limit exceeded')
    if len(env.finished_job_dags) != len(jobs) or len(jobs) != cfg['initial_jobs'] + cfg['streaming_jobs']:
        raise RuntimeError('Incomplete job execution')
    rows = [{'job_id': i, 'profile': job.name, 'arrival_ms': job.start_time,
             'completion_ms': job.completion_time,
             'jct_ms': job.completion_time - job.start_time} for i, job in enumerate(jobs)]
    summary = metrics(rows)
    # Independent consistency check against the simulator's integrated JCT reward.
    if not math.isclose(-reward_sum * args.reward_scale / 1000,
                        summary['mean_jct_s'] * len(rows), rel_tol=1e-9, abs_tol=1e-6):
        raise RuntimeError('Reward/JCT accounting mismatch')
    with (opts.output / 'jobs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update(policy=opts.policy, seed=opts.seed, decisions=steps,
                   elapsed_wall_s=time.perf_counter()-started,
                   reward_jct_check='passed', effective_upstream_args=vars(args))
    write_json(opts.output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['policy', 'seed', 'completed_jobs', 'mean_jct_s', 'makespan_s']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--policy', choices=['dynamic_partition', 'spark_fifo'])
    parser.add_argument('--seed', type=int)
    opts = parser.parse_args()
    opts.source, opts.config, opts.output = opts.source.resolve(), opts.config.resolve(), opts.output.resolve()
    if opts.worker:
        worker(opts)
        return
    cfg = json.loads(opts.config.read_text())
    for key in ['executors', 'initial_jobs', 'streaming_jobs', 'arrival_interval_ms']:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Provide distinct seeds')
    if not cfg['policies'] or not set(cfg['policies']) <= {'dynamic_partition', 'spark_fifo'}:
        raise ValueError('Unsupported policies')
    commit = subprocess.check_output(['git', '-C', str(opts.source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('Unexpected upstream commit: ' + commit)
    status = subprocess.check_output(['git', '-C', str(opts.source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if status.strip():
        raise ValueError('Upstream tracked files have changes; use a clean checkout')
    # Do not overwrite any previous experiment, including failed runs.
    opts.output.mkdir(parents=True, exist_ok=False)
    write_json(opts.output / 'config.json', cfg)
    files = sorted(list(opts.source.rglob('*.py')) + list((opts.source / 'spark_env/tpch').rglob('*.npy')))
    import numpy, networkx
    write_json(opts.output / 'provenance.json', {
        'upstream_url': 'https://github.com/hongzimao/decima-sim', 'upstream_commit': commit,
        'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
        'logical_cpus': os.cpu_count(), 'numpy': numpy.__version__, 'networkx': networkx.__version__,
        'runner_sha256': digest(Path(__file__)),
        'files_sha256': {str(p.relative_to(opts.source)): digest(p) for p in files}})
    summaries, workloads = [], {}
    for policy in cfg['policies']:
        for seed in cfg['seeds']:
            run = opts.output / f'{policy}_{seed}'
            run.mkdir()
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--source', str(opts.source),
                       '--config', str(opts.output / 'config.json'), '--output', str(run), '--policy', policy, '--seed', str(seed)]
            with (run / 'execution.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1'})
            if result.returncode:
                raise RuntimeError(f'Run failed; preserved log: {run / "execution.log"}')
            rows = list(csv.DictReader((run / 'jobs.csv').open()))
            workload = [(r['profile'], r['arrival_ms']) for r in rows]
            if seed in workloads and workloads[seed] != workload:
                raise RuntimeError('Policies received different arrival workloads')
            workloads[seed] = workload
            summary = json.loads((run / 'summary.json').read_text())
            summaries.append({k: summary[k] for k in ['policy', 'seed', 'completed_jobs', 'mean_jct_s', 'makespan_s']})
            print(summaries[-1], flush=True)
    write_json(opts.output / 'summary.json', summaries)
    write_json(opts.output / 'checks.json', {'all_jobs_completed': True, 'paired_arrivals_identical': True,
                                            'reward_jct_accounting': 'passed'})


if __name__ == '__main__':
    main()
