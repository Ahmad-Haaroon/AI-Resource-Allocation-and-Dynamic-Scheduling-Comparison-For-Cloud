"""Bounded upstream training launcher with reliable worker cleanup and saved logs.

Arguments after -- are passed to the compatibility copy's original train.py.
"""
import argparse
import json
import hashlib
import signal
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('upstream_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.source, a.output = a.source.resolve(), a.output.resolve()
    tail = a.upstream_args[1:] if a.upstream_args[:1] == ['--'] else a.upstream_args
    if a.child:
        sys.path.insert(0, str(a.source))
        sys.argv = ['train.py', '--job_folder', str(a.source/'spark_env/tpch')+'/',
                    '--model_folder', str(a.output/'models')+'/', '--result_folder', str(a.output/'events')+'/', *tail]
        try:
            import train
            train.main()
        finally:
            for child in mp.active_children():
                child.terminate()
                child.join(timeout=10)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--child',
               '--source', str(a.source), '--output', str(a.output), '--', *tail]
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'TF_NUM_INTRAOP_THREADS': '1',
           'TF_NUM_INTEROP_THREADS': '1', 'OMP_NUM_THREADS': '1', 'TF_CPP_MIN_LOG_LEVEL': '2',
           'PYTHONDONTWRITEBYTECODE': '1'}
    (a.output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    (a.output/'compat_manifest.json').write_bytes((a.source/'compat_manifest.json').read_bytes())
    (a.output/'runner_hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}, indent=2)+'\n')
    started = time.perf_counter()
    with (a.output/'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=a.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            code = 124
    (a.output/'status.json').write_text(json.dumps({'exit_code': code,
        'elapsed_wall_s': time.perf_counter()-started, 'python': sys.version}, indent=2)+'\n')
    if code:
        raise SystemExit(f'Training failed; see {a.output / "training.log"}')
    print('Training completed:', a.output)


if __name__ == '__main__':
    main()
