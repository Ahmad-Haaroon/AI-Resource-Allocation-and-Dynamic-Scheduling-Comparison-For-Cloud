"""Run the predeclared short pilot, leaving full training for the next stage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/training-pilot.json')
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    arguments = []
    for key, value in config.items():
        if key not in {'purpose', 'note'}:
            arguments += ['--'+key, str(value)]
    subprocess.run([sys.executable, '-B', str(ROOT/'scripts/train_decima.py'),
                    '--source', a.source, '--output', a.output, '--', *arguments], check=True)
