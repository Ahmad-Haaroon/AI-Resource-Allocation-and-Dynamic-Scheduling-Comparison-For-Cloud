"""Recalculate CSV metrics and compare two independent batches, without hiding differences."""
import csv
import json
from pathlib import Path
import sys
from run_baselines import metrics, write_json


def check(first, second):
    a, b = Path(first), Path(second)
    if json.loads((a/'config.json').read_text()) != json.loads((b/'config.json').read_text()):
        raise ValueError('Different configurations')
    comparisons = []
    for entry in json.loads((a/'summary.json').read_text()):
        name = f"{entry['policy']}_{entry['seed']}"
        for root in [a, b]:
            rows = list(csv.DictReader((root/name/'jobs.csv').open()))
            recorded = json.loads((root/name/'summary.json').read_text())
            for key, value in metrics(rows).items():
                if value != recorded[key]:
                    raise ValueError(f'{root}/{name}: {key} mismatch')
        comparisons.append({'run': name, 'identical_job_csv': (a/name/'jobs.csv').read_bytes() == (b/name/'jobs.csv').read_bytes()})
    report = {'csv_metrics_verified': True, 'comparisons': comparisons,
              'all_job_results_identical': all(x['identical_job_csv'] for x in comparisons)}
    write_json(b.parent/'repeatability.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    check(*sys.argv[1:])
