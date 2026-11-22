"""Validate recorded pilot updates and summarize measured timings (not accuracy)."""
import argparse
import json
from pathlib import Path
import re
import statistics
import tensorflow.compat.v1 as tf
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot', type=Path)
    a = p.parse_args()
    status = json.loads((a.pilot/'status.json').read_text())
    if status['exit_code'] != 0:
        raise ValueError('Pilot did not exit successfully')
    log = (a.pilot/'training.log').read_text()
    checkpoints = sorted((a.pilot/'models').glob('model_ep_*.index'))
    if len(checkpoints) != 3:
        raise AssertionError('Expected three saved updates; investigate skipped/failed epochs')
    first, last = [tf.train.load_checkpoint(str(p.with_suffix(''))) for p in [checkpoints[0], checkpoints[-1]]]
    changed = []
    for name in last.get_variable_to_shape_map():
        value = last.get_tensor(name)
        if not np.isfinite(value).all():
            raise AssertionError('Nonfinite checkpoint value: '+name)
        if 'Adam' not in name and 'beta' not in name and not np.array_equal(first.get_tensor(name), value):
            changed.append(name)
    if not changed:
        raise AssertionError('No model variables changed between first and last update')
    times = []
    for block in re.split(r'training epoch \d+', log)[1:]:
        parts = re.findall(r'(?:got reward from workers|advantage ready|worker send back gradients|apply gradient) ([\d.eE+-]+)', block)
        if len(parts) == 4:
            times.append(sum(float(x) for x in parts))
    summaries = []
    for event_file in a.pilot.rglob('events.out.tfevents.*'):
        for event in tf.train.summary_iterator(str(event_file)):
            if event.summary.value:
                row = {'step': event.step, 'values': {v.tag: v.simple_value for v in event.summary.value}}
                if not all(np.isfinite(v) for v in row['values'].values()):
                    raise AssertionError('Nonfinite training summary')
                summaries.append(row)
    if len(summaries) != 3 or len(times) != 3:
        raise AssertionError('Missing training evidence')
    report = {'successful_updates': 3, 'finite_checkpoint_values': True,
              'changed_model_variables_first_to_last': changed,
              'update_compute_seconds': times, 'median_update_compute_seconds': statistics.median(times),
              'post_startup_update_seconds': times[1:],
              'total_process_wall_seconds': status['elapsed_wall_s'], 'training_summaries': summaries,
              'caution': 'Three updates with two workers; stochastic episode horizons vary. Not convergence or final performance evidence.'}
    (a.pilot/'pilot_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['training_summaries','changed_model_variables_first_to_last']},indent=2))


if __name__ == '__main__':
    main()
