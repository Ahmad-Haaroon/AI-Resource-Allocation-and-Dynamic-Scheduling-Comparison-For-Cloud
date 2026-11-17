import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_baselines import metrics


class MetricsTests(unittest.TestCase):
    def test_arrival_offset_and_units(self):
        result = metrics([{'arrival_ms': 1000, 'completion_ms': 4000},
                          {'arrival_ms': 3000, 'completion_ms': 5000}])
        self.assertEqual(result, {'completed_jobs': 2, 'mean_jct_s': 2.5, 'makespan_s': 5.0})

    def test_invalid_or_missing_completion(self):
        for rows in [[], [{'arrival_ms': 3, 'completion_ms': 2}],
                     [{'arrival_ms': 0, 'completion_ms': float('inf')}]]:
            with self.assertRaises(ValueError):
                metrics(rows)


if __name__ == '__main__':
    unittest.main()
