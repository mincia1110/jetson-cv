import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import camera_conditions as conditions


class ConditionTests(unittest.TestCase):
    def test_rgb_channel_and_inclusive_boundaries(self):
        config = {'BRIGHT_min': 100, 'BRIGHT_max': 100, 'RG_gab': 20}
        result = conditions.assess_rgb_means(100, 80, 20, config)
        self.assertEqual(result['bright_b_actual_R'], 100)
        self.assertEqual(result['RG_diff'], 20)
        self.assertEqual(result['reasons'], [])
        config['BRIGHT_max'] = 99
        config['RG_gab'] = 19
        self.assertEqual(len(conditions.assess_rgb_means(100, 80, 20, config)['reasons']), 2)

    def test_fractional_values_follow_test_py_integer_conversion(self):
        config = {'BRIGHT_min': 100, 'BRIGHT_max': 100, 'RG_gab': 20}
        result = conditions.assess_rgb_means(100.9, 80.1, 0, config)
        self.assertEqual(result['reasons'], [])


if __name__ == '__main__':
    unittest.main()
