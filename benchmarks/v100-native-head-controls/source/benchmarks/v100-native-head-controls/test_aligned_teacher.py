"""CPU-only fail-closed tests for diagnostic schemas, not model qualification."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('aligned_teacher', Path(__file__).with_name('aligned_teacher.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ProbabilitySchemaTests(unittest.TestCase):
    def setUp(self):
        self.response = {'tokens': [0], 'completion_probabilities': [{'top_logprobs': [{'id': i, 'logprob': -math.log(128) - i / 1000} for i in range(64)]}]}

    def test_current_presampling_schema(self):
        self.assertEqual(len(module.parse_probabilities(self.response)), 64)

    def test_legacy_presampling_id_probability_schema(self):
        r = {'tokens': [0], 'probs': [{'probs': [{'id': i, 'prob': math.exp(-math.log(128) - i / 1000)} for i in range(64)]}]}
        self.assertEqual(set(module.parse_probabilities(r)), set(range(64)))

    def test_reject_missing_rows(self):
        for rows in [None, [], [{}, {}]]:
            r = copy.deepcopy(self.response)
            r['completion_probabilities'] = rows
            with self.assertRaises(AssertionError):
                module.parse_probabilities(r)

    def test_reject_ambiguous_or_invalid_ids(self):
        for token in [True, None, -1, 248320, 0]:
            r = copy.deepcopy(self.response)
            r['completion_probabilities'][0]['top_logprobs'][1]['id'] = token
            with self.assertRaises(AssertionError):
                module.parse_probabilities(r)

    def test_reject_nonfinite_or_positive_logprob(self):
        for value in [float('nan'), float('inf'), float('-inf'), .01]:
            r = copy.deepcopy(self.response)
            r['completion_probabilities'][0]['top_logprobs'][0]['logprob'] = value
            with self.assertRaises(AssertionError):
                module.parse_probabilities(r)

    def test_reject_greedy_point_mass(self):
        r = copy.deepcopy(self.response)
        rows = r['completion_probabilities'][0]['top_logprobs']
        for i, row in enumerate(rows):
            row['logprob'] = 0 if i == 0 else -100
        with self.assertRaises(AssertionError):
            module.parse_probabilities(r)

    def test_reject_missing_alternatives(self):
        r = copy.deepcopy(self.response)
        r['completion_probabilities'][0]['top_logprobs'].pop()
        with self.assertRaises(AssertionError):
            module.parse_probabilities(r)

    def test_reject_misaligned_generated_id(self):
        r = copy.deepcopy(self.response)
        r['tokens'] = [1]
        with self.assertRaises(AssertionError):
            module.parse_probabilities(r)

if __name__ == '__main__':
    unittest.main()
