"""Preserve the original fail-closed contract and cover Float32-rounded peaks."""
import copy
import importlib.util
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, OUT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

repaired = load('aligned_teacher_eos_f32', 'aligned_teacher_eos_f32.py')
prior_tests = load('prior_probability_schema_contract', 'test_aligned_teacher.py')
prior_tests.module = repaired

class RepairedProbabilitySchemaTests(prior_tests.ProbabilitySchemaTests):
    def test_accept_float32_rounded_peak_with_resolved_tail(self):
        r = copy.deepcopy(self.response)
        for i, row in enumerate(r['completion_probabilities'][0]['top_logprobs']):
            row['logprob'] = 0.0 if i == 0 else -18.01982879638672 if i == 1 else -28.0
        values = repaired.parse_probabilities(r)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(len(values), 64)

    def test_reject_underflowed_alternatives(self):
        r = copy.deepcopy(self.response)
        for i, row in enumerate(r['completion_probabilities'][0]['top_logprobs']):
            row['logprob'] = 0 if i == 0 else -1000
        with self.assertRaises(AssertionError):
            repaired.parse_probabilities(r)

    def test_reject_rounded_peak_with_inconsistent_tail(self):
        r = copy.deepcopy(self.response)
        for i, row in enumerate(r['completion_probabilities'][0]['top_logprobs']):
            row['logprob'] = 0 if i == 0 else -15
        with self.assertRaises(AssertionError):
            repaired.parse_probabilities(r)

    def test_no_probability_clamping(self):
        values = repaired.parse_probabilities(self.response)
        self.assertEqual(values[0], -math.log(128))

    def test_reject_positive_peak_with_resolved_tail(self):
        r = copy.deepcopy(self.response)
        for i, row in enumerate(r['completion_probabilities'][0]['top_logprobs']):
            row['logprob'] = 1e-8 if i == 0 else -18.1 if i == 1 else -28.0
        with self.assertRaises(AssertionError):
            repaired.parse_probabilities(r)
