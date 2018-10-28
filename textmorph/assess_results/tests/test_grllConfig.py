from unittest import TestCase
from textmorph.assess_results.config import GrllConfig

class TestGrllConfig(TestCase):
    def setUp(self):
        self.configs = GrllConfig("nofile")

    def test_merge_at_first_level(self):
        truth = {"a": 0}
        completing = {"a": 5, "b": 10}
        r = self.configs.merge(truth, completing)

        self.assertDictEqual(r, {"a": 0, "b": 10})

    def test_merge_at_second_level(self):
        truth = {"a": {"b" : 0, "c": 1}}
        completing = {"a": {"b": 1, "d": 10}}
        r = self.configs.merge(truth, completing)

        self.assertDictEqual(r, {"a": {"b" : 0, "c": 1, "d": 10}})

    def test_merge_at_third_level(self):
        truth = {"a": {"b" : {"d": 5}, "c": 1}}
        completing = {"a": {"b": {"d":2, "e":10}, "d": 10}}
        r = self.configs.merge(truth, completing)

        self.assertDictEqual(r, {"a": {"b" : {"d": 5, "e":10}, "d": 10, "c": 1}})

    def test_merge_unbalanced(self):
        truth = {"a": {"b" : 10, "c": 1}}
        completing = {"a": {"b": {"d":2, "e":10}, "d": 10}}
        r = self.configs.merge(truth, completing)

        self.assertDictEqual(r, {"a": {"b" : 10, "d": 10, "c": 1}})