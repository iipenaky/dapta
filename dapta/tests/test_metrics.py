"""
tests/test_metrics.py
----------------------
Unit tests for the Discourse Assessment Engine metrics module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dapta.dae.metrics import DiscourseMetricExtractor, DiscourseMetrics


# Sample utterances for testing
NORMAL_SPEECH = [
    "The woman is washing the dishes at the sink.",
    "The water is overflowing because she is not paying attention.",
    "A boy is standing on a stool trying to get cookies from the jar.",
    "The girl is reaching up asking for a cookie.",
    "The stool is tipping over and the boy is about to fall.",
]

APHASIC_SPEECH = [
    "Woman um washing",
    "Water the um",
    "Boy cookie um jar",
    "Girl um want",
]

EMPTY_SPEECH: list = []


class TestDiscourseMetricExtractor:

    def setup_method(self):
        self.extractor_cookie = DiscourseMetricExtractor(
            duration_minutes=2.0,
            task="cookie_theft",
        )
        self.extractor_cinderella = DiscourseMetricExtractor(
            duration_minutes=3.0,
            task="cinderella",
        )

    # ------------------------------------------------------------------
    # CIU Rate
    # ------------------------------------------------------------------

    def test_ciu_rate_normal_higher_than_aphasic(self):
        normal_metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        aphasic_metrics = self.extractor_cookie.compute(APHASIC_SPEECH)
        assert normal_metrics.ciu_rate > aphasic_metrics.ciu_rate, (
            "Normal speech should yield higher CIU rate than aphasic speech."
        )

    def test_ciu_rate_positive(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        assert metrics.ciu_rate >= 0.0

    def test_ciu_rate_empty(self):
        metrics = self.extractor_cookie.compute(EMPTY_SPEECH)
        assert metrics.ciu_rate == 0.0

    # ------------------------------------------------------------------
    # MC Score
    # ------------------------------------------------------------------

    def test_mc_score_range(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        assert 0.0 <= metrics.mc_score <= 1.0

    def test_mc_score_normal_higher_than_aphasic(self):
        normal_metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        aphasic_metrics = self.extractor_cookie.compute(APHASIC_SPEECH)
        assert normal_metrics.mc_score >= aphasic_metrics.mc_score

    def test_mc_score_unknown_task(self):
        extractor = DiscourseMetricExtractor(task="unknown_task")
        metrics = extractor.compute(NORMAL_SPEECH)
        assert metrics.mc_score == 0.0

    # ------------------------------------------------------------------
    # MLU
    # ------------------------------------------------------------------

    def test_mlu_positive(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        assert metrics.mlu_morphemes > 0.0

    def test_mlu_normal_higher_than_aphasic(self):
        normal_metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        aphasic_metrics = self.extractor_cookie.compute(APHASIC_SPEECH)
        assert normal_metrics.mlu_morphemes > aphasic_metrics.mlu_morphemes

    # ------------------------------------------------------------------
    # TTR
    # ------------------------------------------------------------------

    def test_ttr_range(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        assert 0.0 <= metrics.ttr <= 1.0

    def test_ttr_single_word(self):
        metrics = self.extractor_cookie.compute(["cookie"])
        assert metrics.ttr > 0.0

    # ------------------------------------------------------------------
    # Syntactic complexity
    # ------------------------------------------------------------------

    def test_syn_comp_range(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        assert 0.0 <= metrics.syntactic_complexity <= 1.0

    def test_syn_comp_complex_higher_than_simple(self):
        complex_speech = [
            "The woman who is washing dishes does not notice the water.",
            "Because she is distracted, the sink overflows.",
            "The boy climbs the stool although it is unstable.",
        ]
        simple_speech = ["Woman washes.", "Boy falls.", "Girl wants cookie."]
        complex_m = self.extractor_cookie.compute(complex_speech)
        simple_m = self.extractor_cookie.compute(simple_speech)
        assert complex_m.syntactic_complexity >= simple_m.syntactic_complexity

    # ------------------------------------------------------------------
    # to_array / to_dict
    # ------------------------------------------------------------------

    def test_to_array_shape(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        arr = metrics.to_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float32

    def test_to_dict_keys(self):
        metrics = self.extractor_cookie.compute(NORMAL_SPEECH)
        d = metrics.to_dict()
        expected_keys = {
            "ciu_rate", "mc_score", "mlu_morphemes",
            "ttr", "syntactic_complexity", "n_utterances", "n_words", "task"
        }
        assert set(d.keys()) == expected_keys


class TestDiscourseMetrics:

    def test_empty_metrics_all_zero(self):
        extractor = DiscourseMetricExtractor()
        m = extractor._empty_metrics()
        assert m.ciu_rate == 0.0
        assert m.mc_score == 0.0
        assert m.n_utterances == 0


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
