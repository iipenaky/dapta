"""
Unit tests for PatientStateBuilder and PatientProfile.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dapta.dae.state_builder import (
    PatientProfile, PatientStateBuilder,
    STATE_DIM, N_DISCOURSE, N_SUBTYPE, APHASIA_SUBTYPES
)
from dapta.dae.metrics import DiscourseMetricExtractor


def make_profile(subtype="Anomic", wab_aq=70.0, months=12.0):
    return PatientProfile("P001", aphasia_subtype=subtype, wab_aq=wab_aq, months_post_onset=months)


def make_metrics(ciu=50.0, mc=0.5, mlu=5.0, ttr=0.6, syn=0.3):
    from dapta.dae.metrics import DiscourseMetrics
    return DiscourseMetrics(
        ciu_rate=ciu, mc_score=mc, mlu_morphemes=mlu,
        ttr=ttr, syntactic_complexity=syn,
        n_utterances=10, n_words=50, task="cookie_theft"
    )


def build_fitted_builder(n_patients=20):
    """Build a fitted state builder from synthetic data."""
    metrics_list = [make_metrics(
        ciu=float(np.random.uniform(10, 150)),
        mc=float(np.random.uniform(0.1, 0.9)),
        mlu=float(np.random.uniform(2, 12)),
        ttr=float(np.random.uniform(0.3, 0.9)),
        syn=float(np.random.uniform(0.0, 0.6)),
    ) for _ in range(n_patients)]
    surprisals = list(np.random.uniform(1.0, 8.0, n_patients))
    profiles = [make_profile(
        subtype=APHASIA_SUBTYPES[i % len(APHASIA_SUBTYPES)],
        wab_aq=float(np.random.uniform(10, 100)),
        months=float(np.random.uniform(1, 48)),
    ) for i in range(n_patients)]

    builder = PatientStateBuilder()
    builder.fit(metrics_list, surprisals, profiles)
    return builder


class TestPatientProfile:

    def test_subtype_normalisation(self):
        p = PatientProfile("P001", aphasia_subtype="broca")
        assert p.aphasia_subtype == "Broca"

    def test_wab_aq_clamped(self):
        p = PatientProfile("P001", wab_aq=150.0)
        assert p.wab_aq == 100.0
        p2 = PatientProfile("P001", wab_aq=-10.0)
        assert p2.wab_aq == 0.0

    def test_one_hot_shape(self):
        p = make_profile("Broca")
        ohe = p.subtype_one_hot()
        assert ohe.shape == (N_SUBTYPE,)
        assert ohe.sum() == 1.0

    def test_one_hot_correct_index(self):
        p = make_profile("Anomic")
        ohe = p.subtype_one_hot()
        assert ohe[APHASIA_SUBTYPES.index("Anomic")] == 1.0

    def test_unknown_subtype_maps_to_other(self):
        p = PatientProfile("P001", aphasia_subtype="unknown_type")
        assert p.aphasia_subtype == "Other"


class TestPatientStateBuilder:

    def setup_method(self):
        self.builder = build_fitted_builder(n_patients=30)

    def test_build_shape(self):
        m = make_metrics()
        p = make_profile()
        state = self.builder.build(m, 3.5, p)
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32

    def test_build_normalised_range(self):
        m = make_metrics()
        p = make_profile()
        state = self.builder.build(m, 3.5, p)
        # Discourse part should be roughly [0, 1] after normalisation
        assert np.all(state[:N_DISCOURSE] >= -0.1)
        assert np.all(state[:N_DISCOURSE] <= 1.1)

    def test_build_batch_shape(self):
        n = 5
        metrics = [make_metrics() for _ in range(n)]
        surprisals = [3.0] * n
        profiles = [make_profile() for _ in range(n)]
        batch = self.builder.build_batch(metrics, surprisals, profiles)
        assert batch.shape == (n, STATE_DIM)

    def test_trajectory_shape(self):
        T = 7
        metrics = [make_metrics() for _ in range(T)]
        surprisals = [3.0] * T
        p = make_profile()
        traj = self.builder.build_trajectory(metrics, surprisals, p)
        assert traj.shape == (T, STATE_DIM)

    def test_not_fitted_raises(self):
        builder = PatientStateBuilder()
        with pytest.raises(RuntimeError):
            builder.build(make_metrics(), 3.0, make_profile())

    def test_different_subtypes_different_states(self):
        m = make_metrics()
        s = 3.5
        state_broca  = self.builder.build(m, s, make_profile("Broca"))
        state_anomic = self.builder.build(m, s, make_profile("Anomic"))
        # One-hot differs → states should differ
        assert not np.allclose(state_broca, state_anomic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
