"""
tests/test_reward_and_env.py
-----------------------------
Unit tests for reward function and therapy environment.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dapta.utils.reward import (
    compute_reward,
    compute_batch_rewards,
    DEFAULT_WEIGHTS,
    METRIC_ORDER,
    REGRESSION_PENALTY,
)
from dapta.prta.action_space import (
    THERAPY_EXERCISES, N_ACTIONS, get_exercise,
    DISCOURSE_FUNCTIONAL_ACTIONS, get_generalisation_potentials,
)


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestRewardFunction:

    def test_positive_reward_on_improvement(self):
        before = np.array([0.3, 0.4, 0.5, 0.5, 0.3, 0.8], dtype=np.float32)
        after  = np.array([0.5, 0.6, 0.7, 0.6, 0.4, 0.6], dtype=np.float32)
        reward = compute_reward(before, after)
        assert reward > 0.0, "Improvement in all metrics should yield positive reward."

    def test_negative_reward_on_regression(self):
        before = np.array([0.7, 0.8, 0.6, 0.6, 0.4, 0.4], dtype=np.float32)
        after  = np.array([0.2, 0.3, 0.5, 0.5, 0.3, 0.9], dtype=np.float32)
        reward = compute_reward(before, after)
        assert reward < 0.0, "Regression in CIU/MC should yield negative reward."

    def test_surprisal_is_inverse(self):
        # Surprisal reduction (improvement) should increase reward
        before = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.8], dtype=np.float32)  # high surprisal
        after  = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.4], dtype=np.float32)  # low surprisal
        reward = compute_reward(before, after)
        assert reward > 0.0, "Reduction in surprisal should yield positive reward."

    def test_reward_clipped(self):
        before = np.zeros(6, dtype=np.float32)
        after  = np.ones(6, dtype=np.float32)
        reward = compute_reward(before, after, clip=(-1.0, 1.0))
        assert -1.0 <= reward <= 1.0

    def test_no_change_zero_reward(self):
        state = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        reward = compute_reward(state, state.copy())
        # Regression penalty fires if CIU/MC unchanged (delta=0, no positive)
        # but delta=0 means no regression either; exact value depends on penalty
        # Just check it's small
        assert abs(reward) <= 0.25

    def test_regression_penalty_applied(self):
        before = np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        after  = np.array([0.5, 0.9, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)  # CIU drops
        reward = compute_reward(before, after)
        # Should include regression penalty
        assert reward <= 0.0 or reward < compute_reward(
            np.array([0.8, 0.8, 0.5, 0.5, 0.5, 0.5]),
            np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.5])  # no regression
        )

    def test_batch_rewards_shape(self):
        B = 10
        before = np.random.rand(B, 6).astype(np.float32)
        after  = np.random.rand(B, 6).astype(np.float32)
        rewards = compute_batch_rewards(before, after)
        assert rewards.shape == (B,)
        assert rewards.dtype == np.float32

    def test_batch_equals_single(self):
        before = np.random.rand(5, 6).astype(np.float32)
        after  = np.random.rand(5, 6).astype(np.float32)
        batch_r = compute_batch_rewards(before, after)
        single_r = np.array([compute_reward(before[i], after[i]) for i in range(5)])
        np.testing.assert_allclose(batch_r, single_r, atol=1e-5)

    def test_invalid_weights_raise(self):
        bad_weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
        with pytest.raises(ValueError):
            compute_reward(
                np.zeros(6, dtype=np.float32),
                np.ones(6, dtype=np.float32),
                weights=bad_weights,
            )


# ---------------------------------------------------------------------------
# Action space tests
# ---------------------------------------------------------------------------

class TestActionSpace:

    def test_n_actions(self):
        assert N_ACTIONS == 12
        assert len(THERAPY_EXERCISES) == 12

    def test_action_ids_unique_and_sequential(self):
        ids = [ex.action_id for ex in THERAPY_EXERCISES]
        assert ids == list(range(N_ACTIONS))

    def test_all_target_levels_valid(self):
        valid = {"word", "sentence", "discourse"}
        for ex in THERAPY_EXERCISES:
            assert ex.target_level in valid

    def test_all_contexts_valid(self):
        valid = {"structured_drill", "functional"}
        for ex in THERAPY_EXERCISES:
            assert ex.context in valid

    def test_generalisation_potentials_range(self):
        potentials = get_generalisation_potentials()
        assert len(potentials) == N_ACTIONS
        for p in potentials:
            assert 0.0 <= p <= 1.0

    def test_discourse_functional_highest_potential(self):
        all_potentials = get_generalisation_potentials()
        df_potentials = [all_potentials[i] for i in DISCOURSE_FUNCTIONAL_ACTIONS]
        non_df = [all_potentials[i] for i in range(N_ACTIONS) if i not in DISCOURSE_FUNCTIONAL_ACTIONS]
        assert min(df_potentials) > np.median(non_df), (
            "Discourse-functional exercises should have above-median generalisation potential."
        )

    def test_get_exercise_valid(self):
        for i in range(N_ACTIONS):
            ex = get_exercise(i)
            assert ex.action_id == i

    def test_get_exercise_invalid(self):
        with pytest.raises(ValueError):
            get_exercise(999)

    def test_free_conversation_highest_potential(self):
        ex = get_exercise(11)  # free_conversation_prompting
        potentials = get_generalisation_potentials()
        assert ex.generalisation_potential == max(potentials)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
