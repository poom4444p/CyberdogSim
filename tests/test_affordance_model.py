"""Tests for affordance/model.py: the network's input, the refusal rule, save/load.

No dataset and no training: these check the plumbing every caller relies on.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cyberdog.affordance import data, model as M  # noqa: E402


def _flat(n=2):
    return np.zeros((n, data.NY, data.NX), np.float32), np.tile([[1.0, 0.0]], (n, 1))


class TestFeatures:
    def test_shape(self):
        p, t = _flat(3)
        assert M.features(p, t).shape == (3, M.N_IN)

    def test_flat_ground_has_no_jumps(self):
        p, _ = _flat(1)
        assert M.jumps(p).max() == 0.0

    def test_a_step_jumps_on_both_sides_of_its_edge(self):
        p, _ = _flat(1)
        p[0, :, 10:] = 0.15
        j = M.jumps(p)[0]
        assert np.allclose(j[:, 9], 0.15) and np.allclose(j[:, 10], 0.15)
        assert j[:, :9].max() == 0.0 and j[:, 11:].max() == 0.0

    def test_unseen_is_not_an_edge(self):
        # The border of what the LiDAR saw must not read as a cliff.
        p, _ = _flat(1)
        p[0, :, 10:] = np.nan
        assert M.jumps(p).max() == 0.0
        x = M.features(p, [[1.0, 0.0]])
        assert np.isfinite(x).all()

    def test_mirroring_twice_is_the_same_sample(self):
        rng = np.random.default_rng(0)
        p = rng.normal(0, 0.05, (4, data.NY, data.NX)).astype(np.float32)
        x = torch.as_tensor(M.features(p, rng.uniform(-1, 1, (4, 2))))
        assert torch.equal(M.mirror(M.mirror(x)), x)

    def test_mirror_matches_features_of_the_mirrored_world(self):
        rng = np.random.default_rng(1)
        p = rng.normal(0, 0.05, (2, data.NY, data.NX)).astype(np.float32)
        t = np.array([[1.0, 0.4], [0.8, -0.3]])
        mirrored = M.features(p[:, ::-1, :].copy(), t * [1, -1])
        assert np.allclose(M.mirror(torch.as_tensor(M.features(p, t))).numpy(), mirrored)


class TestDecide:
    def test_refuses_at_refuse_p_even_when_not_the_most_likely(self):
        probs = np.array([[0.7, 0.1, M.REFUSE_P]])
        assert M.decide(probs).tolist() == [data.NOT_WALKABLE]

    def test_below_refuse_p_picks_walkable_or_caution(self):
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        assert M.decide(probs).tolist() == [data.WALKABLE, data.CAUTION]

    def test_refuse_p_is_strict_for_blind_users(self):
        # The design decision: a 1-in-5 chance of an edge is enough to refuse.
        assert M.REFUSE_P <= 0.2


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(0)
    net = M.AffordanceMLP().eval()
    p, t = _flat(2)
    before, _ = M.classify(net, p, t)
    M.save(net, tmp_path / "mlp.pt", note="test")
    loaded, info = M.load(tmp_path / "mlp.pt")
    after, _ = M.classify(loaded, p, t)
    assert info["note"] == "test" and before.tolist() == after.tolist()
