import copy
import numpy as np
import pytest
from complex5g.generator import PROTOCOL, N, generate, datasets
from complex5g.features import LearnedFeatures, trailing_mean


@pytest.mark.parametrize("kind", PROTOCOL["kinds"])
def test_counterfactual_preserves_context_and_unaffected_observables(kind):
    clean = generate(1234, "pair-False", kind, False)
    attacked = generate(1234, "pair-True", kind, True)
    e = attacked.event
    assert np.array_equal(clean.context, attacked.context)
    assert np.array_equal(clean.phase, attacked.phase)
    assert np.array_equal(clean.generation, attacked.generation)
    mask = np.ones(clean.tokens.shape, dtype=bool)
    mask[e["start"]:e["end"], e["nf"]] = False
    for field in ("x", "tokens", "destinations"):
        assert np.array_equal(getattr(clean, field)[mask], getattr(attacked, field)[mask])
    if kind not in {"integrity_only", "combined"}:
        assert not attacked.hard.any()
    if kind == "invisible":
        for field in ("x", "tokens", "destinations", "hard", "unknown"):
            assert np.array_equal(getattr(clean, field), getattr(attacked, field))
        assert not attacked.behavior.any()


@pytest.mark.parametrize("kind", ["order_permutation", "cross_nf_desync", "sparse_reorder", "combined"])
def test_order_changes_preserve_histogram(kind):
    clean = generate(4321, "clean", kind, False)
    attacked = generate(4321, "attacked", kind, True)
    e = attacked.event
    sl = np.s_[e["start"]:e["end"], e["nf"]]
    assert np.array_equal(np.sort(clean.tokens[sl]), np.sort(attacked.tokens[sl]))
    assert not np.array_equal(clean.tokens[sl], attacked.tokens[sl])
    assert np.array_equal(clean.x, attacked.x)


def test_joint_attack_preserves_all_raw_univariate_values():
    clean = generate(998, "clean", "joint_shuffle", False)
    attacked = generate(998, "attacked", "joint_shuffle", True)
    e = attacked.event
    for feature in range(6):
        sl = np.s_[e["start"]:e["end"], e["nf"], feature]
        assert np.array_equal(np.sort(clean.x[sl]), np.sort(attacked.x[sl]))
    assert not np.array_equal(clean.x, attacked.x)


def test_history_is_causal_and_does_not_read_labels():
    normals = [generate(i, f"normal-{i}", "order_permutation", False) for i in range(3)]
    features = LearnedFeatures(normals)
    r = generate(987, "test", "cross_nf_desync", True)
    original, components = features.transform(r)
    changed = copy.deepcopy(r)
    changed.tokens[128:] = (changed.tokens[128:] + 3) % 6
    changed.destinations[128:] = (changed.destinations[128:] + 4) % N
    changed.x[128:] += 100
    future, future_components = features.transform(changed)
    assert np.allclose(original[:128], future[:128])
    assert np.allclose(components[:128], future_components[:128])
    changed = copy.deepcopy(r)
    changed.attack = ~changed.attack
    changed.behavior = ~changed.behavior
    changed.integrity = ~changed.integrity
    changed.event["kind"] = "arbitrary_evaluation_label"
    assert np.array_equal(original, features.transform(changed)[0])


def test_structural_novelty_component_responds_explicitly():
    normals = [generate(i, f"normal-{i}", "edge_rebinding", False) for i in range(3)]
    features = LearnedFeatures(normals)
    r = generate(765, "test", "edge_rebinding", False)
    original = features.transform(r)[1]
    changed = copy.deepcopy(r)
    nf = 1
    t = int(np.flatnonzero(~r.phase[:, nf])[20])
    known = features.edges[(nf, 0)]
    new = next(peer for peer in range(N) if peer not in known)
    changed.destinations[t, nf] = new
    assert features.transform(changed)[1][t, nf, -1] > original[t, nf, -1]


def test_training_excludes_heldout_and_uses_behavior_labels():
    _, train, calibration, test = datasets(2)
    events = [r.event for r in train if r.event]
    assert not set(PROTOCOL["heldout_kinds"]) & {e["kind"] for e in events}
    assert "invisible" not in {e["kind"] for e in events}
    assert {r.run_id for r in train}.isdisjoint(r.run_id for r in calibration + test)
    assert all(not r.attack.any() for r in calibration)
    for r in train:
        if r.event and r.event["kind"] == "integrity_only":
            assert r.integrity.any() and not r.behavior.any()


def test_hybrid_retains_every_current_integrity_alert():
    r = generate(444, "attack", "combined", True)
    assert r.hard.any()
    model = np.zeros_like(r.hard)
    assert np.array_equal(r.hard | model, r.hard)
    r2 = generate(444, "normal", "combined", False)
    assert not r2.hard.any()
