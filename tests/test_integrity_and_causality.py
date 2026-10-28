import copy
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import pytest
from hybrid.benchmark import schedule, sequential, threshold
from hybrid.simulation import POLICY, SLOTS, fuse, run
from hybrid.vendor.stage39_kie import (
    FreshnessAnchor, KIERequest, KIEResponse, SecurityError,
    StreamRelationalVerifier, default_profiles, mutate,
)


@pytest.mark.parametrize("kind", ["identity", "endpoint", "service", "fqdn", "type", "compound", "duplicate"])
def test_original_verifier_retains_detection(kind):
    profiles, witnesses = default_profiles(POLICY)
    for slot in SLOTS:
        assert StreamRelationalVerifier(POLICY).verify(mutate(profiles, slot, kind), witnesses)


def test_other_slot_constraints_remain_during_authorized_restart():
    profiles, witnesses = default_profiles(POLICY)
    verifier = StreamRelationalVerifier(POLICY)
    verifier.begin_transition("restart", SLOTS[0])
    assert "semantic_mismatch:" + SLOTS[1] in verifier.verify(mutate(profiles, SLOTS[1], "endpoint"), witnesses)


def test_accepted_generation_change_and_missing_evidence_are_distinct():
    profiles, witnesses = default_profiles(POLICY)
    profiles[0]["nfInstanceId"] = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    witnesses[SLOTS[0]]["nf_instance_id"] = profiles[0]["nfInstanceId"]
    assert StreamRelationalVerifier(POLICY).verify(profiles, witnesses) == []
    del witnesses[SLOTS[0]]
    assert StreamRelationalVerifier(POLICY).verify(profiles, witnesses) == ["missing_witness:" + SLOTS[0]]


def test_original_cu_replay_binding_and_tamper_protection():
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
        key = hashlib.sha256(b"non-production-test-key").digest()
        anchor = FreshnessAnchor(Path(directory) / "state.json")
        req = KIERequest.create("monitor", "nf", 1, POLICY.digest, "telemetry-A", key, now_ms=1000, nonce="a" * 32)
        req.verify(key, anchor, POLICY.digest, "telemetry-A", now_ms=1000)
        with pytest.raises(SecurityError):
            req.verify(key, anchor, POLICY.digest, "telemetry-A", now_ms=1000)
        with pytest.raises(SecurityError):
            req.verify(key, anchor, POLICY.digest, "telemetry-B", now_ms=1000)
        response = KIEResponse.create(req, {"verdict": "PASS"}, key, now_ms=1000)
        response.verify(key, req, now_ms=1000)
        response.report["verdict"] = "FAIL"
        with pytest.raises(SecurityError):
            response.verify(key, req, now_ms=1000)


def test_ml_cannot_suppress_hard_violation_even_when_unavailable():
    hard = np.array([True, False, True, False])
    unknown = np.array([False, True, False, False])
    assert np.array_equal(fuse(hard, unknown, ~hard, model_available=False), hard)
    assert fuse(hard, unknown, ~hard).all()


@pytest.mark.parametrize("mode", ["uniform", "oldest", "ml"])
def test_exact_budget_and_mandatory_coverage(mode):
    rng = np.random.default_rng(7654)
    selected = schedule(rng.random((200, len(SLOTS))), mode)
    assert (selected.sum(axis=1) == 2).all()
    for t in range(200 - 13):
        assert selected[t:t + 13].any(axis=0).all()


def test_scheduler_does_not_see_current_or_future_risk():
    risk = np.random.default_rng(100).random((200, len(SLOTS)))
    original = schedule(risk, "ml")
    changed = risk.copy()
    changed[80:] = -1e9
    assert np.array_equal(original[:81], schedule(changed, "ml")[:81])


@pytest.mark.parametrize("mode", ["ewma", "cusum"])
def test_temporal_statistic_uses_no_future_windows(mode):
    z = np.random.default_rng(999).normal(size=(100, 4, 6))
    changed = z.copy()
    changed[50:] += 1000
    assert np.array_equal(sequential(z, mode)[:50], sequential(changed, mode)[:50])


def test_calibration_ties_do_not_force_false_positives():
    assert threshold(np.zeros(100), .01) == 0
    assert not (np.zeros(100) > threshold(np.zeros(100), .01)).any()


def test_negative_control_is_observationally_indistinguishable():
    attacked = run(887, "a", attacks=True)
    normal = run(887, "b", attacks=False)
    for event in attacked.events:
        if event["kind"] == "stealth":
            sl = np.s_[event["start"]:event["end"], event["nf"]]
            assert np.array_equal(attacked.x[sl], normal.x[sl])
            assert not attacked.hard[sl].any()


def test_vendored_code_matches_frozen_original_blob():
    from hybrid.fetch_data import git_blob_sha
    root = Path(__file__).resolve().parents[1] / "hybrid" / "vendor"
    assert git_blob_sha((root / "stage39_kie.py").read_bytes()) == "050eb43dbb4a4fb0eecaa5b81ef65ab130c8dc28"
    assert git_blob_sha((root / "policy.json").read_bytes()) == "31050e12660afa933d422497debeab7e5ca693f3"
