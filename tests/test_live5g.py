import copy
import pytest
from live5g.evaluate import features
from live5g.transport import validate_target
from live5g.capture import profile_policy, converged


def observation():
    return {"direct": {"ok": True, "elapsed_ms": 2},
            "nrf": {"ok": True, "status": 200, "elapsed_ms": 3,
                    "body": {"nfInstances": [{"nfType": "AUSF", "nfInstanceId": "uuid", "nfStatus": "REGISTERED"}]}}}


def test_truth_not_in_features():
    original = observation()
    altered = copy.deepcopy(original)
    altered.update({"stage": "test_fault", "attack": True, "episode": "secret"})
    assert features("ausf", original) == features("ausf", altered)


def test_missing_transport_is_observed_not_truth():
    original = observation()
    missing = copy.deepcopy(original)
    missing["direct"]["ok"] = False
    assert features("ausf", original)[0] == 0
    assert features("ausf", missing)[0] == 1


def test_no_witness_is_not_full_integrity():
    assert profile_policy("ausf", observation()) == "SCHEMA_VALID"
    unknown = observation()
    unknown["nrf"]["ok"] = False
    assert profile_policy("ausf", unknown) == "UNKNOWN"
    wrong = observation()
    wrong["nrf"]["body"]["nfInstances"][0]["nfType"] = "SMF"
    assert profile_policy("ausf", wrong) == "VIOLATION"


def test_remote_targets_rejected_before_network():
    with pytest.raises(ValueError):
        validate_target(443, "/", host="example.com")
    with pytest.raises(ValueError):
        validate_target(7777, "/", host="192.0.2.1")


def test_convergence_requires_actual_profiles():
    ausf = observation()
    nssf = copy.deepcopy(ausf)
    nssf["nrf"]["body"]["nfInstances"][0]["nfType"] = "NSSF"
    assert converged({"ausf": ausf, "nssf": nssf})
    nssf["nrf"]["body"]["nfInstances"] = []
    assert not converged({"ausf": ausf, "nssf": nssf})
