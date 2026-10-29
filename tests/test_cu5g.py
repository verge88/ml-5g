import copy
from pathlib import Path
import time
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from hybrid.vendor.stage39_kie import StreamRelationalVerifier, default_profiles
from cu5g.common import POLICY, assess, native_profile, sign, validate_lifecycle, validate_witness


def keys(tmp_path):
    k = ec.generate_private_key(ec.SECP256R1())
    private, public = tmp_path/"key", tmp_path/"public"
    private.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    public.write_bytes(k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return private, public


def test_adapter_does_not_fill_missing_policy_fields():
    row = native_profile("ausf", {"nfInstanceId":"real-id", "nfType":"AUSF"})
    assert row["fqdn"] == "" and row["ipv4Addresses"] == [] and row["nfServices"] == []
    assert not POLICY.matches(row)


def test_absent_witness_does_not_confirm_integrity():
    profiles, _ = default_profiles(POLICY)
    assert assess(profiles, None, Path("absent"), {}, StreamRelationalVerifier(POLICY))["verdict"] == "UNKNOWN"


def test_witness_signature_freshness_binding_and_relations(tmp_path):
    private, public = keys(tmp_path)
    profiles, witnesses = default_profiles(POLICY)
    challenge = {"nonce":"unique", "cycle":1}
    payload = {"schema":"native-cu-runtime-witness-v1", "challenge":challenge, "issued_ns":time.time_ns(),
               "nrf_discovery_used":False, "slots":witnesses}
    bundle = sign(payload, private)
    verifier = StreamRelationalVerifier(POLICY)
    assert assess(profiles, bundle, public, challenge, verifier)["verdict"].startswith("CONFIRMED")
    changed = copy.deepcopy(profiles)
    changed[0]["nfInstanceId"] = "different"
    assert assess(changed, bundle, public, challenge, verifier)["verdict"] == "VIOLATION"
    changed = copy.deepcopy(bundle)
    changed["payload"]["slots"]["ausf"]["nf_instance_id"] = "changed"
    assert assess(profiles, changed, public, challenge, verifier)["verdict"] == "UNKNOWN"
    with pytest.raises(ValueError):
        validate_witness(bundle, public, {"nonce":"other"})
    with pytest.raises(ValueError):
        validate_witness(bundle, public, challenge, now_ns=payload["issued_ns"] + 6_000_000_000)


def test_lifecycle_authorization_is_local_and_time_bounded(tmp_path):
    private, public = keys(tmp_path)
    p = {"operation":"restart", "slot":"ausf", "policy_sha256":POLICY.digest, "issued_ns":time.time_ns()}
    b = sign(p, private)
    assert validate_lifecycle(b, public, "ausf") == p
    with pytest.raises(ValueError):
        validate_lifecycle(b, public, "nssf")
    with pytest.raises(ValueError):
        validate_lifecycle(b, public, "ausf", now_ns=p["issued_ns"] + 121_000_000_000)
