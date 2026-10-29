import base64
import json
import os
from pathlib import Path
import socket
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from hybrid.vendor.stage39_kie import canonical_bytes, FullProfilePolicy, normalize_profile

ROOT = Path(__file__).resolve().parent
POLICY = FullProfilePolicy.load(ROOT / "policy.json")
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
PORTS = {"ausf": 7778, "nssf": 7779}


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    os.replace(tmp, path)


def native_profile(slot, row):
    """Map actual standard fields, without substituting expected policy values."""
    if slot not in POLICY.slots:
        raise ValueError("unknown local slot")
    services = []
    for service in row.get("nfServices", []):
        scheme = service.get("scheme", "")
        services.append({"serviceName": service.get("serviceName", ""), "scheme": scheme,
                         "status": service.get("nfServiceStatus", ""),
                         "versions": [v.get("apiVersionInUri", "") for v in service.get("versions", [])],
                         "endpoints": [{"ip": e.get("ipv4Address", e.get("ipv6Address", "")),
                                        "port": e.get("port", 443 if scheme == "https" else 80),
                                        "transport": e.get("transport", "TCP")} for e in service.get("ipEndPoints", [])]})
    return normalize_profile({"slot": slot, "nfInstanceId": row.get("nfInstanceId", ""),
                              "nfType": row.get("nfType", ""), "nfStatus": row.get("nfStatus", ""),
                              "fqdn": row.get("fqdn", ""), "ipv4Addresses": row.get("ipv4Addresses", []),
                              "nfServices": services})


def runtime(slot):
    found = []
    for p in Path("/runtime").glob("*.tsv"):
        try:
            pid_s, enum, uid = p.read_text().strip().split("\t")
            pid = int(pid_s)
            parts = [x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
            if parts[0].endswith(f"open5gs-{slot}d") and any(x.endswith(f"/{slot}.yaml") for x in parts):
                found.append({"pid": pid, "nf_instance_id": uid.lower(), "binary": parts[0],
                              "nf_enum": int(enum), "process_alive": True})
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            continue
    if len(found) != 1:
        raise RuntimeError("runtime identity absent or ambiguous: " + slot)
    return found[0]


def sign(payload, key_path):
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    return {"payload": payload, "signature": base64.b64encode(key.sign(canonical_bytes(payload), ec.ECDSA(hashes.SHA256()))).decode()}


def verified(bundle, public_path):
    key = serialization.load_pem_public_key(Path(public_path).read_bytes())
    key.verify(base64.b64decode(bundle["signature"], validate=True), canonical_bytes(bundle["payload"]), ec.ECDSA(hashes.SHA256()))
    return bundle["payload"]


def validate_witness(bundle, public_path, challenge, now_ns=None):
    p = verified(bundle, public_path)
    if p.get("schema") != "native-cu-runtime-witness-v1" or p.get("challenge") != challenge:
        raise ValueError("witness binding mismatch")
    if p.get("nrf_discovery_used") is not False:
        raise ValueError("witness source not independent of discovery")
    now = time.time_ns() if now_ns is None else now_ns
    if abs(now - p.get("issued_ns", 0)) > PROTOCOL["witness_max_age_seconds"] * 1e9:
        raise ValueError("stale witness")
    if set(p.get("slots", {})) != set(POLICY.slots):
        raise ValueError("incomplete witness")
    return p["slots"]


def validate_lifecycle(bundle, public_path, slot, now_ns=None):
    p = verified(bundle, public_path)
    now = time.time_ns() if now_ns is None else now_ns
    if p.get("operation") != "restart" or p.get("slot") != slot or p.get("policy_sha256") != POLICY.digest:
        raise ValueError("lifecycle scope mismatch")
    if abs(now - p.get("issued_ns", 0)) > PROTOCOL["lifecycle_max_age_seconds"] * 1e9:
        raise ValueError("stale lifecycle authorization")
    return p


def assess(profiles, bundle, public_path, challenge, verifier):
    if bundle is None:
        return {"verdict": "UNKNOWN", "reason": "missing_witness", "alerts": []}
    try:
        rows = validate_witness(bundle, public_path, challenge)
    except Exception as exc:
        return {"verdict": "UNKNOWN", "reason": "invalid_witness", "authentication_alarm": True,
                "error": type(exc).__name__, "alerts": []}
    alerts = verifier.verify(profiles, rows)
    return {"verdict": "VIOLATION" if alerts else "CONFIRMED_MATCH_IN_DECLARED_SCOPE", "alerts": alerts}
