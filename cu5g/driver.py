"""Benign native transport, restart and offline-copy integrity controls."""
import asyncio
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from hybrid.vendor.stage39_kie import (FreshnessAnchor, KIERequest, KIEResponse, SecurityError,
                                       StreamRelationalVerifier, sha256_json)
from live5g.transport import request
from .common import (ROOT, POLICY, PROTOCOL, atomic, native_profile, runtime,
                     validate_lifecycle, validate_witness, assess)

OUT = Path("/out")
PROCS = {}
EVIDENCE = []


def start(slot):
    env = dict(os.environ, STAGE38_WITNESS_DIR="/runtime")
    if slot != "nrf":
        env["STAGE39_KIE_CUSTOM_INFO_PATH"] = f"/nf/{slot}/custom_info.json"
    log = (OUT / f"{slot}-native.log").open("a")
    p = subprocess.Popen([f"/src/open5gs/build/src/{slot}/open5gs-{slot}d", "-c", str(ROOT / f"{slot}.yaml")], env=env, stdout=log, stderr=subprocess.STDOUT)
    PROCS[slot] = p
    return p


def stop(slot):
    p = PROCS[slot]
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=3)


async def discovery(slot):
    typ = POLICY.slots[slot]["nf_type"]
    r = await request(7777, f"/nnrf-disc/v1/nf-instances?target-nf-type={typ}&requester-nf-type=AMF", timeout=2)
    if r["status"] != 200 or not isinstance(r["body"], dict):
        raise RuntimeError("discovery unavailable: " + json.dumps(r))
    return r["body"].get("nfInstances", [])


async def ready():
    last = None
    for _ in range(160):
        try:
            found = {}
            for slot in POLICY.slots:
                rows = await discovery(slot)
                if len(rows) != 1:
                    raise ValueError("wrong profile count")
                actual = native_profile(slot, rows[0])
                rec = runtime(slot)
                if not POLICY.matches(actual) or actual["nfInstanceId"] != rec["nf_instance_id"]:
                    raise ValueError("native profile does not match independent deployment policy: " + json.dumps(actual))
                found[slot] = rows[0]
            return found
        except Exception as exc:
            last = repr(exc)
        await asyncio.sleep(0.25)
    raise RuntimeError("native readiness failed: " + str(last))


async def witness(cycle):
    challenge = {"cycle": cycle, "nonce": secrets.token_hex(16), "policy_sha256": POLICY.digest}
    atomic("/control/witness-request.json", challenge)
    for _ in range(100):
        try:
            bundle = json.loads(Path("/witness/response.json").read_text())
            rows = validate_witness(bundle, "/public/witness.pub", challenge)
            return challenge, bundle, rows
        except Exception:
            await asyncio.sleep(0.05)
    raise RuntimeError("separate witness did not deliver fresh evidence")


async def authorization(slot):
    req = {"operation": "restart", "slot": slot, "nonce": secrets.token_hex(16)}
    atomic("/control/lifecycle-request.json", req)
    for _ in range(100):
        try:
            bundle = json.loads(Path("/control/lifecycle-response.json").read_text())
            payload = validate_lifecycle(bundle, "/public/lifecycle.pub", slot)
            if payload["nonce"] == req["nonce"]:
                return bundle
        except Exception:
            pass
        await asyncio.sleep(0.05)
    raise RuntimeError("independent lifecycle authorization unavailable")


async def cycle(number, label, verifier, anchor):
    raw = await ready()
    requests, keys = {}, {}
    for slot in POLICY.slots:
        keys[slot] = Path(f"/nfkeys/{slot}.key").read_bytes()
        digest = sha256_json(native_profile(slot, raw[slot]))
        req = KIERequest.create("native-controller", slot, number, POLICY.digest, digest, keys[slot])
        requests[slot] = req
        atomic(f"/nf/{slot}/request.json", asdict(req))
    accepted, old_cache = {}, 0
    deadline = time.monotonic() + PROTOCOL["cu_wait_seconds"]
    while len(accepted) != len(POLICY.slots) and time.monotonic() < deadline:
        for slot in POLICY.slots:
            if slot in accepted:
                continue
            try:
                rows = await discovery(slot)
                if len(rows) != 1:
                    continue
                profile = rows[0]
                response = KIEResponse(**profile.get("customInfo", {}).get("labKIE", {})["response"])
                response.verify(keys[slot], requests[slot])
                report = response.report
                if report["policy_sha256"] != POLICY.digest or report["telemetry_digest"] != requests[slot].telemetry_digest:
                    raise ValueError("KIE report binding mismatch")
                normalized = native_profile(slot, profile)
                if sha256_json(normalized) != requests[slot].telemetry_digest:
                    raise ValueError("discovered profile digest mismatch")
                if report["runtime"]["nf_instance_id"] != normalized["nfInstanceId"]:
                    raise ValueError("NF-agent runtime identity mismatch")
                accepted[slot] = {"profile": profile, "response": asdict(response), "request": asdict(requests[slot])}
            except SecurityError:
                old_cache += 1
            except (KeyError, TypeError, ValueError):
                pass
        if len(accepted) != len(POLICY.slots):
            await asyncio.sleep(0.15)
    if len(accepted) != len(POLICY.slots):
        raise RuntimeError("fresh KIE missing from native NRF discovery: " + json.dumps({"accepted": list(accepted), "last": await ready()}))
    challenge, bundle, witness_rows = await witness(number)
    profiles = [native_profile(s, accepted[s]["profile"]) for s in POLICY.slots]
    decision = assess(profiles, bundle, "/public/witness.pub", challenge, verifier)
    alerts = decision["alerts"]
    if decision["verdict"] != "CONFIRMED_MATCH_IN_DECLARED_SCOPE":
        raise RuntimeError("fresh witness did not confirm native state: " + str(decision))
    if alerts:
        raise RuntimeError("clean native full-profile verification failed: " + str(alerts))
    for slot, req in requests.items():
        anchor.check_and_commit(req.controller_id, req.subject_id, req.cycle, req.nonce, req.issued_at_ms, int(time.time()*1000))
    result = {"cycle": number, "label": label, "accepted": accepted, "challenge": challenge,
              "witness": bundle, "alerts": alerts, "verdict": "CONFIRMED_MATCH_IN_DECLARED_SCOPE",
              "stale_cached_cu_rejections_while_waiting": old_cache}
    EVIDENCE.append(result)
    atomic(OUT / "live-cycles.json", EVIDENCE)
    print(json.dumps({"cycle": number, "label": label, "native_cu_count": len(accepted), "alerts": alerts, "cached_old_rejected": old_cache}), flush=True)
    return result


def controls(current, verifier, anchor):
    passed = {}
    slot = "ausf"
    req = KIERequest(**current["accepted"][slot]["request"])
    resp = KIEResponse(**current["accepted"][slot]["response"])
    key = Path("/nfkeys/ausf.key").read_bytes()
    def rejected(call):
        try:
            call()
        except Exception:
            return True
        return False
    passed["cu_mac"] = rejected(lambda: replace(resp, mac="0"*64).verify(key, req))
    passed["cu_binding"] = rejected(lambda: resp.verify(key, replace(req, nonce=secrets.token_hex(16))))
    passed["cu_replay"] = rejected(lambda: anchor.check_and_commit(req.controller_id, req.subject_id, req.cycle, req.nonce, req.issued_at_ms, int(time.time()*1000)))
    altered = copy.deepcopy(current["witness"])
    altered["payload"]["slots"][slot]["nf_instance_id"] = "offline-corrupted-copy"
    passed["witness_signature"] = rejected(lambda: validate_witness(altered, "/public/witness.pub", current["challenge"]))
    other = dict(current["challenge"], nonce=secrets.token_hex(16))
    passed["witness_replay"] = rejected(lambda: validate_witness(current["witness"], "/public/witness.pub", other))
    rows = current["witness"]["payload"]["slots"]
    profiles = [native_profile(s, current["accepted"][s]["profile"]) for s in POLICY.slots]
    changed = copy.deepcopy(profiles)
    changed[0]["nfInstanceId"] = "00000000-0000-0000-0000-000000000000"
    identity_alerts = verifier.verify(changed, rows)
    passed["offline_profile_identity"] = "witness_uuid:ausf" in identity_alerts
    changed = copy.deepcopy(profiles)
    changed[0]["nfServices"][0]["serviceName"] = "offline-invalid-service"
    service_alerts = verifier.verify(changed, rows)
    passed["offline_profile_service"] = "semantic_mismatch:ausf" in service_alerts
    return passed, {"identity": identity_alerts, "service": service_alerts, "changes_sent_to_core": False}


async def run():
    OUT.mkdir(parents=True, exist_ok=True)
    agents = []
    anchor = FreshnessAnchor(OUT / "controller-response-anchor.json")
    verifier = StreamRelationalVerifier(POLICY)
    lifecycle = []
    try:
        start("nrf")
        await asyncio.sleep(0.4)
        for slot in POLICY.slots:
            start(slot)
            log = (OUT / f"{slot}-agent.log").open("a")
            agents.append(subprocess.Popen([sys.executable, "-m", "cu5g.agent", "--slot", slot], stdout=log, stderr=subprocess.STDOUT))
        await ready()
        await cycle(1, "initial", verifier, anchor)
        last = await cycle(2, "steady", verifier, anchor)
        for number, slot in enumerate(PROTOCOL["restart_slots"], 3):
            auth = await authorization(slot)
            old = runtime(slot)
            verifier.begin_transition("restart", slot)
            stop(slot)
            start(slot)
            await ready()
            new = runtime(slot)
            if old["pid"] == new["pid"] or old["nf_instance_id"] == new["nf_instance_id"]:
                raise RuntimeError("real PID and runtime UUID rotation required")
            last = await cycle(number, f"post-{slot}-restart", verifier, anchor)
            verifier.end_transition()
            lifecycle.append({"slot": slot, "authorization": auth, "old": old, "new": new})
        tests, details = controls(last, verifier, anchor)
        tests["lifecycle_scope"] = False
        try:
            validate_lifecycle(lifecycle[0]["authorization"], "/public/lifecycle.pub", "nssf")
        except Exception:
            tests["lifecycle_scope"] = True
        # Actual witness channel outage: remove response and do not issue a request.
        Path("/witness/response.json").unlink()
        missing_actual = not Path("/witness/response.json").exists()
        profiles = [native_profile(s, last["accepted"][s]["profile"]) for s in POLICY.slots]
        outcome = assess(profiles, None if missing_actual else {}, "/public/witness.pub", last["challenge"], verifier)
        outage = {"source": "actual local witness response removed; signer receives no new challenge",
                  **outcome, "signature_bypassed": False}
        tests["missing_witness"] = missing_actual and outage["verdict"] == "UNKNOWN"
        last = await cycle(5, "witness-recovery", verifier, anchor)
        if set(tests) != set(PROTOCOL["offline_controls"]) or not all(tests.values()):
            raise RuntimeError("control failed: " + json.dumps(tests))
        vendor = Path("hybrid/vendor/stage39_kie.py").read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(vendor)).encode() + b"\0" + vendor).hexdigest()
        if blob != PROTOCOL["vendor_git_blob"]:
            raise RuntimeError("original verifier was changed")
        report = {"replica": int(os.environ["REPLICA"]), "commit": os.environ["SOURCE_COMMIT"],
                  "open5gs_commit": PROTOCOL["open5gs_commit"], "verifier_git_blob": blob,
                  "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
                  "policy_sha256": POLICY.digest, "native_cu_responses_verified": sum(len(c["accepted"]) for c in EVIDENCE),
                  "live_cycles": len(EVIDENCE), "real_restarts": lifecycle,
                  "controls": tests, "offline_details": details, "witness_outage": outage,
                  "clean_relational_alerts": sum(len(c["alerts"]) for c in EVIDENCE),
                  "scope": "two NF types, benign native CU transport and restarts; offline corrupted-copy controls",
                  "trust_limits": ["NF-side Python agent signs; C NF forwards metadata", "runtime UUID export is NF-produced", "same host kernel", "no native malicious response mutation", "no production traffic or end-to-end UE test in this series"]}
        atomic(OUT / "evaluation.json", report)
        Path("/tmp/instrumentation.patch").replace(OUT / "instrumentation.patch")
        print(json.dumps({"responses": report["native_cu_responses_verified"], "restarts": len(lifecycle), "controls": tests}), flush=True)
    finally:
        Path("/control/done").touch()
        for _ in range(60):
            if Path("/witness/stopped.json").exists():
                break
            await asyncio.sleep(0.05)
        for p in agents:
            p.terminate()
            p.wait(timeout=3)
        for slot in reversed(list(PROCS)):
            stop(slot)


if __name__ == "__main__":
    asyncio.run(run())
