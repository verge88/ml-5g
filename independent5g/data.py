"""Explicit feature whitelist; truth and identifiers never enter feature tensors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from neural5g.data import Stream, api_vector
from .collect import P, assignment

SLOTS = ("ausf", "nssf")
STATES = ("AUTHENTICATED_CURRENT_CU", "PENDING_CURRENT_CU", "UNKNOWN_BINDING_AFTER_CHANGE",
    "UNKNOWN_STALE_CU", "AUTHENTICATION_ALARM", "UNKNOWN_CU_OR_PROFILE_MISSING")


def features(observation, slot, input_name):
    if input_name not in P["inputs"]:
        raise ValueError("unknown input composition")
    values = [float(slot == s) for s in SLOTS]
    for kind in ("nrf", "direct"):
        r = observation[kind]
        status = r.get("status")
        values.extend([float(bool(r.get("ok"))), float(status is None),
            float(status or 0)/600., np.log1p(max(0., float(r.get("elapsed_ms", 0.)))),
            np.log1p(max(0., float(r.get("received_bytes", 0.))))])
    if input_name != "metadata":
        # These are the actual fixed read-only operations issued by capture.observe.
        typ = "AUSF" if slot == "ausf" else "NSSF"
        for path in ("/", f"/nnrf-disc/v1/nf-instances?target-nf-type={typ}&requester-nf-type=AMF"):
            values.extend(api_vector({"method": "GET", "path": path}).tolist())
    if input_name == "metadata_api_witness":
        m = observation.get("integrity_monitor", {})
        state = m.get("states", {}).get(slot)
        values.extend(float(state == s) for s in STATES)
        values.append(float(state not in STATES))
        values.extend(float(bool(m.get(k))) for k in ("hard_alarm", "witness_available", "confirmed"))
        verdict = m.get("decision", {}).get("verdict", "UNKNOWN")
        values.extend(float(verdict == v) for v in ("UNKNOWN", "VIOLATION", "CONFIRMED_MATCH_IN_DECLARED_SCOPE"))
        values.append(float(bool(observation.get("deployment_policy_alert"))))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite whitelisted feature")
    return result


def read_deployment(folder, replica, input_name):
    folder = Path(folder)
    manifest = json.loads((folder/"capture-manifest.json").read_text())
    expected = hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest()
    split, period = assignment(replica)
    if manifest.get("replica") != replica or manifest.get("protocol_sha256") != expected:
        raise ValueError("deployment identity/protocol mismatch")
    if manifest.get("split") != split or manifest.get("period_seconds") != period:
        raise ValueError("preregistered split/load mismatch")
    ticks = [json.loads(line) for line in (folder/"observations.jsonl").read_text().splitlines()]
    events = json.loads((folder/"events.json").read_text())
    if len(ticks) != manifest["observed_ticks"] or not ticks:
        raise ValueError("incomplete capture")
    expected_ticks = P["normal_ticks"] + (0 if split == "calibration" else
        len(P["fault_plan"])*(P["baseline_ticks"]+P["fault_ticks"]+P["recovery_ticks"]))
    if len(ticks) != expected_ticks or manifest.get("fresh_deployment") is not True:
        raise ValueError("capture does not satisfy registered duration/freshness")
    if split == "calibration" and events:
        raise ValueError("calibration must contain no interventions")
    if split != "calibration" and [(e["slot"],e["kind"]) for e in events] != [tuple(p) for p in P["fault_plan"]]:
        raise ValueError("missing or changed fault plan")
    rows, vectors = [], []
    for i, tick in enumerate(ticks):
        if tick["tick"] != i or not tick["started_ns"] <= tick["monotonic_ns"] <= tick["interval_end_ns"]:
            raise ValueError("invalid tick or observation timing")
        if i and tick["started_ns"] < ticks[i-1]["interval_end_ns"]:
            raise ValueError("overlapping or nonmonotonic intervals")
        active = [e for e in events if e["start_tick"] <= i < e["end_tick"]]
        if len(active) > 1 or bool(active) != (tick["phase"] == "fault"):
            raise ValueError("truth schedule mismatch")
        event = active[0] if active else None
        if tick["phase"] not in ("normal", "fault", "recovery") or (split == "calibration" and tick["phase"] != "normal"):
            raise ValueError("invalid or contaminated calibration phase")
        for slot in SLOTS:
            normal = tick["phase"] == "normal"
            fault = event is not None and event["slot"] == slot
            rows.append(dict(replica=replica, tick=i, slot=slot, normal=normal,
                selected=normal or fault, label=int(fault),
                episode=event["episode"] if fault else None,
                fault_start_ns=event["start_ns"] if fault else None,
                monotonic_ns=tick["monotonic_ns"],
                exposure_seconds=(tick["interval_end_ns"]-tick["started_ns"])/1e9 if normal else 0.,
                direct_ok=bool(tick["observations"][slot]["direct"]["ok"])))
            vectors.append(features(tick["observations"][slot], slot, input_name))
    x = np.stack(vectors)
    return Stream(x=x, source=np.zeros(len(rows), dtype=np.int64),
        destination=np.array([1+SLOTS.index(r["slot"]) for r in rows], dtype=np.int64),
        time=np.array([(r["monotonic_ns"]-ticks[0]["started_ns"])/1e9 for r in rows], dtype=np.float64),
        y=np.array([r["label"] for r in rows], dtype=np.float32), rows=rows, name=str(replica))


def prepare(root, input_name):
    root = Path(root)
    partitions = {}
    digests = set()
    identifiers = set()
    commits = set()
    audit = {}
    for split, ids in P["splits"].items():
        partitions[split] = []
        for replica in ids:
            folder = root/f"independent-native-{replica}"/"native"
            raw = (folder/"observations.jsonl").read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in digests:
                raise ValueError("duplicate observations across deployments")
            digests.add(digest)
            manifest = json.loads((folder/"capture-manifest.json").read_text())
            commits.add(manifest["commit"])
            ids = [profile["nfInstanceId"] for slot in SLOTS
                for profile in manifest["initial_discovery"][slot]["nrf"]["body"]["nfInstances"]]
            if len(ids) != 2 or identifiers.intersection(ids) or len(set(ids)) != 2:
                raise ValueError("initial NF UUID reused across deployments")
            identifiers.update(ids)
            partitions[split].append(read_deployment(folder, replica, input_name))
            audit[str(replica)] = dict(sha256=digest, split=split)
    if len(commits) != 1:
        raise ValueError("different source commits across deployments")
    train = partitions["train"]
    normal = np.concatenate([s.x[[r["normal"] for r in s.rows]] for s in train])
    center = np.median(normal, axis=0)
    scale = np.maximum(1.4826*np.median(abs(normal-center), axis=0), .1)
    for streams in (train, partitions["calibration"], partitions["test"]):
        for s in streams:
            s.x = np.clip((s.x-center)/scale, -20., 20.).astype(np.float32)
    return train, partitions["calibration"], partitions["test"], dict(
        input=input_name, input_dim=len(center), n_nodes=3, center=center.tolist(),
        scale=scale.tolist(), deployment_audit=audit)
