"""Synthetic observable-generating mechanism; no claim of protocol fidelity.

ML receives x and causal context only. Attack truth is kept separately.
The original Stage 39 verifier processes actual profile/witness dictionaries.
"""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import uuid
import numpy as np
from .vendor.stage39_kie import FullProfilePolicy, StreamRelationalVerifier, default_profiles, mutate

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())
POLICY = FullProfilePolicy.load(Path(__file__).parent / "vendor" / "policy.json")
SLOTS = sorted(POLICY.slots)
TYPES = sorted({v["nf_type"] for v in POLICY.slots.values()})
FEATURES = ["request_rate", "error_ratio", "retry_ratio", "latency", "fanout", "sequence_deviation"]
TRAIN_KINDS = ["integrity", "fanout", "burst", "mixed"]
TEST_KINDS = ["integrity", "fanout", "burst", "mixed", "sequence", "stealth"]


@dataclass
class Run:
    run_id: str
    x: np.ndarray
    context: np.ndarray
    y: np.ndarray
    hard: np.ndarray
    unknown: np.ndarray
    phase: np.ndarray
    events: list
    regime: str


def fuse(hard, unknown, model_alarm, model_available=True):
    """Never allow an absent/negative model to suppress a verified violation."""
    behavioral = np.asarray(model_alarm, dtype=bool) & np.asarray(model_available, dtype=bool)
    return np.asarray(hard, dtype=bool) | behavioral


def run(seed, run_id, attacks=False, training=False, drift=False, lifecycle=True):
    rng = np.random.default_rng(seed)
    ticks, n = PROTOCOL["ticks"], len(SLOTS)
    phase = np.zeros((ticks, n))
    restart_at = {}
    if lifecycle:
        for i in range(n):
            at = int(rng.integers(24, ticks - 36))
            restart_at[i] = at
            phase[at:at + 12, i] = 1
    role_ids = np.array([TYPES.index(POLICY.slots[s]["nf_type"]) for s in SLOTS])
    onehot = np.broadcast_to(np.eye(len(TYPES))[role_ids], (ticks, n, len(TYPES)))
    global_load = np.zeros(ticks)
    for t in range(1, ticks):
        global_load[t] = .92 * global_load[t - 1] + rng.normal(0, .12)
    load = np.clip(1.5 + global_load[:, None] + rng.normal(0, .08, (ticks, n)), .2, 3)
    context = np.concatenate([onehot, load[..., None], phase[..., None]], axis=2)
    r = role_ids[None, :]
    means = np.stack(np.broadcast_arrays(
        20 + 10 * load + r,
        .015 + .055 * phase,
        .025 + .12 * phase,
        25 + 7 * load + 22 * phase,
        2.5 + .15 * r + 2 * phase,
        .04 + .12 * phase), axis=2)
    scale = np.array([3., .012, .025, 3., .35, .025])
    noise = rng.standard_t(6, size=(ticks, n, 6)) / np.sqrt(1.5)
    for t in range(1, ticks):
        noise[t] = .35 * noise[t - 1] + np.sqrt(1 - .35 ** 2) * noise[t]
    x = means + scale * noise
    # Benign distribution shift is deliberately held out from training/calibration.
    if drift:
        x[ticks // 2:, :, 3] += 5
        x[ticks // 2:, :, 2] += .025
    y = np.zeros((ticks, n), dtype=bool)
    kinds = np.full((ticks, n), "", dtype="U16")
    events = []
    if attacks:
        choices = TRAIN_KINDS if training else TEST_KINDS
        for j in range(8):
            nf = int(rng.integers(n))
            start = 36 + (j // 4) * 80 + (j % 4) * 2
            end = start + PROTOCOL["event_length_ticks"]
            # Keep complete attack episodes disjoint for each NF.
            while y[start:end, nf].any():
                nf = (nf + 1) % n
            kind = choices[j % len(choices)]
            y[start:end, nf] = True
            kinds[start:end, nf] = kind
            if kind in {"fanout", "mixed"}:
                x[start:end, nf, 4] += rng.uniform(1, 2)
                x[start:end, nf, 5] += .06
            if kind in {"burst", "mixed"}:
                x[start:end, nf, 0] += rng.uniform(10, 18)
                x[start:end, nf, 2] += .08
            if kind == "sequence":
                x[start:end, nf, 5] += np.linspace(.02, .18, end - start)
            events.append({"run_id": run_id, "nf": nf, "start": start, "end": end, "kind": kind})
    profiles0, witnesses0 = default_profiles(POLICY)
    hard = np.zeros_like(y)
    unknown = np.zeros_like(y)
    # Missing evidence is observable, independent of attack assignment.
    missing = np.zeros_like(y)
    for i in range(n):
        if rng.random() < .25:
            at = int(rng.integers(16, ticks - 16))
            missing[at:at + 8, i] = True
    for t in range(ticks):
        profiles, witnesses = copy.deepcopy(profiles0), copy.deepcopy(witnesses0)
        verifier = StreamRelationalVerifier(POLICY)
        for i, at in restart_at.items():
            if t >= at:
                uid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{seed}:{i}:generation2"))
                profiles[i]["nfInstanceId"] = uid
                witnesses[SLOTS[i]]["nf_instance_id"] = uid
        # Authorized restarts preserve every unrelated policy constraint.
        # The instantaneous generation swap is an abstraction, not a live restart.
        for i, slot in enumerate(SLOTS):
            if kinds[t, i] in {"integrity", "mixed"}:
                profiles = mutate(profiles, slot, "identity")
                next(p for p in profiles if p["slot"] == slot)["nfInstanceId"] = str(
                    uuid.uuid5(uuid.NAMESPACE_OID, f"spoof:{seed}:{i}"))
            if missing[t, i]:
                del witnesses[slot]
        issues = verifier.verify(profiles, witnesses)
        for issue in issues:
            name, _, slot = issue.partition(":")
            if slot in SLOTS:
                i = SLOTS.index(slot)
                if name == "missing_witness":
                    unknown[t, i] = True
                else:
                    hard[t, i] = True
            else:
                # A global uniqueness violation cannot be localized by this API.
                hard[t] = True
    return Run(run_id, x, context, y, hard, unknown, phase, events,
               "drift" if drift else ("lifecycle" if lifecycle else "stable"))


def split(seed, part):
    """Disjoint SeedSequence namespaces and whole runs; no random row split."""
    blocks = {"normal": 10, "attack": 20, "calibration": 30, "test": 40}
    counts = {"normal": PROTOCOL["train_normal_runs"], "attack": PROTOCOL["train_attack_runs"],
              "calibration": PROTOCOL["calibration_normal_runs"], "test": PROTOCOL["test_runs"]}
    runs = []
    for i in range(counts[part]):
        child = int(np.random.SeedSequence([seed, blocks[part], i]).generate_state(1)[0])
        attacked = part == "attack" or (part == "test" and i >= 10)
        runs.append(run(child, f"{seed}-{part}-{i}", attacks=attacked, training=part == "attack",
                        drift=part == "test" and i % 5 == 4, lifecycle=i % 4 != 0))
    return runs
