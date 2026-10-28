"""Increasing structural complexity without exploit payloads or real targets.

Operations 0..5 are abstract procedure stages, not actual 3GPP message codes.
The paired counterfactual is generated before any controlled perturbation.
"""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import uuid
import numpy as np
from hybrid.simulation import POLICY, SLOTS, TYPES
from hybrid.vendor.stage39_kie import StreamRelationalVerifier, default_profiles, mutate

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())
N = len(SLOTS)
K = 6


@dataclass
class Trace:
    run_id: str
    x: np.ndarray
    context: np.ndarray
    tokens: np.ndarray
    destinations: np.ndarray
    phase: np.ndarray
    generation: np.ndarray
    hard: np.ndarray
    unknown: np.ndarray
    attack: np.ndarray
    behavior: np.ndarray
    integrity: np.ndarray
    event: dict | None
    regime: str


def profile_results(seed, generation, missing, integrity):
    ticks = generation.shape[0]
    hard, unknown = np.zeros_like(integrity), np.zeros_like(integrity)
    profiles0, witnesses0 = default_profiles(POLICY)
    changes = {0, ticks}
    for array in (generation, missing, integrity):
        changes.update((np.flatnonzero(np.any(array[1:] != array[:-1], axis=1)) + 1).tolist())
    bounds = sorted(changes)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        profiles, witnesses = copy.deepcopy(profiles0), copy.deepcopy(witnesses0)
        for nf, slot in enumerate(SLOTS):
            if generation[start, nf] > 0:
                uid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"complex:{seed}:{nf}:{generation[start,nf]}"))
                profiles[nf]["nfInstanceId"] = uid
                witnesses[slot]["nf_instance_id"] = uid
            if integrity[start, nf]:
                profiles = mutate(profiles, slot, "identity")
                next(p for p in profiles if p["slot"] == slot)["nfInstanceId"] = str(
                    uuid.uuid5(uuid.NAMESPACE_OID, f"invalid:{seed}:{nf}"))
            if missing[start, nf]:
                del witnesses[slot]
        for issue in StreamRelationalVerifier(POLICY).verify(profiles, witnesses):
            name, _, slot = issue.partition(":")
            if slot in SLOTS:
                nf = SLOTS.index(slot)
                if name == "missing_witness":
                    unknown[start:stop, nf] = True
                else:
                    hard[start:stop, nf] = True
            else:
                hard[start:stop] = True
    return hard, unknown


def generate(seed, run_id, kind, attacked, regime="ordinary"):
    rng = np.random.default_rng(seed)
    ticks, length = PROTOCOL["ticks"], PROTOCOL["event_length"]
    # Followers occupy odd indices; attacks on them can affect a learned relation.
    victim = int(rng.choice(np.arange(1, N - 1, 2)))
    start = int(rng.integers(72, ticks - length - 40))
    end = start + length
    phase = np.zeros((ticks, N), dtype=bool)
    generation = np.zeros((ticks, N), dtype=int)
    for nf in range(N):
        at = int(rng.integers(20, ticks - 40))
        phase[at:at + 12, nf] = True
        generation[at:, nf] = 1
    # Scope is local: a neighboring NF is legitimately updating, the victim is stable.
    if kind == "lifecycle_abuse":
        phase[:, victim] = False
        phase[24:36, victim] = True
        generation[:, victim] = 0
        generation[24:, victim] = 1
        other = (victim + 2) % N
        phase[start:start + 12, other] = True
        generation[start:, other] += 1
    global_load = np.zeros(ticks)
    for t in range(1, ticks):
        global_load[t] = .9 * global_load[t - 1] + rng.normal(0, .15)
    load = np.clip(1.4 + global_load[:, None] + rng.normal(0, .05, (ticks, N)), .2, 3)
    if regime == "load_change":
        load[ticks // 2:] += .7
    role = np.array([TYPES.index(POLICY.slots[s]["nf_type"]) for s in SLOTS])
    context = np.concatenate([np.broadcast_to(np.eye(len(TYPES))[role], (ticks, N, len(TYPES))),
                              load[..., None], phase[..., None].astype(float)], axis=2)
    driver = rng.normal(size=(ticks, N))
    for t in range(1, ticks):
        driver[t] = .65 * driver[t - 1] + np.sqrt(1 - .65 ** 2) * driver[t]
    noise = rng.normal(size=(ticks, N, 6))
    mean = np.stack(np.broadcast_arrays(
        20 + 8 * load + role[None, :],
        20 + 8 * load + role[None, :],
        .02 + .08 * phase,
        25 + 6 * load + 15 * phase,
        3 + .1 * role[None, :] + phase,
        .03 + .08 * phase), axis=2)
    residual = np.stack([driver, driver + .08 * noise[:, :, 1], noise[:, :, 2],
                         noise[:, :, 3], noise[:, :, 4], noise[:, :, 5]], axis=2)
    x = mean + residual * np.array([2.5, 2.5, .01, 2., .25, .02])
    tokens = np.zeros((ticks, N), dtype=int)
    for nf in range(0, N, 2):
        state = int(rng.integers(K))
        for t in range(ticks):
            if phase[t, nf]:
                state = 0 if rng.random() < .3 else state
            elif rng.random() < .82:
                state = (state + 1) % K
            tokens[t, nf] = state
        if nf + 1 < N:
            tokens[0, nf + 1] = tokens[0, nf]
            tokens[1:, nf + 1] = tokens[:-1, nf]
            for t in range(ticks):
                if phase[t, nf + 1]:
                    tokens[t, nf + 1] = int(rng.integers(K))
    # Sparse normal transport duplication is included in every split.
    repeat = rng.random((ticks, N)) < .008
    for t in range(1, ticks):
        tokens[t, repeat[t]] = tokens[t - 1, repeat[t]]
    destinations = np.empty((ticks, N), dtype=int)
    for nf in range(N):
        peer = nf - 1 if nf % 2 else (nf + 1) % N
        destinations[:, nf] = peer
        for t in np.flatnonzero(phase[:, nf]):
            destinations[t, nf] = (peer + 2 + 2 * int(rng.integers(2))) % N
    missing = np.zeros((ticks, N), dtype=bool)
    for nf in range(N):
        if rng.random() < .2:
            at = int(rng.integers(20, ticks - 20))
            missing[at:at + 6, nf] = True
    behavior = np.zeros((ticks, N), dtype=bool)
    integrity = np.zeros_like(behavior)
    attack = np.zeros_like(behavior)
    event = None
    if attacked:
        attack[start:end, victim] = True
        if kind in {"integrity_only", "combined"}:
            integrity[start:end, victim] = True
        if kind not in {"integrity_only", "invisible"}:
            behavior[start:end, victim] = True
        if kind == "joint_shuffle":
            x[start:end, victim, 1] = x[start:end, victim, 1][::-1]
        elif kind in {"order_permutation", "combined"}:
            values = tokens[start:end, victim].copy()
            for lo in range(0, length, 8):
                values[lo:lo + 8] = rng.permutation(values[lo:lo + 8])
            tokens[start:end, victim] = values
        elif kind == "cross_nf_desync":
            tokens[start:end, victim] = np.roll(tokens[start:end, victim], 7)
        elif kind == "sparse_reorder":
            original = tokens[start:end, victim].copy()
            for offset in (5, 13, 21):
                lo = start + offset
                tokens[lo:lo + 2, victim] = tokens[lo:lo + 2, victim][::-1]
            if np.array_equal(tokens[start:end, victim], original):
                changes = np.flatnonzero(original[1:] != original[:-1])
                if len(changes):
                    lo = start + int(changes[0])
                    tokens[lo:lo + 2, victim] = tokens[lo:lo + 2, victim][::-1]
        elif kind == "lifecycle_abuse":
            tokens[start:start + 12, victim] = 0
        elif kind == "edge_rebinding":
            destinations[start:end, victim] = (victim + 6) % N
        event = {"run_id": run_id, "kind": kind, "nf": victim, "start": start, "end": end,
                 "observable": kind != "invisible", "has_behavior": kind not in {"integrity_only", "invisible"},
                 "has_integrity": kind in {"integrity_only", "combined"}}
    hard, unknown = profile_results(seed, generation, missing, integrity)
    return Trace(run_id, x, context, tokens, destinations, phase, generation, hard, unknown,
                 attack, behavior, integrity, event, regime)


def child_seed(seed, namespace, index):
    return int(np.random.SeedSequence([seed, namespace, index]).generate_state(1)[0])


def datasets(seed):
    normals = [generate(child_seed(seed, 1, i), f"{seed}-normal-{i}", PROTOCOL["kinds"][i % 9], False,
                        "load_change" if i % 3 == 0 else "ordinary")
               for i in range(PROTOCOL["normal_training_runs"])]
    train = list(normals)
    known = [k for k in PROTOCOL["kinds"] if k not in PROTOCOL["heldout_kinds"] + ["invisible"]]
    for i, kind in enumerate(known):
        for rep in range(PROTOCOL["training_pairs_per_known_kind"]):
            root = child_seed(seed, 2, i * 100 + rep)
            train += [generate(root, f"{seed}-train-{kind}-{rep}-{flag}", kind, flag) for flag in (False, True)]
    calibration = [generate(child_seed(seed, 3, i), f"{seed}-cal-{i}", PROTOCOL["kinds"][i % 9], False,
                            "load_change" if i % 3 == 0 else "ordinary")
                   for i in range(PROTOCOL["calibration_normal_runs"])]
    test = []
    for i, kind in enumerate(PROTOCOL["kinds"]):
        for rep in range(PROTOCOL["test_pairs_per_kind"]):
            root = child_seed(seed, 4, i * 100 + rep)
            regime = "load_change" if rep % 3 == 0 else "ordinary"
            test += [generate(root, f"{seed}-test-{kind}-{rep}-{flag}", kind, flag, regime) for flag in (False, True)]
    return normals, train, calibration, test
