"""Actual native NRF replies and direct SBI liveness during local container faults.

All target names and destinations are fixed. Event truth is stored separately
and is not used in features. GET / is a transport check, not a service success.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from .transport import request

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "protocol.json").read_text())
COMPOSE = ["docker", "compose", "-p", "ml5g-live", "-f", str(ROOT / "compose.lock.yaml")]
TARGETS = {"ausf": ("AUSF", 7778), "nssf": ("NSSF", 7779)}


def command(*args):
    return subprocess.run([*COMPOSE, *args], check=True, capture_output=True, text=True, timeout=60).stdout


def discover_path(typ):
    return f"/nnrf-disc/v1/nf-instances?target-nf-type={typ}&requester-nf-type=AMF"


async def observe():
    observations = {}
    for slot, (typ, port) in TARGETS.items():
        nrf, direct = await asyncio.gather(request(7777, discover_path(typ)), request(port, "/"))
        observations[slot] = {"nrf": nrf, "direct": direct}
    return observations


def converged(row):
    for slot, (typ, _) in TARGETS.items():
        r = row[slot]
        body = r["nrf"]["body"]
        profiles = body.get("nfInstances", []) if isinstance(body, dict) else []
        if not r["direct"]["ok"] or r["nrf"]["status"] != 200:
            return False
        if len(profiles) != 1 or profiles[0].get("nfType") != typ or not profiles[0].get("nfInstanceId"):
            return False
    return True


async def wait_ready():
    last = None
    for _ in range(90):
        last = await observe()
        if converged(last):
            return last
        await asyncio.sleep(0.5)
    raise RuntimeError("native NRF discovery / direct SBI did not converge: " + json.dumps(last))


def profile_policy(slot, observation):
    """Limited schema policy, not the original full CU/witness verifier."""
    nrf = observation["nrf"]
    if not nrf["ok"] or nrf["status"] != 200 or not isinstance(nrf["body"], dict):
        return "UNKNOWN"
    rows = nrf["body"].get("nfInstances", [])
    if not rows:
        return "UNAVAILABLE"
    if any(r.get("nfType") != TARGETS[slot][0] or not r.get("nfInstanceId") for r in rows):
        return "VIOLATION"
    return "SCHEMA_VALID"


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    stream = args.output.joinpath("observations.jsonl").open("w", encoding="utf-8")
    events, lifecycle, tick = [], [], 0
    async def sample(stage, count, episode=None):
        nonlocal tick
        for _ in range(count):
            started = time.monotonic()
            obs = await observe()
            item = {"tick": tick, "monotonic_ns": time.monotonic_ns(), "unix_ns": time.time_ns(),
                    "stage": stage, "episode": episode, "observations": obs,
                    "profile_policy": {s: profile_policy(s, obs[s]) for s in TARGETS}}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            tick += 1
            await asyncio.sleep(max(0, P["period_seconds"] - (time.monotonic() - started)))
    try:
        initial = await wait_ready()
        await sample("fit_normal", P["fit_ticks"])
        await sample("calibration_normal", P["calibration_ticks"])
        for split, plan in (("train", P["training_faults"]), ("test", P["test_faults"])):
            if split == "test":
                await wait_ready()
                await sample("test_normal", P["normal_test_ticks"])
            for index, (slot, kind) in enumerate(plan):
                episode = f"{split}-{index}-{slot}-{kind}"
                await wait_ready()
                await sample(split + "_baseline", P["baseline_ticks"], episode)
                before = await observe()
                old_id = before[slot]["nrf"]["body"]["nfInstances"][0]["nfInstanceId"]
                lifecycle.append({"episode": episode, "operation": kind, "slot": slot,
                                  "authorized_by": "local experiment controller", "issued_ns": time.monotonic_ns(),
                                  "old_nf_instance_id": old_id})
                command("stop", "-t", "1", slot) if kind == "stop" else command("pause", slot)
                began_ns, began_tick = time.monotonic_ns(), tick
                await sample(split + "_fault", P["fault_ticks"], episode)
                ended_ns = time.monotonic_ns()
                command("start", slot) if kind == "stop" else command("unpause", slot)
                events.append({"episode": episode, "split": split, "slot": slot, "kind": kind,
                               "start_tick": began_tick, "end_tick": tick,
                               "start_ns": began_ns, "end_ns": ended_ns})
                await sample(split + "_recovery", P["recovery_ticks"], episode)
                recovered = await wait_ready()
                new_id = recovered[slot]["nrf"]["body"]["nfInstances"][0]["nfInstanceId"]
                lifecycle[-1].update({"new_nf_instance_id": new_id, "uuid_changed": old_id != new_id,
                                      "recovery_verified_ns": time.monotonic_ns()})
        (args.output / "events.json").write_text(json.dumps(events, indent=2))
        (args.output / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2))
        manifest = {"replica": args.replica, "commit": os.environ.get("GITHUB_SHA", "local"),
                    "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
                    "observed_ticks": tick, "initial_discovery": initial,
                    "final_discovery": await wait_ready(), "profiles_mutated": False,
                    "full_cu_witness_status": "UNKNOWN: native images have no original CU/witness integration"}
        (args.output / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        stream.close()
        # Ensure a failed capture does not leave paused local lab containers.
        for slot in TARGETS:
            subprocess.run([*COMPOSE, "unpause", slot], capture_output=True, timeout=15)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--replica", type=int, choices=P["replicas"], required=True)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(run(p.parse_args()))
