"""Read-only loopback monitoring plus declared local stop/pause fault schedule."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

P = json.loads(Path(__file__).with_name("protocol.json").read_text())


def assignment(replica):
    found = [(name, ids.index(replica)) for name, ids in P["splits"].items() if replica in ids]
    if len(found) != 1:
        raise ValueError("deployment is not preregistered")
    name, index = found[0]
    return name, P["period_seconds"][index]


async def capture_run(args):
    from live5g import capture
    split, period = assignment(args.replica)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    events, tick = [], 0
    stream = (output / "observations.jsonl").open("w", encoding="utf-8")

    async def sample(phase, count, episode=None):
        nonlocal tick
        for _ in range(count):
            start = time.monotonic_ns()
            observations = await capture.observe()
            observed = time.monotonic_ns()
            await asyncio.sleep(max(0., period - (observed-start)/1e9))
            end = time.monotonic_ns()
            stream.write(json.dumps(dict(tick=tick, phase=phase, episode=episode,
                started_ns=start, monotonic_ns=observed, interval_end_ns=end,
                observations=observations)) + "\n")
            stream.flush()
            tick += 1

    try:
        initial = await capture.wait_ready()
        await sample("normal", P["normal_ticks"])
        if split != "calibration":
            for index, (slot, kind) in enumerate(P["fault_plan"]):
                await capture.wait_ready()
                episode = f"fault-{index}"
                await sample("normal", P["baseline_ticks"])
                if kind == "stop":
                    capture.command("stop", "-t", "1", slot)
                else:
                    capture.command("pause", slot)
                begin, first = time.monotonic_ns(), tick
                await sample("fault", P["fault_ticks"], episode)
                ended = time.monotonic_ns()
                capture.command("start" if kind == "stop" else "unpause", slot)
                events.append(dict(episode=episode, slot=slot, kind=kind,
                    start_tick=first, end_tick=tick, start_ns=begin, end_ns=ended))
                await sample("recovery", P["recovery_ticks"])
                await capture.wait_ready()
        (output / "events.json").write_text(json.dumps(events, indent=2))
        manifest = dict(replica=args.replica, split=split, period_seconds=period,
            observed_ticks=tick, initial_discovery=initial,
            protocol_sha256=hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest(),
            scope=P["scope"], profiles_mutated=False, fresh_deployment=True)
        # native_hook adds the source commit and actual CU provenance afterwards.
        (output / "capture-manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        stream.close()
        for slot in capture.TARGETS:
            capture.command("unpause", slot)


async def main():
    from live5g import capture
    from fullsuite import native_hook
    assignment(int(os.environ["REPLICA"]))
    capture.run = capture_run
    await native_hook.run()


if __name__ == "__main__":
    asyncio.run(main())
