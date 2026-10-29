"""Separate signing process. Reads runtime records and /proc; never NRF replies."""
import json
from pathlib import Path
import socket
import time
from .common import POLICY, PORTS, atomic, runtime, sign


def collect(challenge):
    rows = {}
    for slot in POLICY.slots:
        record = runtime(slot)
        reachable = False
        try:
            with socket.create_connection(("127.0.0.1", PORTS[slot]), timeout=0.3):
                reachable = True
        except OSError:
            pass
        rows[slot] = {**record, "slot": slot, "nf_type": POLICY.slots[slot]["nf_type"],
                      "endpoint": "127.0.0.1", "direct_sbi_tcp_reachable": reachable}
    return {"schema": "native-cu-runtime-witness-v1", "challenge": challenge,
            "issued_ns": time.time_ns(), "nrf_discovery_used": False,
            "source": "separate signer: native context export + proc + direct TCP", "slots": rows}


def run():
    last = None
    while not Path("/control/done").exists():
        try:
            raw = Path("/control/witness-request.json").read_text()
            if raw != last:
                challenge = json.loads(raw)
                if challenge["policy_sha256"] != POLICY.digest:
                    raise ValueError("witness policy mismatch")
                payload = collect(challenge)
                atomic("/witness/response.json", sign(payload, "/witnesskey/witness.key"))
                last = raw
        except Exception:
            pass
        time.sleep(0.05)
    atomic("/witness/stopped.json", {"stopped": True, "issued_ns": time.time_ns()})


if __name__ == "__main__":
    run()
