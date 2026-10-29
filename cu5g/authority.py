"""Independent lifecycle signing helper, run outside the NF container."""
import argparse
import json
from pathlib import Path
import time
from .common import POLICY, atomic, sign


def run(folder, key):
    last, sequence = None, 0
    while not (folder / "done").exists():
        try:
            raw = (folder / "lifecycle-request.json").read_text()
            if raw != last:
                request = json.loads(raw)
                if request["slot"] not in POLICY.slots or request["operation"] != "restart":
                    raise ValueError("only declared local NF restart allowed")
                sequence += 1
                payload = {**request, "sequence": sequence, "issued_ns": time.time_ns(), "policy_sha256": POLICY.digest}
                atomic(folder / "lifecycle-response.json", sign(payload, key))
                last = raw
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.05)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    a = p.parse_args()
    run(a.control, a.key)
