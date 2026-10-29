"""NF-side KIE signer; not a native C cryptographic implementation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
from hybrid.vendor.stage39_kie import FreshnessAnchor, KIERequest, KIEResponse, sha256_json
from .common import POLICY, atomic, runtime


def run(slot):
    folder = Path("/nf") / slot
    key = Path(f"/nfkeys/{slot}.key").read_bytes()
    anchor = FreshnessAnchor(folder / "request-anchor.json")
    last = None
    while True:
        try:
            raw = (folder / "request.json").read_text()
            if raw != last:
                import json
                request = KIERequest(**json.loads(raw))
                if request.subject_id != slot or request.controller_id != "native-controller":
                    raise ValueError("wrong request subject")
                record = runtime(slot)
                digest = sha256_json(POLICY.expected_profile(slot, record["nf_instance_id"]))
                request.verify(key, anchor, POLICY.digest, digest)
                report = {"slot": slot, "policy_sha256": POLICY.digest, "telemetry_digest": digest,
                          "runtime": record, "report_source": "NF-side agent reading native context export"}
                response = KIEResponse.create(request, report, key)
                atomic(folder / "custom_info.json", {"labKIE": {"response": asdict(response)}})
                atomic(folder / "last-response.json", asdict(response))
                last = raw
        except Exception as exc:
            print(f"{slot}: awaiting valid current runtime request: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", choices=tuple(POLICY.slots), required=True)
    run(p.parse_args().slot)
