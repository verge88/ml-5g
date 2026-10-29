import argparse
import json
from pathlib import Path


def main(root, out):
    rows = [json.loads(p.read_text()) for p in sorted(root.rglob("evaluation.json"))]
    if len(rows) != 3 or {r["replica"] for r in rows} != {301,302,303}:
        raise RuntimeError("three independent deployments required")
    result = {"replicas": rows, "verified_native_cu_responses": sum(r["native_cu_responses_verified"] for r in rows),
              "real_restarts": sum(len(r["real_restarts"]) for r in rows),
              "clean_relational_alerts": sum(r["clean_relational_alerts"] for r in rows),
              "control_checks": sum(len(r["controls"]) for r in rows),
              "controls_passed": all(all(r["controls"].values()) for r in rows)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["# Native KIE transport and separately signed witness integration", "",
             f"Native NRF-delivered authenticated KIE responses: {result['verified_native_cu_responses']}",
             f"Actual NF restarts with PID and UUID changes: {result['real_restarts']}",
             f"Relational alarms in clean accepted cycles: {result['clean_relational_alerts']}",
             f"Negative controls: {result['control_checks']}; all passed: {result['controls_passed']}", "",
             "CU signing is implemented by NF-side Python agents; instrumented C NFs transport metadata in NFProfile.customInfo.",
             "Witness signatures come from a separate container with no NRF reads and no signing key mounted into the core.",
             "Runtime identity export is produced by the NF and all processes share a host kernel: not independent hardware attestation.",
             "Only two NF types. Live scenarios are benign and authorized restarts; integrity corruption changes offline copies only.",
             "This is plumbing and consistency evidence, not a measured malicious semantic attack detection rate or ML advantage."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.input, a.output)
