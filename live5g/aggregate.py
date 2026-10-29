import argparse
import csv
import json
from pathlib import Path


def main(root, out):
    evaluations = [json.loads(p.read_text()) for p in sorted(root.rglob("evaluation.json"))]
    if len(evaluations) != 3 or {x["manifest"]["replica"] for x in evaluations} != {201, 202, 203}:
        raise RuntimeError("all three independent live deployments required")
    out.mkdir(parents=True, exist_ok=True)
    comparison = []
    for name in evaluations[0]["models"]:
        models = [r["models"][name] for r in evaluations]
        events = [e for m in models for e in m["episodes"]]
        total, false = sum(m["normal_points"] for m in models), sum(m["false_points"] for m in models)
        comparison.append({"model": name, "detected_episodes": sum(e["detected"] for e in events),
                           "total_episodes": len(events), "false_points": false, "normal_points": total,
                           "normal_point_fpr": false / total,
                           "stop_detected": sum(e["detected"] for e in events if e["kind"] == "stop"),
                           "pause_detected": sum(e["detected"] for e in events if e["kind"] == "pause")})
    with (out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0]))
        w.writeheader()
        w.writerows(comparison)
    (out / "summary.json").write_text(json.dumps({"replicas": evaluations, "comparison": comparison}, indent=2))
    lines = ["# Real Open5GS availability integration", "", "Three independent native NRF/AUSF/NSSF deployments; read-only HTTP/2 observations.",
             "Native binary source is unmodified. All faults are local container lifecycle operations.", "",
             "|Model|Detected availability episodes|Normal point false alarms|", "|---|---:|---:|"]
    for r in comparison:
        lines.append(f"|{r['model']}|{r['detected_episodes']}/{r['total_episodes']}|{r['false_points']}/{r['normal_points']} ({100*r['normal_point_fpr']:.3f}%)|")
    lines += ["", "This tests availability and telemetry integration, not adversarial semantic detection.",
              "Train stop and held-out pause share a missing-response symptom; pause is not a novel attack family.",
              "GET / checks transport responsiveness; an HTTP 404 is not a successful authentication procedure.",
              "Independent official registration-suite results are in the native-registration artifact; ML is not evaluated on that suite.",
              "Full original CU/witness verdict is UNKNOWN because unmodified images do not expose those inputs.",
              "Normal calibration is empirical, small and correlated. No 1% test or anytime false-alarm guarantee.",
              "Three deployments and 12 availability episodes cannot establish operational detection quality or scientific novelty."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.input, a.output)
